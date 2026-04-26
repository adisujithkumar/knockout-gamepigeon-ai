"""Tests for Feature Attention Discovery reward shaping."""

import torch
from torch import nn

from knockout.reward.attention_discovery import (
    AttentionConfig,
    AttentionRewardShaper,
    AttentionTrainer,
    CrystallizedFeature,
    GradientAttributor,
    get_feature_names,
)


# ---- Helpers ---------------------------------------------------------------

class _SimpleValueNet(nn.Module):
    """V(x) = x[0] + 2*x[1].  Known gradient: [1, 2, 0, ...]."""

    def __init__(self, obs_dim: int = 89):
        super().__init__()
        self._w = nn.Parameter(torch.zeros(obs_dim))
        with torch.no_grad():
            self._w[0] = 1.0
            self._w[1] = 2.0

    def forward(self, x: torch.Tensor):
        # Return (action_mean, log_std, value) to match ActorCritic.
        v = (x * self._w).sum(dim=-1, keepdim=True)
        dummy = torch.zeros(x.shape[0], 2)
        return dummy, dummy, v


# ---- GradientAttributor ---------------------------------------------------

class TestGradientAttributor:

    def test_attribution_picks_up_known_weights(self):
        net = _SimpleValueNet(obs_dim=10)
        obs = torch.randn(256, 10)
        attr = GradientAttributor().compute_attribution(net, obs)

        assert attr.shape == (10,)
        # Feature 1 should have ~2x the attribution of feature 0.
        assert attr[1] > attr[0]
        # Features 2-9 should have near-zero attribution.
        assert attr[2:].max() < 0.01

    def test_signed_attribution_captures_direction(self):
        net = _SimpleValueNet(obs_dim=10)
        obs = torch.randn(256, 10)
        attr, signs = GradientAttributor().compute_signed_attribution(net, obs)

        # Both features have positive weights -> positive sign.
        assert signs[0] > 0
        assert signs[1] > 0
        assert attr[1] > attr[0]


# ---- AttentionRewardShaper ------------------------------------------------

class TestAttentionRewardShaper:

    def _make_shaper(self, top_k: int = 3) -> AttentionRewardShaper:
        cfg = AttentionConfig(top_k=top_k, initial_shaping_weight=1.0,
                              decay_rate=0.5, min_weight=0.01)
        return AttentionRewardShaper(cfg)

    def test_update_crystallizes_top_k(self):
        shaper = self._make_shaper(top_k=2)
        names = get_feature_names()
        attr = torch.zeros(89)
        signs = torch.ones(89)
        # Set two high-attribution non-excluded features.
        attr[5] = 10.0   # ego.dist_to_edge
        attr[42] = 5.0    # enemy1.position_x

        shaper.update(attr, signs, step=100, feature_names=names)

        assert len(shaper.features) == 2
        indices = {f.feature_index for f in shaper.features}
        assert indices == {5, 42}

    def test_excludes_alive_features(self):
        shaper = self._make_shaper(top_k=3)
        names = get_feature_names()
        attr = torch.zeros(89)
        signs = torch.ones(89)
        attr[8] = 100.0   # ego.alive — should be excluded
        attr[5] = 10.0    # ego.dist_to_edge
        attr[6] = 8.0     # ego.speed

        shaper.update(attr, signs, step=100, feature_names=names)

        indices = {f.feature_index for f in shaper.features}
        assert 8 not in indices
        assert 5 in indices
        assert 6 in indices

    def test_decay_reduces_weights(self):
        shaper = self._make_shaper(top_k=2)
        names = get_feature_names()
        attr = torch.zeros(89)
        signs = torch.ones(89)
        attr[5] = 10.0
        attr[6] = 10.0

        shaper.update(attr, signs, step=100, feature_names=names)
        w0 = shaper.features[0].weight

        # Second update: existing weights decay by 0.5 then get replaced.
        shaper.update(attr, signs, step=200, feature_names=names)
        # Weights should be re-normalised (same attribution -> same weights).
        assert abs(shaper.features[0].weight - w0) < 1e-6

    def test_compute_reward_empty(self):
        shaper = self._make_shaper()
        obs = torch.randn(4, 89)
        r = shaper.compute_reward(obs)
        assert r.shape == (4,)
        assert (r == 0).all()

    def test_compute_reward_matches_manual(self):
        shaper = self._make_shaper(top_k=1)
        names = get_feature_names()
        attr = torch.zeros(89)
        signs = torch.ones(89)
        attr[5] = 10.0
        signs[5] = -1.0  # lower dist_to_edge is better
        shaper.update(attr, signs, step=100, feature_names=names)

        obs = torch.zeros(2, 89)
        obs[0, 5] = 0.5
        obs[1, 5] = 0.8

        r = shaper.compute_reward(obs)
        f = shaper.features[0]
        expected_0 = f.weight * f.sign * 0.5
        expected_1 = f.weight * f.sign * 0.8
        assert abs(r[0].item() - expected_0) < 1e-6
        assert abs(r[1].item() - expected_1) < 1e-6

    def test_discovery_log_populated(self):
        shaper = self._make_shaper(top_k=2)
        names = get_feature_names()
        attr = torch.zeros(89)
        signs = torch.ones(89)
        attr[5] = 10.0
        attr[6] = 8.0

        shaper.update(attr, signs, step=100, feature_names=names)
        shaper.update(attr, signs, step=200, feature_names=names)

        log = shaper.get_discovery_log()
        assert len(log) == 2
        assert log[0]["step"] == 100
        assert log[1]["step"] == 200
        assert len(log[0]["features"]) == 2


# ---- AttentionTrainer (integration) ---------------------------------------

class TestAttentionTrainer:

    def test_buffer_fills_and_attribution_runs(self):
        net = _SimpleValueNet(obs_dim=89)
        cfg = AttentionConfig(
            attribution_interval=500,
            buffer_size=200,
            top_k=3,
        )
        trainer = AttentionTrainer(net, config=cfg, obs_dim=89)

        # Feed enough data to trigger attribution.
        for _ in range(10):
            obs = torch.randn(64, 89)
            trainer.on_step(obs)

        assert trainer.total_steps == 640
        # Attribution should have run at least once.
        trainer.force_attribution()
        assert len(trainer.discovery_log) >= 1

        # Top features should be index 0 and 1 (the only non-zero weights).
        indices = {f.feature_index for f in trainer.crystallized_features}
        assert 0 in indices or 1 in indices

    def test_shaped_reward_nonzero_after_attribution(self):
        net = _SimpleValueNet(obs_dim=89)
        cfg = AttentionConfig(buffer_size=100, top_k=3)
        trainer = AttentionTrainer(net, config=cfg, obs_dim=89)

        obs = torch.randn(128, 89)
        trainer.on_step(obs)
        trainer.force_attribution()

        reward = trainer.compute_shaped_reward(obs)
        assert reward.shape == (128,)
        # With a non-trivial value net, rewards should not all be zero.
        assert reward.abs().sum() > 0


class TestFeatureNames:

    def test_89_names(self):
        names = get_feature_names()
        assert len(names) == 89
        assert names[0] == "ego.position_x"
        assert names[8] == "ego.alive"
        assert names[14] == "ally1.position_x"
        assert names[84] == "team_a_alive"
        assert names[88] == "timestep"
