"""Tests for reward co-evolution module."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from knockout.reward.coevolution import (
    CoevolutionConfig,
    EvolutionSnapshot,
    RewardCoevolution,
)
from knockout.reward.contrastive import FEATURE_NAMES


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_trajectory_data(
    n_steps: int = 2000,
    effect_indices: list[int] | None = None,
    effect_magnitude: float = 3.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create synthetic trajectory data with known effects on given features."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n_steps, 89)).astype(np.float32)
    outcomes = np.where(rng.random(n_steps) > 0.5, 1.0, -1.0).astype(np.float32)

    if effect_indices:
        win_mask = outcomes > 0
        for idx in effect_indices:
            obs[win_mask, idx] += effect_magnitude

    return {
        "observations": obs,
        "outcomes": outcomes,
        "episode_ids": np.arange(n_steps, dtype=np.int64),
    }


# --------------------------------------------------------------------------
# analyze_and_update
# --------------------------------------------------------------------------

class TestAnalyzeAndUpdate:
    def test_discovers_injected_features(self):
        """Features with large effect sizes should be discovered."""
        config = CoevolutionConfig(
            min_effect_size=0.3,
            max_reward_features=10,
            feature_momentum=0.0,  # no blending, pure new
        )
        coev = RewardCoevolution(config)

        # Inject effect on feature 5 (ego.dist_to_edge)
        data = _make_trajectory_data(effect_indices=[5])
        report = coev.analyze_and_update(data, current_elo=1000.0, current_step=0)

        assert report["num_active_features"] > 0
        assert "ego.dist_to_edge" in report["new_features"]
        assert report["elo_at_discovery"] == 1000.0

    def test_returns_empty_report_on_insufficient_data(self):
        """Too few steps should produce an empty report."""
        config = CoevolutionConfig()
        coev = RewardCoevolution(config)

        data = {
            "observations": np.zeros((2, 89), dtype=np.float32),
            "outcomes": np.array([1.0, -1.0], dtype=np.float32),
        }
        report = coev.analyze_and_update(data, current_elo=1000.0, current_step=0)

        assert report["num_active_features"] == 0
        assert report["note"] == "insufficient data"

    def test_history_grows_with_each_call(self):
        """Each successful analysis should add a snapshot."""
        config = CoevolutionConfig(feature_momentum=0.0, min_effect_size=0.3)
        coev = RewardCoevolution(config)

        for step in range(3):
            data = _make_trajectory_data(effect_indices=[5], seed=step + 10)
            coev.analyze_and_update(data, current_elo=1000 + step * 50,
                                    current_step=step * 1000)

        assert len(coev.history) == 3
        assert coev.history[0].elo == 1000.0
        assert coev.history[2].elo == 1100.0


# --------------------------------------------------------------------------
# EMA blending
# --------------------------------------------------------------------------

class TestEMABlending:
    def test_momentum_preserves_old_features(self):
        """With high momentum, old features should persist even when new
        analysis does not rediscover them."""
        config = CoevolutionConfig(
            feature_momentum=0.9,
            min_effect_size=0.3,
        )
        coev = RewardCoevolution(config)

        # Step 1: discover feature 5
        data1 = _make_trajectory_data(effect_indices=[5], seed=1)
        coev.analyze_and_update(data1, current_elo=1000, current_step=0)

        weight_after_first = coev.current_weights.get(5, 0.0)
        assert abs(weight_after_first) > 0

        # Step 2: discover feature 10 instead (feature 5 not significant)
        data2 = _make_trajectory_data(effect_indices=[10], seed=2)
        coev.analyze_and_update(data2, current_elo=1050, current_step=1000)

        # Feature 5 should still be present (decayed by momentum * old)
        weight_5 = coev.current_weights.get(5, 0.0)
        assert abs(weight_5) > 0, "high momentum should preserve old feature"
        # Feature 5 weight should have decayed
        assert abs(weight_5) < abs(weight_after_first)

    def test_zero_momentum_replaces_completely(self):
        """With zero momentum, old features not rediscovered should vanish."""
        config = CoevolutionConfig(
            feature_momentum=0.0,
            min_effect_size=0.3,
        )
        coev = RewardCoevolution(config)

        # Step 1: feature 5
        data1 = _make_trajectory_data(effect_indices=[5], seed=1)
        coev.analyze_and_update(data1, current_elo=1000, current_step=0)
        assert 5 in coev.current_weights

        # Step 2: feature 10 only (no effect on 5)
        data2 = _make_trajectory_data(effect_indices=[10], seed=2)
        coev.analyze_and_update(data2, current_elo=1050, current_step=1000)

        # Feature 5 should be gone (0 momentum, new weight is 0 -> pruned)
        assert 5 not in coev.current_weights

    def test_ema_blend_values(self):
        """Verify the EMA formula: new = momentum * old + (1 - momentum) * raw."""
        config = CoevolutionConfig(
            feature_momentum=0.7,
            min_effect_size=0.1,  # low threshold
        )
        coev = RewardCoevolution(config)

        # Manually set weights to a known state
        coev.current_weights = {5: 1.0}
        coev._feature_first_seen = {5: 0}
        coev._feature_last_seen = {5: 0}

        # Create data where feature 5 has a known effect size ~2.0
        data = _make_trajectory_data(
            effect_indices=[5], effect_magnitude=2.0, seed=42,
        )
        coev.analyze_and_update(data, current_elo=1000, current_step=100)

        # The blended weight for feature 5 should be:
        # 0.7 * 1.0 + 0.3 * (new_effect_size_for_5)
        w5 = coev.current_weights.get(5, 0.0)
        # It should be different from both the old (1.0) and pure new
        assert w5 != 1.0
        assert abs(w5) > 0


# --------------------------------------------------------------------------
# Feature shift detection
# --------------------------------------------------------------------------

class TestFeatureShift:
    def test_shift_is_zero_when_unchanged(self):
        """No change in weights -> zero shift."""
        shift = RewardCoevolution._compute_shift(
            {5: 1.0, 10: -0.5},
            {5: 1.0, 10: -0.5},
        )
        assert shift == pytest.approx(0.0)

    def test_shift_matches_l2_distance(self):
        """Shift should equal L2 distance between weight vectors."""
        old = {5: 1.0, 10: 0.0}
        new = {5: 0.0, 10: 1.0}
        shift = RewardCoevolution._compute_shift(old, new)
        # sqrt(1^2 + 1^2) = sqrt(2)
        assert shift == pytest.approx(np.sqrt(2.0), abs=1e-6)

    def test_report_contains_feature_shift(self):
        """analyze_and_update should report a non-zero shift when features change."""
        config = CoevolutionConfig(feature_momentum=0.0, min_effect_size=0.3)
        coev = RewardCoevolution(config)

        data1 = _make_trajectory_data(effect_indices=[5], seed=1)
        report1 = coev.analyze_and_update(data1, current_elo=1000, current_step=0)

        data2 = _make_trajectory_data(effect_indices=[10], seed=2)
        report2 = coev.analyze_and_update(data2, current_elo=1050, current_step=1000)

        assert report2["feature_shift"] > 0


# --------------------------------------------------------------------------
# Reward computation
# --------------------------------------------------------------------------

class TestComputeReward:
    def test_output_shape_2d(self):
        """(B, 89) input -> (B,) reward."""
        config = CoevolutionConfig(feature_momentum=0.0, min_effect_size=0.3)
        coev = RewardCoevolution(config)

        data = _make_trajectory_data(effect_indices=[5])
        coev.analyze_and_update(data, current_elo=1000, current_step=0)

        obs = torch.randn(8, 89)
        reward = coev.compute_reward(obs)
        assert reward.shape == (8,)

    def test_output_shape_3d(self):
        """(B, 3, 89) input -> (B, 3) reward."""
        config = CoevolutionConfig(feature_momentum=0.0, min_effect_size=0.3)
        coev = RewardCoevolution(config)

        data = _make_trajectory_data(effect_indices=[5])
        coev.analyze_and_update(data, current_elo=1000, current_step=0)

        obs = torch.randn(4, 3, 89)
        reward = coev.compute_reward(obs)
        assert reward.shape == (4, 3)

    def test_zero_reward_before_any_analysis(self):
        """Before any analysis, reward should be zero."""
        coev = RewardCoevolution(CoevolutionConfig())
        obs = torch.randn(4, 89)
        reward = coev.compute_reward(obs)
        assert (reward == 0).all()


# --------------------------------------------------------------------------
# Save / load roundtrip
# --------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        """Save and load should restore identical state."""
        config = CoevolutionConfig(feature_momentum=0.5, min_effect_size=0.3)
        coev = RewardCoevolution(config)

        # Build up some history
        data1 = _make_trajectory_data(effect_indices=[5], seed=1)
        coev.analyze_and_update(data1, current_elo=1000, current_step=0)

        data2 = _make_trajectory_data(effect_indices=[5, 10], seed=2)
        coev.analyze_and_update(data2, current_elo=1050, current_step=1000)

        # Save
        save_path = tmp_path / "coev_state.json"
        coev.save(save_path)

        # Load into fresh instance
        coev2 = RewardCoevolution(config)
        coev2.load(save_path)

        # Verify state matches
        assert len(coev2.history) == len(coev.history)
        assert coev2.current_weights == pytest.approx(coev.current_weights)
        assert coev2._feature_first_seen == coev._feature_first_seen
        assert coev2._feature_last_seen == coev._feature_last_seen

        # Verify reward computation matches
        obs = torch.randn(4, 89)
        r1 = coev.compute_reward(obs)
        r2 = coev2.compute_reward(obs)
        assert torch.allclose(r1, r2)

    def test_saved_file_is_valid_json(self, tmp_path):
        """Saved file should be parseable JSON."""
        config = CoevolutionConfig(feature_momentum=0.0, min_effect_size=0.3)
        coev = RewardCoevolution(config)

        data = _make_trajectory_data(effect_indices=[5])
        coev.analyze_and_update(data, current_elo=1000, current_step=0)

        save_path = tmp_path / "coev.json"
        coev.save(save_path)

        with open(save_path) as f:
            state = json.load(f)

        assert "current_weights" in state
        assert "history" in state
        assert len(state["history"]) == 1


# --------------------------------------------------------------------------
# Evolution timeline
# --------------------------------------------------------------------------

class TestEvolutionTimeline:
    def test_timeline_structure(self):
        """Timeline entries should have the expected keys."""
        config = CoevolutionConfig(feature_momentum=0.0, min_effect_size=0.3)
        coev = RewardCoevolution(config)

        data = _make_trajectory_data(effect_indices=[5], seed=1)
        coev.analyze_and_update(data, current_elo=1000, current_step=0)

        timeline = coev.get_evolution_timeline()
        assert len(timeline) == 1

        entry = timeline[0]
        assert entry["step"] == 0
        assert entry["elo"] == 1000.0
        assert isinstance(entry["features"], list)
        assert isinstance(entry["note"], str)
        assert "feature_shift" in entry
        assert "feature_lifetimes" in entry

    def test_timeline_is_json_serializable(self):
        """Timeline should be serializable to JSON without errors."""
        config = CoevolutionConfig(feature_momentum=0.5, min_effect_size=0.3)
        coev = RewardCoevolution(config)

        for i in range(3):
            indices = [5] if i < 2 else [5, 10]
            data = _make_trajectory_data(effect_indices=indices, seed=i + 10)
            coev.analyze_and_update(data, current_elo=1000 + i * 50,
                                    current_step=i * 1000)

        timeline = coev.get_evolution_timeline()
        # This should not raise
        serialized = json.dumps(timeline, indent=2)
        parsed = json.loads(serialized)
        assert len(parsed) == 3

    def test_feature_lifetimes(self):
        """Feature lifetimes should track first/last seen correctly."""
        config = CoevolutionConfig(feature_momentum=0.0, min_effect_size=0.3)
        coev = RewardCoevolution(config)

        data1 = _make_trajectory_data(effect_indices=[5], seed=1)
        coev.analyze_and_update(data1, current_elo=1000, current_step=0)

        data2 = _make_trajectory_data(effect_indices=[5], seed=2)
        coev.analyze_and_update(data2, current_elo=1050, current_step=5000)

        lifetimes = coev.get_feature_lifetimes()
        assert "ego.dist_to_edge" in lifetimes
        lt = lifetimes["ego.dist_to_edge"]
        assert lt["first_seen"] == 0
        assert lt["last_seen"] == 5000
        assert lt["duration"] == 5000
