"""Tests for contrastive trajectory mining reward discovery."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from knockout.reward.contrastive import (
    FEATURE_NAMES,
    ContrastiveAnalyzer,
    ContrastiveConfig,
    ContrastiveRewardShaper,
    ContrastiveTrainer,
    DiscoveryResult,
    TrajectoryCollector,
)


# --------------------------------------------------------------------------
# Feature names
# --------------------------------------------------------------------------

class TestFeatureNames:
    def test_length(self):
        assert len(FEATURE_NAMES) == 89

    def test_ego_prefix(self):
        assert FEATURE_NAMES[0] == "ego.position_x"
        assert FEATURE_NAMES[13] == "ego.dist_to_ego"

    def test_enemy_prefix(self):
        assert FEATURE_NAMES[42] == "enemy1.position_x"

    def test_global_suffix(self):
        assert FEATURE_NAMES[84] == "global.team_a_alive"
        assert FEATURE_NAMES[88] == "global.timestep"


# --------------------------------------------------------------------------
# TrajectoryCollector
# --------------------------------------------------------------------------

class TestTrajectoryCollector:
    def test_shapes_and_outcomes(self):
        """Collector returns correct shapes with +1/-1 outcome labels."""
        # Use a mock env that finishes quickly
        env = _MockEnv(num_envs=4, rounds_to_done=3)
        collector = TrajectoryCollector()

        data = collector.collect(env, policy=None, num_episodes=8)

        assert data["observations"].ndim == 2
        assert data["observations"].shape[1] == 89
        assert data["outcomes"].shape == (data["observations"].shape[0],)
        assert data["episode_ids"].shape == (data["observations"].shape[0],)

        # All outcomes should be +1 or -1
        unique = set(np.unique(data["outcomes"]))
        assert unique.issubset({-1.0, 1.0})

    def test_episode_ids_monotonic(self):
        """Episode IDs should be non-decreasing."""
        env = _MockEnv(num_envs=2, rounds_to_done=2)
        collector = TrajectoryCollector()
        data = collector.collect(env, policy=None, num_episodes=4)

        ids = data["episode_ids"]
        # Each episode's IDs form a contiguous block
        unique_ids = np.unique(ids)
        assert len(unique_ids) == 4


# --------------------------------------------------------------------------
# ContrastiveAnalyzer
# --------------------------------------------------------------------------

class TestContrastiveAnalyzer:
    def test_synthetic_known_effect(self):
        """With a known large effect on one feature, analyzer finds it."""
        rng = np.random.default_rng(42)
        n = 2000

        # 89-dim observations, mostly noise
        obs = rng.standard_normal((n, 89)).astype(np.float32)
        outcomes = np.where(rng.random(n) > 0.5, 1.0, -1.0).astype(np.float32)

        # Inject large effect on feature 5 (ego.dist_to_edge):
        # winners have higher values
        win_mask = outcomes > 0
        obs[win_mask, 5] += 2.0  # large shift -> Cohen's d >> 0.3

        config = ContrastiveConfig(min_effect_size=0.3, max_reward_features=5)
        analyzer = ContrastiveAnalyzer(config)
        result = analyzer.analyze(obs, outcomes)

        assert 5 in result.feature_indices
        idx_pos = result.feature_indices.index(5)
        # Effect should be positive (higher in wins)
        assert result.directions[idx_pos] > 0
        assert abs(result.effect_sizes[idx_pos]) > 0.3

    def test_no_effect_returns_empty(self):
        """Pure noise should discover nothing (with high threshold)."""
        rng = np.random.default_rng(123)
        n = 500
        obs = rng.standard_normal((n, 89)).astype(np.float32)
        outcomes = np.where(rng.random(n) > 0.5, 1.0, -1.0).astype(np.float32)

        config = ContrastiveConfig(min_effect_size=1.0, max_reward_features=5)
        analyzer = ContrastiveAnalyzer(config)
        result = analyzer.analyze(obs, outcomes)

        # With high threshold and pure noise, should find very few or none
        assert len(result.feature_indices) <= 2

    def test_respects_max_features(self):
        """Should cap at max_reward_features even with many significant."""
        rng = np.random.default_rng(7)
        n = 5000
        obs = rng.standard_normal((n, 89)).astype(np.float32)
        outcomes = np.where(rng.random(n) > 0.5, 1.0, -1.0).astype(np.float32)
        win_mask = outcomes > 0

        # Inject large effects on 15 features
        for i in range(15):
            obs[win_mask, i * 5] += 3.0

        config = ContrastiveConfig(min_effect_size=0.1, max_reward_features=5)
        analyzer = ContrastiveAnalyzer(config)
        result = analyzer.analyze(obs, outcomes)

        assert len(result.feature_indices) <= 5

    def test_too_few_samples(self):
        """Should return empty result with insufficient data."""
        obs = np.zeros((1, 89), dtype=np.float32)
        outcomes = np.array([1.0], dtype=np.float32)

        config = ContrastiveConfig()
        analyzer = ContrastiveAnalyzer(config)
        result = analyzer.analyze(obs, outcomes)

        assert len(result.feature_indices) == 0

    def test_feature_names_populated(self):
        """Discovered features should have correct names."""
        rng = np.random.default_rng(42)
        n = 1000
        obs = rng.standard_normal((n, 89)).astype(np.float32)
        outcomes = np.where(rng.random(n) > 0.5, 1.0, -1.0).astype(np.float32)
        obs[outcomes > 0, 5] += 3.0  # ego.dist_to_edge

        config = ContrastiveConfig(min_effect_size=0.3, max_reward_features=5)
        analyzer = ContrastiveAnalyzer(config)
        result = analyzer.analyze(obs, outcomes)

        if 5 in result.feature_indices:
            idx_pos = result.feature_indices.index(5)
            assert result.feature_names[idx_pos] == "ego.dist_to_edge"


# --------------------------------------------------------------------------
# ContrastiveRewardShaper
# --------------------------------------------------------------------------

class TestContrastiveRewardShaper:
    def test_empty_result_zero_reward(self):
        """Empty discovery -> zero shaped reward."""
        config = ContrastiveConfig()
        shaper = ContrastiveRewardShaper(config)
        shaper.update(DiscoveryResult([], [], [], []))

        obs = torch.randn(4, 3, 89)
        reward = shaper.compute_reward(obs)
        assert reward.shape == (4, 3)
        assert (reward == 0).all()

    def test_single_feature_positive_direction(self):
        """Positive direction should reward high feature values."""
        config = ContrastiveConfig(reward_scale=1.0)
        shaper = ContrastiveRewardShaper(config)

        result = DiscoveryResult(
            feature_indices=[5],
            effect_sizes=[1.0],
            directions=[1.0],
            feature_names=["ego.dist_to_edge"],
        )
        shaper.update(result)

        # Observation with high value at index 5
        obs = torch.zeros(2, 89)
        obs[0, 5] = 1.0   # high dist_to_edge
        obs[1, 5] = -1.0  # low dist_to_edge

        reward = shaper.compute_reward(obs)
        assert reward[0] > reward[1]

    def test_reward_scale_normalization(self):
        """Weights should be normalized to sum to reward_scale."""
        config = ContrastiveConfig(reward_scale=0.5)
        shaper = ContrastiveRewardShaper(config)

        result = DiscoveryResult(
            feature_indices=[0, 5],
            effect_sizes=[2.0, -1.0],
            directions=[1.0, -1.0],
            feature_names=["ego.position_x", "ego.dist_to_edge"],
        )
        shaper.update(result)

        # With features at 1.0, max possible shaped reward = reward_scale
        obs = torch.ones(1, 89)
        reward = shaper.compute_reward(obs)
        # Sum of |weights| = 0.5, so with all-ones input
        # reward = w0 * 1.0 + w1 * 1.0 where w0 = 2/3 * 0.5, w1 = -1/3 * 0.5
        expected = (2.0 / 3.0 * 0.5) + (-1.0 / 3.0 * 0.5)
        assert abs(reward.item() - expected) < 1e-5

    def test_3d_input(self):
        """Should handle (B, 3, 89) input returning (B, 3)."""
        config = ContrastiveConfig(reward_scale=1.0)
        shaper = ContrastiveRewardShaper(config)

        result = DiscoveryResult(
            feature_indices=[5],
            effect_sizes=[1.0],
            directions=[1.0],
            feature_names=["ego.dist_to_edge"],
        )
        shaper.update(result)

        obs = torch.randn(4, 3, 89)
        reward = shaper.compute_reward(obs)
        assert reward.shape == (4, 3)

    def test_num_features(self):
        config = ContrastiveConfig()
        shaper = ContrastiveRewardShaper(config)
        assert shaper.num_features == 0

        result = DiscoveryResult(
            feature_indices=[0, 5, 10],
            effect_sizes=[1.0, 0.5, 0.3],
            directions=[1.0, -1.0, 1.0],
            feature_names=["a", "b", "c"],
        )
        shaper.update(result)
        assert shaper.num_features == 3


# --------------------------------------------------------------------------
# DiscoveryResult
# --------------------------------------------------------------------------

class TestDiscoveryResult:
    def test_to_dict(self):
        result = DiscoveryResult(
            feature_indices=[5],
            effect_sizes=[1.23],
            directions=[1.0],
            feature_names=["ego.dist_to_edge"],
        )
        d = result.to_dict()
        assert len(d["features"]) == 1
        assert d["features"][0]["index"] == 5
        assert d["features"][0]["name"] == "ego.dist_to_edge"

    def test_summary(self):
        result = DiscoveryResult(
            feature_indices=[5],
            effect_sizes=[1.23],
            directions=[1.0],
            feature_names=["ego.dist_to_edge"],
        )
        s = result.summary()
        assert "ego.dist_to_edge" in s
        assert "higher" in s


# --------------------------------------------------------------------------
# Full loop integration (small scale)
# --------------------------------------------------------------------------

class TestContrastiveTrainerLoop:
    def test_one_iteration_no_crash(self):
        """Full loop with tiny config runs 1 iteration without error."""
        env = _MockEnv(num_envs=4, rounds_to_done=2)
        ppo = _MockPPO()

        config = ContrastiveConfig(
            collection_episodes=8,
            min_effect_size=0.1,
            max_reward_features=3,
            reward_scale=0.5,
            num_iterations=1,
            steps_per_iteration=100,
        )

        trainer = ContrastiveTrainer(
            env=env,
            ppo_trainer=ppo,
            config=config,
            log_dir="/tmp/test_contrastive_log",
        )

        metrics = trainer.run()
        assert len(metrics) == 1
        assert "iteration" in metrics[0]
        assert "discovery" in metrics[0]


# --------------------------------------------------------------------------
# Mock helpers
# --------------------------------------------------------------------------

class _MockEnv:
    """Minimal mock of TensorVecEnv for testing."""

    def __init__(self, num_envs: int = 4, rounds_to_done: int = 3):
        self.num_envs = num_envs
        self._rounds_to_done = rounds_to_done
        self._round = 0

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self._round = 0
        obs = np.random.randn(self.num_envs, 3, 89).astype(np.float32)
        masks = np.ones((self.num_envs, 3), dtype=bool)
        return obs, masks

    def step(self, actions: np.ndarray):
        self._round += 1
        obs = np.random.randn(self.num_envs, 3, 89).astype(np.float32)
        rewards = np.zeros((self.num_envs, 3), dtype=np.float32)
        masks = np.ones((self.num_envs, 3), dtype=bool)

        done = self._round >= self._rounds_to_done
        dones = np.full(self.num_envs, done, dtype=bool)

        infos: list[dict] = []
        for i in range(self.num_envs):
            if done:
                # Alternate wins and losses
                winner = 0 if i % 2 == 0 else 1
                rewards[i] = 1.0 if winner == 0 else -1.0
                infos.append({
                    "winner": winner,
                    "team_a_alive": 2 if winner == 0 else 0,
                    "team_b_alive": 0 if winner == 0 else 2,
                    "terminal_observation": True,
                })
            else:
                infos.append({})

        if done:
            self._round = 0

        return obs, rewards, dones, masks, infos

    def close(self):
        pass


class _MockNetwork:
    """Minimal mock of ActorCritic for testing."""

    def get_action_and_value(self, obs, action=None):
        batch = obs.shape[0]
        actions = torch.randn(batch, 2)
        log_probs = torch.zeros(batch)
        entropy = torch.ones(batch)
        values = torch.zeros(batch)
        return actions, log_probs, entropy, values


class _MockAgent:
    def __init__(self):
        self.network = _MockNetwork()

    def save(self, path):
        """No-op save for testing."""
        pass


class _MockPPO:
    """Minimal mock of PPOTrainer for the outer loop test."""

    def __init__(self):
        self.agent = _MockAgent()
        self.device = torch.device("cpu")
        self.rollout_steps = 4
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def collect_rollout_vec(self, env):
        """Minimal rollout collection for testing."""
        from knockout.training.rollout_buffer import RolloutBuffer

        n_streams = env.num_envs * 3
        buffer = RolloutBuffer(
            buffer_size=self.rollout_steps * n_streams,
            obs_dim=89,
            action_dim=2,
        )
        buffer.init_structured(self.rollout_steps, n_streams)

        obs, masks = env.reset()
        for step in range(self.rollout_steps):
            obs_np = np.asarray(obs)
            masks_np = np.asarray(masks)
            flat_obs = obs_np.reshape(-1, 89)
            flat_masks = masks_np.reshape(-1)
            actions = np.random.randn(n_streams, 2).astype(np.float32)

            angles = np.random.uniform(0, 360, (env.num_envs, 3, 1))
            powers = np.random.uniform(0, 400, (env.num_envs, 3, 1))
            env_actions = np.concatenate([angles, powers], axis=-1).astype(np.float32)
            obs, rewards, dones, masks, infos = env.step(env_actions)

            buffer.add_step(
                step=step,
                obs=flat_obs,
                actions=actions,
                log_probs=np.zeros(n_streams, dtype=np.float32),
                rewards=np.asarray(rewards).reshape(-1),
                values=np.zeros(n_streams, dtype=np.float32),
                dones=np.repeat(np.asarray(dones), 3).astype(np.float32),
                masks=flat_masks,
            )

        buffer.compute_gae_structured(gamma=self.gamma, gae_lambda=self.gae_lambda)
        return buffer

    def train_step(self, buffer):
        return {"policy_loss": 0.1, "value_loss": 0.2, "entropy": 0.5}
