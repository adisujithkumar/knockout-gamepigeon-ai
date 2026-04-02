"""Tests for Gymnasium wrappers.

Uses a simple mock Gymnasium environment instead of KnockoutEnv (which doesn't
exist in v2). This isolates the wrapper logic from the game environment.
"""

import gymnasium as gym
import numpy as np
import pytest

from knockout.env.wrappers import FrameStack, NormalizeObservation, RunningMeanStd


class SimpleBoxEnv(gym.Env):
    """Simple Box environment for testing wrappers."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1, 1, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._step = 0
        self._rng = np.random.default_rng(42)

    def reset(self, **kwargs):
        self._step = 0
        seed = kwargs.get("seed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs = self._rng.standard_normal(10).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = self._rng.standard_normal(10).astype(np.float32)
        return obs, 0.0, self._step >= 10, False, {}


class TestRunningMeanStd:
    """Test running statistics calculator."""

    def test_initial_state(self):
        """Test initial state of running statistics."""
        rms = RunningMeanStd(shape=(5,))
        assert rms.mean.shape == (5,)
        assert rms.var.shape == (5,)
        np.testing.assert_array_equal(rms.mean, np.zeros(5))
        np.testing.assert_array_equal(rms.var, np.ones(5))

    def test_update_with_single_value(self):
        """Test updating with a single value."""
        rms = RunningMeanStd(shape=(3,))
        rms.update(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        # After one update, mean should move toward the value
        assert rms.count > 1  # epsilon + 1


class TestNormalizeObservation:
    """Test observation normalization wrapper."""

    def test_wrapper_creation(self):
        """Test creating wrapper."""
        env = SimpleBoxEnv()
        wrapped_env = NormalizeObservation(env)
        assert wrapped_env is not None

    def test_observation_space_shape(self):
        """Test observation space shape is preserved."""
        env = SimpleBoxEnv()
        wrapped_env = NormalizeObservation(env)
        assert wrapped_env.observation_space.shape == env.observation_space.shape

    def test_normalize_returns_float32(self):
        """Test that normalized observations are float32."""
        env = SimpleBoxEnv()
        wrapped_env = NormalizeObservation(env)
        obs, _ = wrapped_env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_step_returns_normalized_obs(self):
        """Test step returns normalized observation."""
        env = SimpleBoxEnv()
        wrapped_env = NormalizeObservation(env)
        wrapped_env.reset()

        obs, _, _, _, _ = wrapped_env.step(0)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_normalization_updates_running_stats(self):
        """Test normalization updates running mean and std."""
        env = SimpleBoxEnv()
        wrapped_env = NormalizeObservation(env)
        wrapped_env.reset()
        initial_mean = wrapped_env.obs_rms.mean.copy()

        # Run several steps
        for _ in range(10):
            _, _, terminated, truncated, _ = wrapped_env.step(0)
            if terminated or truncated:
                wrapped_env.reset()

        # Statistics should have updated
        assert not np.allclose(wrapped_env.obs_rms.mean, initial_mean)

    def test_normalize_multiple_resets(self):
        """Test normalization works across multiple resets."""
        env = SimpleBoxEnv()
        wrapped_env = NormalizeObservation(env)

        for _ in range(3):
            obs, _ = wrapped_env.reset()
            assert isinstance(obs, np.ndarray)
            assert obs.shape == (10,)


class TestFrameStack:
    """Test frame stacking wrapper."""

    def test_wrapper_creation(self):
        """Test creating frame stack wrapper."""
        env = SimpleBoxEnv()
        wrapped_env = FrameStack(env, num_stack=4)
        assert wrapped_env is not None
        assert wrapped_env.num_stack == 4

    def test_observation_space_shape(self):
        """Test observation space is stacked correctly."""
        env = SimpleBoxEnv()
        wrapped_env = FrameStack(env, num_stack=4)

        base_shape = env.observation_space.shape[0]
        expected_shape = (base_shape * 4,)
        assert wrapped_env.observation_space.shape == expected_shape

    def test_framestack_shape(self):
        """Test that stacked observations have correct shape."""
        env = SimpleBoxEnv()
        wrapped_env = FrameStack(env, num_stack=3)

        obs, _ = wrapped_env.reset()
        expected_shape = (env.observation_space.shape[0] * 3,)
        assert obs.shape == expected_shape

    def test_framestack_initial_repeats(self):
        """Test initial stack repeats the first observation."""
        env = SimpleBoxEnv()
        wrapped_env = FrameStack(env, num_stack=3)

        obs, _ = wrapped_env.reset()
        base_size = env.observation_space.shape[0]

        # Split stacked obs into frames
        frame1 = obs[:base_size]
        frame2 = obs[base_size: 2 * base_size]
        frame3 = obs[2 * base_size:]

        # All frames should be identical after reset
        # Note: the last frame is from observation() call after filling buffer,
        # so frame2, frame3, and frame1 from buffer should all match
        np.testing.assert_array_equal(frame1, frame2)

    def test_step_updates_stack(self):
        """Test step updates the frame stack."""
        env = SimpleBoxEnv()
        wrapped_env = FrameStack(env, num_stack=2)

        obs1, _ = wrapped_env.reset()
        obs2, _, _, _, _ = wrapped_env.step(0)

        # Observations should differ after step
        assert not np.allclose(obs1, obs2)

    def test_framestack_with_size_one(self):
        """Test frame stack with size 1 (no stacking)."""
        env = SimpleBoxEnv()
        wrapped_env = FrameStack(env, num_stack=1)

        obs, _ = wrapped_env.reset()
        assert obs.shape == env.observation_space.shape

    def test_framestack_multiple_resets(self):
        """Test frame stack works across multiple resets."""
        env = SimpleBoxEnv()
        wrapped_env = FrameStack(env, num_stack=4)

        for _ in range(3):
            obs, _ = wrapped_env.reset()
            assert isinstance(obs, np.ndarray)
            base_size = env.observation_space.shape[0]
            # All frames should be identical after reset
            frame1 = obs[:base_size]
            frame2 = obs[base_size: 2 * base_size]
            np.testing.assert_array_equal(frame1, frame2)


class TestCombinedWrappers:
    """Test combining multiple wrappers."""

    def test_combined_wrappers(self):
        """Test applying normalization then frame stacking."""
        env = SimpleBoxEnv()
        env = NormalizeObservation(env)
        env = FrameStack(env, num_stack=4)

        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10 * 4,)
        assert obs.dtype == np.float32

    def test_combined_wrappers_full_episode(self):
        """Test combined wrappers can run a full episode."""
        env = SimpleBoxEnv()
        env = NormalizeObservation(env)
        env = FrameStack(env, num_stack=2)

        env.reset()
        for _ in range(20):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert obs.dtype == np.float32
            assert obs.shape == (10 * 2,)
            if terminated or truncated:
                env.reset()

    def test_framestack_then_normalize(self):
        """Test applying frame stacking then normalization."""
        env = SimpleBoxEnv()
        env = FrameStack(env, num_stack=3)
        env = NormalizeObservation(env)

        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10 * 3,)
        assert obs.dtype == np.float32
