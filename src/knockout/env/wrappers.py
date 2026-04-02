"""Gymnasium wrappers for observation preprocessing."""

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


class RunningMeanStd:
    """Running mean and standard deviation calculator using Welford's algorithm."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        """Initialize running statistics.

        Args:
            shape: Shape of the data
            epsilon: Small constant for numerical stability
        """
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """Update statistics with new data.

        Args:
            x: New data point
        """
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var = np.var(x, axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Update from batch statistics.

        Args:
            batch_mean: Mean of the batch
            batch_var: Variance of the batch
            batch_count: Number of samples in batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormalizeObservation(gym.ObservationWrapper[NDArray[np.float32], int, int]):
    """Wrapper to normalize observations using running mean and std."""

    def __init__(self, env: gym.Env[NDArray[np.float32], int], epsilon: float = 1e-8) -> None:
        """Initialize normalization wrapper.

        Args:
            env: Environment to wrap
            epsilon: Small constant for numerical stability
        """
        super().__init__(env)
        self.epsilon = epsilon

        # Initialize running statistics
        assert isinstance(env.observation_space, spaces.Box)
        self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)

        # Observation space remains the same
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize observation.

        Args:
            obs: Raw observation

        Returns:
            Normalized observation
        """
        # Update running statistics
        self.obs_rms.update(obs)

        # Normalize
        normalized = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return normalized.astype(np.float32)  # type: ignore[return-value]


class FrameStack(gym.ObservationWrapper[NDArray[np.float32], int, int]):
    """Wrapper to stack multiple consecutive observations."""

    def __init__(self, env: gym.Env[NDArray[np.float32], int], num_stack: int) -> None:
        """Initialize frame stacking wrapper.

        Args:
            env: Environment to wrap
            num_stack: Number of frames to stack
        """
        super().__init__(env)
        self.num_stack = num_stack

        # Initialize frame buffer
        assert isinstance(env.observation_space, spaces.Box)
        self.frames: deque[NDArray[np.float32]] = deque(maxlen=num_stack)

        # Update observation space to reflect stacked frames
        low = np.repeat(env.observation_space.low, num_stack, axis=0)
        high = np.repeat(env.observation_space.high, num_stack, axis=0)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Stack observations.

        Args:
            obs: Current observation

        Returns:
            Stacked observations
        """
        self.frames.append(obs)
        # Stack along first axis
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)  # type: ignore[return-value]

    def reset(self, **kwargs: Any) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment and frame buffer.

        Args:
            **kwargs: Arguments to pass to environment reset

        Returns:
            observation: Initial stacked observation
            info: Additional information dictionary
        """
        obs, info = self.env.reset(**kwargs)

        # Fill buffer with initial observation
        for _ in range(self.num_stack):
            self.frames.append(obs)

        return self.observation(obs), info
