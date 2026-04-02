"""PPO rollout buffer with GAE computation."""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Stores rollout data and computes GAE advantages.

    Supports both flat (single-trajectory) and structured (multi-stream)
    storage.  For the multi-stream case, ``add_structured()`` and
    ``compute_gae_structured()`` ensure GAE is computed independently
    along each agent trajectory rather than across interleaved streams.
    """

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False

        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        # Structured storage (populated by add_structured)
        self._structured = False
        self._n_steps: int = 0
        self._n_streams: int = 0

    def add(self, obs, action, log_prob, reward, value, done):
        """Add a single transition."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = float(done)
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True

    # ------------------------------------------------------------------
    # Structured (multi-stream) API for vectorized collection
    # ------------------------------------------------------------------

    def init_structured(self, n_steps: int, n_streams: int) -> None:
        """Initialize structured storage for *n_steps* x *n_streams*.

        Each stream is an independent agent trajectory.  Data is stored
        in (n_steps, n_streams) layout and flattened after GAE.

        Args:
            n_steps: Number of rollout steps (temporal axis).
            n_streams: Number of parallel agent streams
                       (``num_envs * agents_per_env``).
        """
        self._structured = True
        self._n_steps = n_steps
        self._n_streams = n_streams
        total = n_steps * n_streams

        # Resize if needed
        if total > self.buffer_size:
            self.buffer_size = total
            self.observations = np.zeros((total, self.obs_dim), dtype=np.float32)
            self.actions = np.zeros((total, self.action_dim), dtype=np.float32)
            self.log_probs = np.zeros(total, dtype=np.float32)
            self.rewards = np.zeros(total, dtype=np.float32)
            self.values = np.zeros(total, dtype=np.float32)
            self.dones = np.zeros(total, dtype=np.float32)
            self.advantages = np.zeros(total, dtype=np.float32)
            self.returns = np.zeros(total, dtype=np.float32)

        # Masks: True if the agent was alive and contributing data
        self._masks = np.zeros((n_steps, n_streams), dtype=bool)
        self.pos = 0

    def add_step(
        self,
        step: int,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        """Add one timestep of data for all streams.

        Args:
            step: Timestep index (0 to n_steps-1).
            obs: (n_streams, obs_dim) observations.
            actions: (n_streams, action_dim) raw actions.
            log_probs: (n_streams,) log probabilities.
            rewards: (n_streams,) rewards.
            values: (n_streams,) value estimates.
            dones: (n_streams,) done flags (True = episode ended).
            masks: (n_streams,) alive masks (True = agent was alive).
        """
        ns = self._n_streams
        base = step * ns
        end = base + ns

        self.observations[base:end] = obs
        self.actions[base:end] = actions
        self.log_probs[base:end] = log_probs
        self.rewards[base:end] = rewards
        self.values[base:end] = values
        self.dones[base:end] = dones.astype(np.float32)
        self._masks[step] = masks

        self.pos = max(self.pos, end)

    def compute_gae_structured(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_values: np.ndarray | None = None,
        last_dones: np.ndarray | None = None,
    ) -> None:
        """Compute GAE independently for each agent stream.

        Unlike ``compute_gae``, this respects per-stream temporal ordering:
        advantages propagate backward along each agent's trajectory without
        cross-contamination from other agents' transitions.

        The done flag at step ``t`` means the episode ended after taking
        action at step ``t``.  Therefore ``dones[t]`` gates whether step
        ``t`` bootstraps from ``V(s_{t+1})``:

        .. code::

            non_terminal_t = 1 - dones[t]
            delta_t = r_t + gamma * V(s_{t+1}) * non_terminal_t - V(s_t)
            A_t = delta_t + gamma * lambda * non_terminal_t * A_{t+1}

        After computation, data is compacted to only include masked (alive)
        transitions.

        Args:
            gamma: Discount factor.
            gae_lambda: GAE lambda.
            last_values: (n_streams,) bootstrap values for last step.
            last_dones: (n_streams,) done flags for last step.
        """
        T = self._n_steps
        S = self._n_streams

        if last_values is None:
            last_values = np.zeros(S, dtype=np.float32)
        if last_dones is None:
            last_dones = np.ones(S, dtype=np.float32)

        # Reshape to (T, S) for temporal GAE
        rewards_2d = self.rewards[: T * S].reshape(T, S)
        values_2d = self.values[: T * S].reshape(T, S)
        dones_2d = self.dones[: T * S].reshape(T, S)
        advantages_2d = np.zeros((T, S), dtype=np.float32)

        last_gae = np.zeros(S, dtype=np.float32)

        for t in reversed(range(T)):
            # done at step t means the episode ended at step t.
            # Use dones[t] to decide whether to bootstrap from t+1.
            non_terminal_t = 1.0 - dones_2d[t]

            if t == T - 1:
                # For the very last step, the "next" value comes from
                # the caller (or is zero if the rollout is treated as
                # ending).  We use *non_terminal_t* (not last_dones)
                # to gate the bootstrap.
                next_values = last_values
            else:
                next_values = values_2d[t + 1]

            delta = rewards_2d[t] + gamma * next_values * non_terminal_t - values_2d[t]
            last_gae = delta + gamma * gae_lambda * non_terminal_t * last_gae
            advantages_2d[t] = last_gae

        returns_2d = advantages_2d + values_2d

        # Write back (still in structured layout)
        self.advantages[: T * S] = advantages_2d.reshape(-1)
        self.returns[: T * S] = returns_2d.reshape(-1)

        # Compact: keep only alive transitions
        mask_flat = self._masks.reshape(-1)
        valid_idx = np.where(mask_flat)[0]
        n_valid = len(valid_idx)

        if n_valid < T * S:
            # Compact in-place
            self.observations[:n_valid] = self.observations[valid_idx]
            self.actions[:n_valid] = self.actions[valid_idx]
            self.log_probs[:n_valid] = self.log_probs[valid_idx]
            self.rewards[:n_valid] = self.rewards[valid_idx]
            self.values[:n_valid] = self.values[valid_idx]
            self.dones[:n_valid] = self.dones[valid_idx]
            self.advantages[:n_valid] = self.advantages[valid_idx]
            self.returns[:n_valid] = self.returns[valid_idx]

        self.pos = n_valid
        self.full = False
        self._structured = False  # Now flat for get_batches

    # ------------------------------------------------------------------
    # Original flat API
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
        last_done: bool = True,
    ):
        """Compute Generalized Advantage Estimation.

        The done flag at step ``t`` means the episode ended at step ``t``.
        ``dones[t]`` gates bootstrapping from step ``t+1``.

        WARNING: Only correct for flat single-trajectory buffers (or
        interleaved streams where dones act as natural boundaries).
        For vectorized multi-agent collection, use ``compute_gae_structured``.
        """
        n = self.pos if not self.full else self.buffer_size
        last_gae = 0.0

        for t in reversed(range(n)):
            non_terminal_t = 1.0 - self.dones[t]

            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * non_terminal_t - self.values[t]
            self.advantages[t] = last_gae = (
                delta + gamma * gae_lambda * non_terminal_t * last_gae
            )

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int):
        """Yield mini-batches of experience as tensors."""
        n = self.pos if not self.full else self.buffer_size
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            yield {
                "observations": torch.as_tensor(self.observations[batch_idx]),
                "actions": torch.as_tensor(self.actions[batch_idx]),
                "log_probs": torch.as_tensor(self.log_probs[batch_idx]),
                "advantages": torch.as_tensor(self.advantages[batch_idx]),
                "returns": torch.as_tensor(self.returns[batch_idx]),
                "values": torch.as_tensor(self.values[batch_idx]),
            }

    def reset(self):
        """Reset buffer for next rollout."""
        self.pos = 0
        self.full = False
        self._structured = False
