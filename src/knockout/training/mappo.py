"""MAPPO (Multi-Agent PPO) trainer with centralized critic.

Key differences from standard PPO:
- Each agent has its own policy loss (computed independently)
- Value loss uses the shared critic
- Advantages are computed per-agent but using the shared value function
- Entropy bonus per agent to encourage exploration

Supports ``backend="tensor"`` for GPU-accelerated TensorVecEnv.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn, optim

from knockout.agents.base import Agent
from knockout.agents.mappo_agent import MAPPOAgent
from knockout.agents.random_agent import RandomAgent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.training.vec_env import SingleTeamVecEnv


class MAPPORolloutBuffer:
    """Stores multi-agent rollout data for MAPPO.

    Stores per-agent data for 3 agents with centralized value estimates.
    """

    def __init__(self, buffer_size: int, obs_dim: int = 89, action_dim: int = 2, num_agents: int = 3):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.pos = 0
        self.full = False

        # Per-agent data: (buffer_size, num_agents, ...)
        self.observations = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.masks = np.zeros((buffer_size, num_agents), dtype=np.float32)

        # Shared across agents (one per team per timestep)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        value: float,
        done: bool,
        masks: np.ndarray,
    ) -> None:
        """Add one timestep of team experience.

        Args:
            obs: (num_agents, obs_dim)
            actions: (num_agents, action_dim)
            log_probs: (num_agents,)
            rewards: (num_agents,)
            value: scalar centralized value
            done: episode terminated
            masks: (num_agents,) alive mask
        """
        if self.pos >= self.buffer_size:
            self.full = True
            return

        self.observations[self.pos] = obs
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.rewards[self.pos] = rewards
        self.values[self.pos] = value
        self.dones[self.pos] = float(done)
        self.masks[self.pos] = masks.astype(np.float32)
        self.pos += 1

    def compute_gae(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
        last_done: bool = True,
    ) -> None:
        """Compute GAE using the shared centralized value.

        Uses mean team reward for advantage computation since the critic
        produces a single team value.

        The done flag at step ``t`` means the episode ended at step ``t``,
        so step ``t`` does NOT bootstrap from V(s_{t+1}).

        NOTE: This method is retained for backward compatibility.
        The MAPPOTrainer now computes GAE directly in collect_rollout()
        with per-env temporal ordering.
        """
        n = self.pos if not self.full else self.buffer_size

        # Use mean reward across alive agents for centralized value target
        mean_rewards = np.zeros(n, dtype=np.float32)
        for t in range(n):
            alive_count = max(self.masks[t].sum(), 1.0)
            mean_rewards[t] = (self.rewards[t] * self.masks[t]).sum() / alive_count

        last_gae = 0.0
        for t in reversed(range(n)):
            non_terminal_t = 1.0 - self.dones[t]

            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            delta = mean_rewards[t] + gamma * next_value * non_terminal_t - self.values[t]
            self.advantages[t] = last_gae = (
                delta + gamma * gae_lambda * non_terminal_t * last_gae
            )

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    @property
    def size(self) -> int:
        """Number of stored transitions."""
        return self.buffer_size if self.full else self.pos

    def get_batches(self, batch_size: int):
        """Yield mini-batches of experience as tensors.

        Each batch dict contains:
            observations: (B, num_agents, obs_dim)
            actions: (B, num_agents, action_dim)
            log_probs: (B, num_agents)
            advantages: (B,)
            returns: (B,)
            masks: (B, num_agents)
        """
        n = self.size
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            yield {
                "observations": torch.as_tensor(self.observations[idx]),
                "actions": torch.as_tensor(self.actions[idx]),
                "log_probs": torch.as_tensor(self.log_probs[idx]),
                "advantages": torch.as_tensor(self.advantages[idx]),
                "returns": torch.as_tensor(self.returns[idx]),
                "masks": torch.as_tensor(self.masks[idx]),
            }


class MAPPOTrainer:
    """Multi-Agent PPO trainer with shared critic."""

    def __init__(
        self,
        config: GameConfig = DEFAULTS,
        num_envs: int = 8,
        opponent_factory: Callable[[], Agent] | None = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        rollout_steps: int = 128,
        device: str = "cpu",
        backend: str = "pymunk",
    ):
        self.config = config
        self.num_envs = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.device = torch.device(device)
        self.backend = backend

        # MAPPO agent with 3 policies + shared critic
        self.agent = MAPPOAgent(device=device)

        # Single optimizer for all parameters (policies + critic)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)

        # Opponent factory for Team B
        if opponent_factory is None:
            self._opponent_factory: Callable[[], Agent] = lambda: RandomAgent("opp")
        else:
            self._opponent_factory = opponent_factory

        # Vectorized environment
        if backend == "tensor":
            from knockout.env.tensor_env import TensorVecEnv

            self.vec_env = TensorVecEnv(
                num_envs=num_envs,
                config=config,
                device=str(self.device),
            )
        else:
            self.vec_env = SingleTeamVecEnv(
                num_envs=num_envs,
                opponent_factory=self._opponent_factory,
                config=config,
            )

    def collect_rollout(self, n_steps: int | None = None) -> MAPPORolloutBuffer:
        """Collect rollout from all envs.

        For each step:
        - Get team observations (num_envs, 3, 89)
        - Each policy acts on its own obs
        - Critic evaluates centralized value
        - Store: obs, actions, log_probs, rewards, values, masks

        Data is stored in (n_steps, num_envs) temporal layout so that
        GAE is computed per-env without cross-contamination from
        interleaved streams.

        Works with both SingleTeamVecEnv (numpy) and TensorVecEnv (numpy
        output but tensor-backed physics).

        Args:
            n_steps: Number of steps to collect. Defaults to self.rollout_steps.

        Returns:
            Filled MAPPORolloutBuffer.
        """
        if n_steps is None:
            n_steps = self.rollout_steps

        n = self.num_envs
        max_transitions = n_steps * n
        buffer = MAPPORolloutBuffer(
            buffer_size=max_transitions,
            obs_dim=89,
            action_dim=2,
            num_agents=3,
        )

        # Structured temporal storage: (n_steps, n_envs) for each field
        all_obs = np.zeros((n_steps, n, 3, 89), dtype=np.float32)
        all_actions = np.zeros((n_steps, n, 3, 2), dtype=np.float32)
        all_log_probs = np.zeros((n_steps, n, 3), dtype=np.float32)
        all_rewards = np.zeros((n_steps, n, 3), dtype=np.float32)
        all_values = np.zeros((n_steps, n), dtype=np.float32)
        all_dones = np.zeros((n_steps, n), dtype=np.float32)
        all_masks = np.zeros((n_steps, n, 3), dtype=np.float32)

        obs, masks = self.vec_env.reset()  # (N, 3, 89), (N, 3) — numpy from both backends

        for step_idx in range(n_steps):
            # Ensure numpy then convert to tensor for network
            obs_np = np.asarray(obs)
            masks_np = np.asarray(masks)

            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                # Each policy acts on its own obs
                raw_actions, log_probs, _ = self.agent.get_actions(obs_t)
                # Critic sees all team obs concatenated
                values = self.agent.get_value(obs_t)

            raw_np = raw_actions.cpu().numpy()         # (N, 3, 2)
            log_prob_np = log_probs.cpu().numpy()       # (N, 3)
            value_np = values.cpu().numpy()             # (N,)

            # Scale raw network output -> action space via sigmoid
            bounded = 1.0 / (1.0 + np.exp(-raw_np))
            scaled = bounded * np.array([360.0, self.config.MAX_LAUNCH_FORCE])
            actions_3d = scaled.astype(np.float32)  # (N, 3, 2)

            next_obs, rewards, dones, next_masks, _infos = self.vec_env.step(actions_3d)

            # Ensure numpy
            next_obs = np.asarray(next_obs)
            rewards_np = np.asarray(rewards)
            dones_np = np.asarray(dones)
            next_masks = np.asarray(next_masks)

            # Store in temporal layout
            all_obs[step_idx] = obs_np
            all_actions[step_idx] = raw_np
            all_log_probs[step_idx] = log_prob_np
            all_rewards[step_idx] = rewards_np
            all_values[step_idx] = value_np
            all_dones[step_idx] = dones_np.astype(np.float32)
            all_masks[step_idx] = masks_np.astype(np.float32)

            obs = next_obs
            masks = next_masks

        # Compute GAE per-env (each env is its own trajectory)
        # Mean reward across alive agents for centralized value target
        all_mean_rewards = np.zeros((n_steps, n), dtype=np.float32)
        for t in range(n_steps):
            for i in range(n):
                alive_count = max(all_masks[t, i].sum(), 1.0)
                all_mean_rewards[t, i] = (all_rewards[t, i] * all_masks[t, i]).sum() / alive_count

        advantages_2d = np.zeros((n_steps, n), dtype=np.float32)
        last_gae = np.zeros(n, dtype=np.float32)

        for t in reversed(range(n_steps)):
            # done at step t means the episode ended at step t.
            non_terminal_t = 1.0 - all_dones[t]

            if t == n_steps - 1:
                next_values = np.zeros(n, dtype=np.float32)
            else:
                next_values = all_values[t + 1]

            delta = all_mean_rewards[t] + self.gamma * next_values * non_terminal_t - all_values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal_t * last_gae
            advantages_2d[t] = last_gae

        returns_2d = advantages_2d + all_values

        # Now fill buffer in temporal order (step-major) for correct ordering
        for t in range(n_steps):
            for i in range(n):
                buffer.add(
                    obs=all_obs[t, i],
                    actions=all_actions[t, i],
                    log_probs=all_log_probs[t, i],
                    rewards=all_rewards[t, i],
                    value=all_values[t, i],
                    done=all_dones[t, i],
                    masks=all_masks[t, i],
                )

        # Overwrite the buffer's advantages and returns with correctly computed values
        for t in range(n_steps):
            for i in range(n):
                idx = t * n + i
                buffer.advantages[idx] = advantages_2d[t, i]
                buffer.returns[idx] = returns_2d[t, i]

        return buffer

    def train_step(self, rollout: MAPPORolloutBuffer) -> dict[str, float]:
        """PPO update with MAPPO-specific loss.

        - Compute advantages using GAE with shared critic values
        - Update each policy independently (policy loss + entropy)
        - Update shared critic (value loss)

        Args:
            rollout: Filled MAPPORolloutBuffer.

        Returns:
            Dict of training metrics.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _epoch in range(self.n_epochs):
            for batch in rollout.get_batches(self.batch_size):
                obs = batch["observations"].to(self.device)       # (B, 3, 89)
                old_actions = batch["actions"].to(self.device)     # (B, 3, 2)
                old_log_probs = batch["log_probs"].to(self.device) # (B, 3)
                advantages = batch["advantages"].to(self.device)   # (B,)
                returns = batch["returns"].to(self.device)         # (B,)
                agent_masks = batch["masks"].to(self.device)       # (B, 3)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Recompute actions/log_probs/entropy for each policy
                _, new_log_probs, entropies = self.agent.get_actions(obs, old_actions)
                # new_log_probs: (B, 3), entropies: (B, 3)

                # Centralized value
                new_values = self.agent.get_value(obs)  # (B,)

                # --- Per-agent policy loss ---
                # advantages is (B,) shared; broadcast to (B, 3) for per-agent ratio
                adv_expanded = advantages.unsqueeze(1).expand_as(new_log_probs)  # (B, 3)

                ratio = torch.exp(new_log_probs - old_log_probs)  # (B, 3)
                surr1 = ratio * adv_expanded
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_expanded

                # Mask dead agents so they don't contribute to loss
                raw_policy_loss = -torch.min(surr1, surr2)  # (B, 3)
                masked_policy_loss = (raw_policy_loss * agent_masks).sum() / agent_masks.sum().clamp(min=1.0)

                # Entropy bonus (per agent, masked)
                masked_entropy = (entropies * agent_masks).sum() / agent_masks.sum().clamp(min=1.0)

                # --- Centralized value loss ---
                value_loss = nn.functional.mse_loss(new_values, returns)

                # Total loss
                loss = (
                    masked_policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * masked_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients for all parameters
                all_params = list(self.agent.parameters())
                nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)

                self.optimizer.step()

                total_policy_loss += masked_policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += masked_entropy.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def train(
        self,
        total_timesteps: int = 100000,
        log_interval: int = 10,
    ) -> list[dict]:
        """Main training loop.

        Args:
            total_timesteps: Total environment steps to train for.
            log_interval: Print metrics every N rollouts.

        Returns:
            List of per-rollout metric dicts.
        """
        steps_per_rollout = self.rollout_steps * self.num_envs
        n_rollouts = max(total_timesteps // steps_per_rollout, 1)

        logs: list[dict] = []
        for rollout_idx in range(n_rollouts):
            buffer = self.collect_rollout()
            metrics = self.train_step(buffer)
            metrics["rollout"] = rollout_idx
            metrics["timesteps"] = (rollout_idx + 1) * steps_per_rollout
            logs.append(metrics)

            if (rollout_idx + 1) % log_interval == 0:
                print(
                    f"[MAPPO-{self.backend} x{self.num_envs}] Rollout {rollout_idx+1}/{n_rollouts}: "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, "
                    f"entropy={metrics['entropy']:.4f}"
                )

        return logs

    def close(self) -> None:
        """Release environment resources."""
        self.vec_env.close()


def main() -> None:
    """CLI entry point for MAPPO training."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train MAPPO agent for Knockout")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--opponent",
        choices=["random", "heuristic"],
        default="random",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="checkpoints/mappo.pt")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for networks and tensor backend (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--backend",
        choices=["pymunk", "tensor"],
        default="pymunk",
        help="Environment backend: pymunk (CPU) or tensor (GPU-accelerated)",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    from knockout.agents.heuristic_agent import HeuristicAgent

    if args.opponent == "heuristic":
        opponent_factory: Callable[[], Agent] = lambda: HeuristicAgent("opp")
    else:
        opponent_factory = lambda: RandomAgent("opp")  # noqa: E731

    print(f"Training MAPPO: backend={args.backend}, device={device}, "
          f"num_envs={args.num_envs}, opponent={args.opponent}")

    trainer = MAPPOTrainer(
        num_envs=args.num_envs,
        opponent_factory=opponent_factory,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        device=device,
        backend=args.backend,
    )

    logs = trainer.train(total_timesteps=args.timesteps)

    # Save checkpoint
    save_path = Path(args.save_path)
    trainer.agent.save(save_path)
    print(f"Saved MAPPO agent to {save_path}")

    # Print final stats
    if logs:
        final = logs[-1]
        print(
            f"Final: policy_loss={final['policy_loss']:.4f}, "
            f"value_loss={final['value_loss']:.4f}, "
            f"entropy={final['entropy']:.4f}"
        )

    trainer.close()


if __name__ == "__main__":
    main()
