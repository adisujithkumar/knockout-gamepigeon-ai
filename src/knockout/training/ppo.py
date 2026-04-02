"""PPO-Clip trainer for continuous action multi-agent environment.

Supports both single-env and vectorized (SingleTeamVecEnv) collection.
Supports ``backend="tensor"`` for GPU-accelerated TensorVecEnv.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn, optim

from knockout.agents.base import Agent
from knockout.agents.rl_agent import RLAgent
from knockout.agents.random_agent import RandomAgent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.env.penguin_env import PenguinEnv
from knockout.training.rollout_buffer import RolloutBuffer
from knockout.training.vec_env import SingleTeamVecEnv


class PPOTrainer:
    """PPO-Clip trainer for the knockout environment."""

    def __init__(
        self,
        config: GameConfig = DEFAULTS,
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
        num_envs: int = 1,
        opponent_factory: Callable[[], Agent] | None = None,
        backend: str = "pymunk",
    ):
        self.config = config
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
        self.num_envs = num_envs
        self.backend = backend

        # Create the learning agent (controls Team A: penguin_0,1,2)
        self.agent = RLAgent("learner", device=device)
        self.optimizer = optim.Adam(self.agent.network.parameters(), lr=lr)

        # Opponent for Team B (default: random)
        if opponent_factory is None:
            self._opponent_factory: Callable[[], Agent] = lambda: RandomAgent("opponent")
        else:
            self._opponent_factory = opponent_factory

        # Keep a single RandomAgent for backward compat in single-env mode
        self.opponent = self._opponent_factory()

    # ------------------------------------------------------------------
    # Vectorized rollout collection
    # ------------------------------------------------------------------

    def collect_rollout_vec(self, vec_env) -> RolloutBuffer:
        """Collect rollout_steps of experience from a vectorized env.

        Each step produces ``num_envs * 3`` transitions (one per Team-A
        agent per env).  Data is stored in structured (step, stream)
        layout so that GAE is computed per-agent-trajectory without
        cross-contamination from interleaved streams.

        Works with both SingleTeamVecEnv (numpy) and TensorVecEnv (numpy
        output but tensor-backed physics).

        Args:
            vec_env: A ``SingleTeamVecEnv`` or ``TensorVecEnv`` instance.

        Returns:
            Filled ``RolloutBuffer``.
        """
        n = vec_env.num_envs
        n_streams = n * 3  # one stream per Team-A agent per env
        buffer = RolloutBuffer(
            buffer_size=self.rollout_steps * n_streams,
            obs_dim=89,
            action_dim=2,
        )
        buffer.init_structured(self.rollout_steps, n_streams)

        obs, masks = vec_env.reset()  # (N,3,89), (N,3) — numpy from both backends

        for step_idx in range(self.rollout_steps):
            # Ensure numpy arrays (both backends return numpy)
            obs_np = np.asarray(obs)
            masks_np = np.asarray(masks)

            # Flatten for network: (N*3, 89)
            flat_obs = obs_np.reshape(-1, 89)
            flat_masks = masks_np.reshape(-1)

            obs_t = torch.as_tensor(flat_obs, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                raw_actions, log_probs, _, values = (
                    self.agent.network.get_action_and_value(obs_t)
                )

            raw_np = raw_actions.cpu().numpy()             # (N*3, 2)
            log_prob_np = log_probs.cpu().numpy()           # (N*3,)
            value_np = values.cpu().numpy()                 # (N*3,)

            # Scale raw network output -> action space via sigmoid
            bounded = 1.0 / (1.0 + np.exp(-raw_np))
            scaled = bounded * np.array([360.0, self.config.MAX_LAUNCH_FORCE])
            actions_3d = scaled.reshape(n, 3, 2).astype(np.float32)

            next_obs, rewards, dones, next_masks, _infos = vec_env.step(actions_3d)

            # Ensure numpy
            next_obs = np.asarray(next_obs)
            rewards = np.asarray(rewards)
            dones = np.asarray(dones)
            next_masks = np.asarray(next_masks)

            # Broadcast env-done to per-agent
            flat_rewards = rewards.reshape(-1)         # (N*3,)
            flat_dones = np.repeat(dones, 3)           # (N*3,)

            # Store in structured layout (step, stream)
            buffer.add_step(
                step=step_idx,
                obs=flat_obs,
                actions=raw_np,
                log_probs=log_prob_np,
                rewards=flat_rewards,
                values=value_np,
                dones=flat_dones,
                masks=flat_masks,
            )

            obs = next_obs
            masks = next_masks

        # Compute GAE per-stream (each agent in each env is its own trajectory)
        buffer.compute_gae_structured(gamma=self.gamma, gae_lambda=self.gae_lambda)
        return buffer

    # ------------------------------------------------------------------
    # Single-env rollout (kept for backward compatibility)
    # ------------------------------------------------------------------

    def collect_rollout(self, env: PenguinEnv) -> RolloutBuffer:
        """Collect rollout_steps of experience from a single environment."""
        buffer = RolloutBuffer(
            buffer_size=self.rollout_steps * 3,  # 3 agents on our team
            obs_dim=89,
            action_dim=2,
        )

        obs_dict, _ = env.reset()

        for step in range(self.rollout_steps):
            if not env.agents:
                obs_dict, _ = env.reset()

            actions = {}
            step_data = {}  # Store per-agent data for buffer

            for agent_id in env.agents:
                obs = obs_dict[agent_id]
                agent_idx = int(agent_id.split("_")[1])

                if agent_idx < 3:  # Team A - our learning agent
                    obs_tensor = torch.as_tensor(
                        obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action_t, log_prob, _, value = self.agent.network.get_action_and_value(
                            obs_tensor
                        )

                    action_np = action_t.squeeze(0).cpu().numpy()
                    # Scale action
                    action_bounded = 1.0 / (1.0 + np.exp(-action_np))
                    scaled = np.array([0.0, 0.0]) + action_bounded * np.array(
                        [360.0, self.config.MAX_LAUNCH_FORCE]
                    )
                    actions[agent_id] = scaled.astype(np.float32)

                    step_data[agent_id] = {
                        "obs": obs,
                        "action": action_np,
                        "log_prob": log_prob.item(),
                        "value": value.item(),
                    }
                else:  # Team B - opponent
                    actions[agent_id] = self.opponent.get_action(obs)

            next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)

            # Store transitions for Team A agents
            for agent_id, data in step_data.items():
                reward = rewards.get(agent_id, 0.0)
                done = terminations.get(agent_id, False) or truncations.get(agent_id, False)
                buffer.add(
                    obs=data["obs"],
                    action=data["action"],
                    log_prob=data["log_prob"],
                    reward=reward,
                    value=data["value"],
                    done=done,
                )

            if any(terminations.values()) or any(truncations.values()):
                obs_dict, _ = env.reset()
            else:
                obs_dict = next_obs_dict

        # Compute GAE
        buffer.compute_gae(gamma=self.gamma, gae_lambda=self.gae_lambda)
        return buffer

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def train_step(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update epochs on collected buffer."""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size):
                obs = batch["observations"].to(self.device)
                old_actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                # Normalize advantages (use correction=0 to avoid NaN with single-element batches)
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                else:
                    advantages = advantages - advantages.mean()

                # Get current policy evaluation
                _, new_log_probs, entropy, values = self.agent.network.get_action_and_value(
                    obs, old_actions
                )

                # Policy loss (PPO-Clip)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns)

                # Total loss
                loss = (
                    policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 10000, log_interval: int = 10) -> list[dict]:
        """Main training loop.

        When ``num_envs > 1`` uses the vectorized path automatically;
        ``num_envs == 1`` falls back to the original single-env code so
        behaviour is fully backward-compatible.
        """
        logs: list[dict] = []

        if self.num_envs > 1:
            return self._train_vec(total_timesteps, log_interval)

        # --- Single-env path (original) ----------------------------------
        env = PenguinEnv(config=self.config)
        n_rollouts = max(total_timesteps // (self.rollout_steps * 3), 1)

        for rollout in range(n_rollouts):
            buffer = self.collect_rollout(env)
            metrics = self.train_step(buffer)
            metrics["rollout"] = rollout
            logs.append(metrics)

            if (rollout + 1) % log_interval == 0:
                print(
                    f"Rollout {rollout+1}/{n_rollouts}: "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, "
                    f"entropy={metrics['entropy']:.4f}"
                )

        env.close()
        return logs

    def _train_vec(
        self, total_timesteps: int, log_interval: int
    ) -> list[dict]:
        """Training loop using the vectorized environment."""
        if self.backend == "tensor":
            from knockout.env.tensor_env import TensorVecEnv

            vec_env = TensorVecEnv(
                num_envs=self.num_envs,
                config=self.config,
                device=str(self.device),
            )
            backend_label = "tensor"
        else:
            vec_env = SingleTeamVecEnv(
                num_envs=self.num_envs,
                opponent_factory=self._opponent_factory,
                config=self.config,
            )
            backend_label = "pymunk"

        steps_per_rollout = self.rollout_steps * self.num_envs * 3
        n_rollouts = max(total_timesteps // steps_per_rollout, 1)

        logs: list[dict] = []
        for rollout in range(n_rollouts):
            buffer = self.collect_rollout_vec(vec_env)
            metrics = self.train_step(buffer)
            metrics["rollout"] = rollout
            logs.append(metrics)

            if (rollout + 1) % log_interval == 0:
                print(
                    f"[{backend_label} x{self.num_envs}] Rollout {rollout+1}/{n_rollouts}: "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, "
                    f"entropy={metrics['entropy']:.4f}"
                )

        vec_env.close()
        return logs


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO agent for Knockout")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", type=str, default="checkpoints/ppo_agent.pt")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--opponent",
        choices=["random", "heuristic"],
        default="random",
    )
    parser.add_argument(
        "--backend",
        choices=["pymunk", "tensor"],
        default="pymunk",
        help="Environment backend: pymunk (CPU) or tensor (GPU-accelerated)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for networks and tensor backend (cpu, cuda, auto)",
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

    trainer = PPOTrainer(
        lr=args.lr,
        num_envs=args.num_envs,
        opponent_factory=opponent_factory,
        device=device,
        backend=args.backend,
    )
    trainer.train(total_timesteps=args.timesteps)

    from pathlib import Path

    trainer.agent.save(Path(args.save_path))
    print(f"Saved agent to {args.save_path}")


if __name__ == "__main__":
    main()
