"""Contrastive Trajectory Mining for zero-knowledge reward discovery.

Discovers which observation features matter by statistically comparing
winning vs losing game trajectories. Starts with ONLY sparse +1/-1
win/loss reward and automatically finds intermediate reward signals.

Algorithm (outer loop):
    1. Collect game trajectories with outcomes
    2. Compute Cohen's d effect sizes between winning/losing trajectories
    3. Convert significant features into shaped reward components
    4. Train PPO with shaped + sparse reward
    5. Repeat with updated policy
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from knockout.core.config import GameConfig, DEFAULTS

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Observation feature name registry (89 dims)
# --------------------------------------------------------------------------

_PER_PENGUIN_NAMES = [
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "dist_from_center",
    "dist_to_edge",
    "speed",
    "heading",
    "alive",
    "rel_position_x",
    "rel_position_y",
    "rel_velocity_x",
    "rel_velocity_y",
    "dist_to_ego",
]

_SLOT_LABELS = ["ego", "ally1", "ally2", "enemy1", "enemy2", "enemy3"]

_GLOBAL_NAMES = [
    "team_a_alive",
    "team_b_alive",
    "ego_team_alive",
    "opp_team_alive",
    "timestep",
]


def _build_feature_names() -> list[str]:
    names: list[str] = []
    for slot in _SLOT_LABELS:
        for feat in _PER_PENGUIN_NAMES:
            names.append(f"{slot}.{feat}")
    for gname in _GLOBAL_NAMES:
        names.append(f"global.{gname}")
    assert len(names) == 89
    return names


FEATURE_NAMES: list[str] = _build_feature_names()


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastiveConfig:
    """Configuration for contrastive trajectory mining."""

    collection_episodes: int = 2000       # games per data collection phase
    min_effect_size: float = 0.3          # Cohen's d threshold
    max_reward_features: int = 10         # cap on discovered features
    reward_scale: float = 0.5            # scale of shaped reward vs sparse
    num_iterations: int = 10              # outer loop iterations
    steps_per_iteration: int = 1_000_000  # PPO steps between analyses


# --------------------------------------------------------------------------
# Discovery result
# --------------------------------------------------------------------------

@dataclass
class DiscoveryResult:
    """Result of contrastive analysis: which features distinguish wins."""

    feature_indices: list[int]
    effect_sizes: list[float]
    directions: list[float]     # +1 = higher in wins, -1 = lower in wins
    feature_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": [
                {
                    "index": idx,
                    "name": name,
                    "effect_size": float(es),
                    "direction": float(d),
                }
                for idx, name, es, d in zip(
                    self.feature_indices,
                    self.feature_names,
                    self.effect_sizes,
                    self.directions,
                )
            ]
        }

    def summary(self) -> str:
        lines = ["Discovered features (by |Cohen's d|):"]
        for idx, name, es, d in zip(
            self.feature_indices,
            self.feature_names,
            self.effect_sizes,
            self.directions,
        ):
            sign = "higher" if d > 0 else "lower"
            lines.append(f"  {name} (idx={idx}): d={es:+.3f}, {sign} in wins")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Trajectory collector
# --------------------------------------------------------------------------

class TrajectoryCollector:
    """Collects full game trajectories with outcomes from TensorVecEnv."""

    def __init__(self, config: GameConfig = DEFAULTS) -> None:
        self.config = config

    def collect(
        self,
        env: Any,
        policy: Any,
        num_episodes: int,
        device: str = "cpu",
    ) -> dict[str, np.ndarray]:
        """Run games, return per-step observations with outcome labels.

        Args:
            env: A TensorVecEnv instance.
            policy: An ActorCritic network (or None for random).
            num_episodes: Number of complete games to collect.
            device: Torch device string.

        Returns:
            Dictionary with:
                'observations': (N_total_steps, 89) all obs across episodes
                'outcomes': (N_total_steps,) +1/-1 for eventual game outcome
                'episode_ids': (N_total_steps,) which episode each step belongs to
        """
        torch_device = torch.device(device)
        n_envs = env.num_envs

        # Per-env accumulators
        env_obs_buffers: list[list[np.ndarray]] = [[] for _ in range(n_envs)]
        all_obs: list[np.ndarray] = []
        all_outcomes: list[np.ndarray] = []
        all_episode_ids: list[np.ndarray] = []

        episodes_collected = 0
        episodes_seen = 0  # total episodes including draws
        episode_counter = 0

        # Safety: cap total episodes attempted to avoid infinite loops when
        # draws dominate (winner == -1 episodes are skipped).
        max_total_episodes = num_episodes * 5

        log_interval = max(num_episodes // 4, 1)
        next_log_at = log_interval

        obs, masks = env.reset()  # (B, 3, 89), (B, 3)

        while episodes_collected < num_episodes:
            obs_np = np.asarray(obs)

            # Store per-agent observations for each env
            for e in range(n_envs):
                # Average across alive agents for a per-env observation
                alive_mask = masks[e].astype(bool)
                if alive_mask.any():
                    mean_obs = obs_np[e, alive_mask].mean(axis=0)
                else:
                    mean_obs = np.zeros(89, dtype=np.float32)
                env_obs_buffers[e].append(mean_obs)

            # Get actions
            if policy is not None:
                flat_obs = obs_np.reshape(-1, 89)
                obs_t = torch.as_tensor(
                    flat_obs, dtype=torch.float32, device=torch_device
                )
                with torch.no_grad():
                    raw_actions, _, _, _ = policy.get_action_and_value(obs_t)
                raw_np = raw_actions.cpu().numpy()
                bounded = 1.0 / (1.0 + np.exp(-raw_np))
                scaled = bounded * np.array(
                    [360.0, self.config.MAX_LAUNCH_FORCE]
                )
                actions_3d = scaled.reshape(n_envs, 3, 2).astype(np.float32)
            else:
                # Random actions
                angles = np.random.uniform(0, 360, (n_envs, 3, 1))
                powers = np.random.uniform(
                    0, self.config.MAX_LAUNCH_FORCE, (n_envs, 3, 1)
                )
                actions_3d = np.concatenate(
                    [angles, powers], axis=-1
                ).astype(np.float32)

            obs, rewards, dones, masks, infos = env.step(actions_3d)

            # Process finished episodes
            for e in range(n_envs):
                if dones[e] and episodes_collected < num_episodes:
                    info = infos[e]
                    winner = info.get("winner", -1)
                    episodes_seen += 1

                    # Team A perspective: +1 if team A won, -1 if team B won
                    if winner == 0:
                        outcome = 1.0
                    elif winner == 1:
                        outcome = -1.0
                    else:
                        # Draw — skip this episode
                        env_obs_buffers[e] = []
                        continue

                    traj = np.array(env_obs_buffers[e], dtype=np.float32)
                    n_steps = len(traj)
                    if n_steps == 0:
                        env_obs_buffers[e] = []
                        continue

                    all_obs.append(traj)
                    all_outcomes.append(
                        np.full(n_steps, outcome, dtype=np.float32)
                    )
                    all_episode_ids.append(
                        np.full(n_steps, episode_counter, dtype=np.int64)
                    )

                    episode_counter += 1
                    episodes_collected += 1
                    env_obs_buffers[e] = []

            # Safety break: too many draws
            if episodes_seen >= max_total_episodes:
                n_draws = episodes_seen - episodes_collected
                logger.warning(
                    "Collection halted: %d/%d episodes were draws "
                    "(only %d/%d non-draw episodes collected)",
                    n_draws, episodes_seen,
                    episodes_collected, num_episodes,
                )
                break

            # Progress logging (every ~25% of target)
            if episodes_collected >= next_log_at:
                logger.info(
                    "  ... collected %d/%d episodes so far",
                    episodes_collected, num_episodes,
                )
                # Jump ahead past the current count so we don't re-log
                next_log_at = episodes_collected + log_interval

        if episodes_collected == 0:
            # Return empty arrays instead of crashing on np.concatenate([])
            return {
                "observations": np.zeros((0, 89), dtype=np.float32),
                "outcomes": np.zeros(0, dtype=np.float32),
                "episode_ids": np.zeros(0, dtype=np.int64),
            }

        return {
            "observations": np.concatenate(all_obs, axis=0),
            "outcomes": np.concatenate(all_outcomes, axis=0),
            "episode_ids": np.concatenate(all_episode_ids, axis=0),
        }


# --------------------------------------------------------------------------
# Contrastive analyzer
# --------------------------------------------------------------------------

class ContrastiveAnalyzer:
    """Computes Cohen's d effect sizes between winning/losing trajectories."""

    def __init__(self, config: ContrastiveConfig) -> None:
        self.config = config

    def analyze(
        self,
        observations: np.ndarray,
        outcomes: np.ndarray,
    ) -> DiscoveryResult:
        """Compute per-feature effect sizes between wins and losses.

        Args:
            observations: (N, 89) array of observation steps.
            outcomes: (N,) array of +1 (win) or -1 (loss).

        Returns:
            DiscoveryResult with features sorted by |effect_size|,
            filtered by min_effect_size and capped at max_reward_features.
        """
        win_mask = outcomes > 0
        loss_mask = outcomes < 0

        win_obs = observations[win_mask]   # (N_win, 89)
        loss_obs = observations[loss_mask]  # (N_loss, 89)

        n_win = len(win_obs)
        n_loss = len(loss_obs)

        if n_win < 2 or n_loss < 2:
            return DiscoveryResult(
                feature_indices=[],
                effect_sizes=[],
                directions=[],
                feature_names=[],
            )

        # Compute Cohen's d for each feature dimension
        win_mean = win_obs.mean(axis=0)   # (89,)
        loss_mean = loss_obs.mean(axis=0)  # (89,)
        win_var = win_obs.var(axis=0, ddof=1)   # (89,)
        loss_var = loss_obs.var(axis=0, ddof=1)  # (89,)

        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((n_win - 1) * win_var + (n_loss - 1) * loss_var)
            / (n_win + n_loss - 2)
        )

        # Avoid division by zero for constant features
        pooled_std = np.maximum(pooled_std, 1e-8)

        cohens_d = (win_mean - loss_mean) / pooled_std  # (89,)

        # Filter by threshold and sort by absolute effect size
        abs_d = np.abs(cohens_d)
        significant = abs_d >= self.config.min_effect_size

        sig_indices = np.where(significant)[0]
        if len(sig_indices) == 0:
            return DiscoveryResult(
                feature_indices=[],
                effect_sizes=[],
                directions=[],
                feature_names=[],
            )

        # Sort by |d| descending, cap at max_reward_features
        sorted_order = np.argsort(-abs_d[sig_indices])
        top_k = sorted_order[: self.config.max_reward_features]
        selected = sig_indices[top_k]

        return DiscoveryResult(
            feature_indices=[int(i) for i in selected],
            effect_sizes=[float(cohens_d[i]) for i in selected],
            directions=[float(np.sign(cohens_d[i])) for i in selected],
            feature_names=[FEATURE_NAMES[i] for i in selected],
        )


# --------------------------------------------------------------------------
# Reward shaper
# --------------------------------------------------------------------------

class ContrastiveRewardShaper:
    """Converts discovery results into a batched reward function."""

    def __init__(self, config: ContrastiveConfig) -> None:
        self.config = config
        self._indices: list[int] = []
        self._weights: torch.Tensor = torch.zeros(0)
        self._device = torch.device("cpu")

    def update(self, result: DiscoveryResult, device: str = "cpu") -> None:
        """Update the reward function with new discovery results.

        Weights are proportional to |effect_size| * direction,
        normalized so sum of |weights| = reward_scale.
        """
        self._device = torch.device(device)
        self._indices = list(result.feature_indices)

        if not self._indices:
            self._weights = torch.zeros(0, device=self._device)
            return

        # Raw weights = effect_size (signed)
        raw = np.array(result.effect_sizes, dtype=np.float32)

        # Normalize so total magnitude = reward_scale
        total_mag = np.sum(np.abs(raw))
        if total_mag > 0:
            raw = raw * (self.config.reward_scale / total_mag)

        self._weights = torch.as_tensor(raw, dtype=torch.float32, device=self._device)

    def compute_reward(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute shaped reward from observations.

        Args:
            observations: (B, 3, 89) or (B, 89) tensor.

        Returns:
            (B, 3) or (B,) shaped reward tensor.
        """
        if len(self._indices) == 0:
            if observations.ndim == 3:
                return torch.zeros(
                    observations.shape[0],
                    observations.shape[1],
                    device=observations.device,
                    dtype=torch.float32,
                )
            return torch.zeros(
                observations.shape[0],
                device=observations.device,
                dtype=torch.float32,
            )

        weights = self._weights.to(observations.device)

        # Extract relevant feature columns
        # observations[..., indices] -> (..., K)
        selected = observations[..., self._indices]  # (..., K)

        # Weighted sum over features -> (...)
        return (selected * weights).sum(dim=-1)

    @property
    def num_features(self) -> int:
        return len(self._indices)


# --------------------------------------------------------------------------
# Outer-loop trainer
# --------------------------------------------------------------------------

class ContrastiveTrainer:
    """Outer loop: collect -> analyze -> shape -> train -> repeat."""

    def __init__(
        self,
        env: Any,
        ppo_trainer: Any,
        config: ContrastiveConfig = ContrastiveConfig(),
        game_config: GameConfig = DEFAULTS,
        device: str = "cpu",
        log_dir: str = "logs/contrastive",
    ) -> None:
        self.env = env
        self.ppo = ppo_trainer
        self.config = config
        self.game_config = game_config
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.collector = TrajectoryCollector(config=game_config)
        self.analyzer = ContrastiveAnalyzer(config=config)
        self.shaper = ContrastiveRewardShaper(config=config)

        self.discovery_log: list[dict[str, Any]] = []

    def run(self) -> list[dict[str, Any]]:
        """Run the full contrastive discovery loop.

        Returns:
            List of per-iteration metrics dicts.
        """
        all_metrics: list[dict[str, Any]] = []

        for iteration in range(self.config.num_iterations):
            logger.info("=== Contrastive iteration %d/%d ===",
                        iteration + 1, self.config.num_iterations)
            t0 = time.time()

            # 1. Collect trajectories with current policy
            logger.info("Collecting %d episodes...",
                        self.config.collection_episodes)
            data = self.collector.collect(
                env=self.env,
                policy=self.ppo.agent.network,
                num_episodes=self.config.collection_episodes,
                device=self.device,
            )
            n_steps = len(data["observations"])
            n_wins = int((data["outcomes"] > 0).sum())
            n_losses = int((data["outcomes"] < 0).sum())
            logger.info("Collected %d steps (%d win-steps, %d loss-steps)",
                        n_steps, n_wins, n_losses)

            if n_steps == 0:
                logger.warning(
                    "No non-draw episodes collected!  Skipping analysis "
                    "and training for this iteration."
                )
                continue

            # 2. Contrastive analysis
            result = self.analyzer.analyze(
                data["observations"], data["outcomes"]
            )

            if not result.feature_indices:
                if n_wins > 0 and n_losses == 0:
                    logger.warning(
                        "Agent wins every game -- no losses to contrast "
                        "against.  Keeping previous shaping features "
                        "(if any) and continuing training."
                    )
                elif n_losses > 0 and n_wins == 0:
                    logger.warning(
                        "Agent loses every game -- no wins to contrast "
                        "against.  Keeping previous shaping features "
                        "(if any) and continuing training."
                    )
                else:
                    logger.info("No significant features found this iteration.")
            else:
                logger.info(result.summary())

            # 3. Update reward shaper (only if new features were found,
            # otherwise keep the previous shaper state)
            if result.feature_indices:
                self.shaper.update(result, device=self.device)

            # 4. Train PPO with shaped reward
            logger.info("Training PPO for %d steps...",
                        self.config.steps_per_iteration)
            ppo_logs = self._train_with_shaped_reward(
                self.config.steps_per_iteration
            )

            elapsed = time.time() - t0

            # 5. Log iteration
            iteration_record = {
                "iteration": iteration,
                "elapsed_seconds": elapsed,
                "total_trajectory_steps": n_steps,
                "win_steps": n_wins,
                "loss_steps": n_losses,
                "num_features_discovered": len(result.feature_indices),
                "discovery": result.to_dict(),
                "ppo_final_metrics": ppo_logs[-1] if ppo_logs else {},
            }
            self.discovery_log.append(iteration_record)
            all_metrics.append(iteration_record)

            # Save discovery log after each iteration
            self._save_discovery_log()

            # Save checkpoint per iteration
            if self.log_dir:
                ckpt_dir = Path(self.log_dir) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"agent_iter_{iteration:03d}.pt"
                self.ppo.agent.save(ckpt_path)
                logger.info("Checkpoint saved: %s", ckpt_path)

            logger.info("Iteration %d complete in %.1fs, "
                        "%d features discovered",
                        iteration, elapsed,
                        len(result.feature_indices))

        return all_metrics

    def _train_with_shaped_reward(
        self, total_steps: int
    ) -> list[dict[str, float]]:
        """Train PPO, adding shaped reward to sparse env reward.

        If there are no shaping features, falls back to standard
        ``PPOTrainer.collect_rollout_vec`` for better performance.
        """
        steps_per_rollout = (
            self.ppo.rollout_steps * self.env.num_envs * 3
        )
        n_rollouts = max(total_steps // steps_per_rollout, 1)

        logs: list[dict[str, float]] = []

        for rollout_idx in range(n_rollouts):
            if self.shaper.num_features > 0:
                buffer = self._collect_rollout_with_shaping()
            else:
                # No shaping needed -- use the faster standard rollout
                buffer = self.ppo.collect_rollout_vec(self.env)
            metrics = self.ppo.train_step(buffer)
            metrics["rollout"] = rollout_idx
            logs.append(metrics)

            if (rollout_idx + 1) % 10 == 0:
                logger.info(
                    "  PPO rollout %d/%d: policy_loss=%.4f",
                    rollout_idx + 1, n_rollouts,
                    metrics.get("policy_loss", 0),
                )

        return logs

    def _collect_rollout_with_shaping(self) -> Any:
        """Collect rollout, adding shaped reward on top of sparse reward.

        Mirrors PPOTrainer.collect_rollout_vec but injects shaped reward.
        """
        from knockout.training.rollout_buffer import RolloutBuffer

        n = self.env.num_envs
        n_streams = n * 3
        buffer = RolloutBuffer(
            buffer_size=self.ppo.rollout_steps * n_streams,
            obs_dim=89,
            action_dim=2,
        )
        buffer.init_structured(self.ppo.rollout_steps, n_streams)

        obs, masks = self.env.reset()

        for step_idx in range(self.ppo.rollout_steps):
            obs_np = np.asarray(obs)
            masks_np = np.asarray(masks)

            flat_obs = obs_np.reshape(-1, 89)
            flat_masks = masks_np.reshape(-1)

            obs_t = torch.as_tensor(
                flat_obs, dtype=torch.float32, device=self.ppo.device
            )

            with torch.no_grad():
                raw_actions, log_probs, _, values = (
                    self.ppo.agent.network.get_action_and_value(obs_t)
                )

            raw_np = raw_actions.cpu().numpy()
            log_prob_np = log_probs.cpu().numpy()
            value_np = values.cpu().numpy()

            bounded = 1.0 / (1.0 + np.exp(-raw_np))
            scaled = bounded * np.array(
                [360.0, self.game_config.MAX_LAUNCH_FORCE]
            )
            actions_3d = scaled.reshape(n, 3, 2).astype(np.float32)

            next_obs, rewards, dones, next_masks, _infos = self.env.step(
                actions_3d
            )

            next_obs = np.asarray(next_obs)
            rewards = np.asarray(rewards)
            dones = np.asarray(dones)
            next_masks = np.asarray(next_masks)

            # Add shaped reward
            if self.shaper.num_features > 0:
                obs_tensor = torch.as_tensor(
                    obs_np, dtype=torch.float32, device=self.ppo.device
                )
                shaped = self.shaper.compute_reward(obs_tensor)
                rewards = rewards + shaped.cpu().numpy()

            flat_rewards = rewards.reshape(-1)
            flat_dones = np.repeat(dones, 3)

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

        buffer.compute_gae_structured(
            gamma=self.ppo.gamma, gae_lambda=self.ppo.gae_lambda
        )
        return buffer

    def _save_discovery_log(self) -> None:
        """Save discovery log to JSON."""
        path = self.log_dir / "discovery_log.json"
        with open(path, "w") as f:
            json.dump(self.discovery_log, f, indent=2)
