#!/usr/bin/env python3
"""Self-play training with contrastive reward discovery and checkpoint pool.

Combines two ideas:
  1. Checkpoint pool self-play: train against progressively harder versions
     of the agent (plus random opponents to prevent forgetting).
  2. Contrastive reward shaping: periodically re-discover which observation
     features distinguish winning from losing, and shape rewards accordingly.
     The features that matter *change* as opponents get smarter.

Designed to start from a pre-trained checkpoint (e.g. runs/contrastive/final_agent.pt)
that already beats random but fails against heuristic opponents.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import csv
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Ensure project root is importable
_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from knockout.agents.rl_agent import ActorCritic, RLAgent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.env.tensor_env import TensorVecEnv
from knockout.reward.contrastive import (
    ContrastiveAnalyzer,
    ContrastiveConfig,
    ContrastiveRewardShaper,
    DiscoveryResult,
    TrajectoryCollector,
)
from knockout.training.elo_rating import ELOTracker
from knockout.training.ppo import PPOTrainer
from knockout.training.rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown support
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_signal(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info(
        "Shutdown requested (signal %d). Will save after current iteration.",
        signum,
    )


# ---------------------------------------------------------------------------
# Checkpoint Pool
# ---------------------------------------------------------------------------

class CheckpointPool:
    """Pool of saved agent checkpoints for opponent sampling."""

    def __init__(self, pool_dir: Path, max_size: int = 20, device: str = "cpu"):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.device = device
        self.checkpoints: list[dict] = []  # [{path, elo, step}, ...]
        self._rng = np.random.default_rng(42)

    def add(self, agent: RLAgent, step: int, elo: float = 1000.0) -> Path:
        """Save agent checkpoint to pool. Evicts lowest-ELO if full."""
        path = self.pool_dir / f"pool_step_{step:07d}.pt"
        agent.save(path)
        entry = {"path": str(path), "elo": elo, "step": step}
        self.checkpoints.append(entry)

        # Evict oldest low-ELO checkpoint if over capacity
        if len(self.checkpoints) > self.max_size:
            # Keep the latest (just added) and the highest-ELO ones
            # Sort by ELO, remove the lowest that is not the most recent
            sorted_by_elo = sorted(
                self.checkpoints[:-1], key=lambda c: c["elo"]
            )
            evict = sorted_by_elo[0]
            self.checkpoints.remove(evict)
            logger.info(
                "Pool full (%d/%d). Evicted step=%d elo=%.0f",
                len(self.checkpoints), self.max_size,
                evict["step"], evict["elo"],
            )

        return path

    def sample_opponent(self, current_elo: float,
                        total_steps: int = 0) -> dict | None:
        """Sample an opponent from the pool.

        Random ratio scales down aggressively since agent starts pre-trained:
          Steps 0-500K:   50% random, 50% pool  (warm up self-play)
          Steps 500K-2M:  20% random, 80% pool  (arms race begins)
          Steps 2M-5M:    10% random, 90% pool  (deep self-play)
          Steps 5M+:      5% random,  95% pool  (near-pure self-play)
        """
        if not self.checkpoints:
            return None

        # Adaptive random ratio — aggressive since starting from pre-trained
        if total_steps < 500_000:
            random_ratio = 0.50
        elif total_steps < 2_000_000:
            t = (total_steps - 500_000) / 1_500_000
            random_ratio = 0.50 - 0.30 * t  # 50% → 20%
        elif total_steps < 5_000_000:
            t = (total_steps - 2_000_000) / 3_000_000
            random_ratio = 0.20 - 0.10 * t  # 20% → 10%
        else:
            random_ratio = 0.05

        roll = self._rng.random()

        if roll < random_ratio:
            # Random opponent
            return None

        # Split remaining probability: 60% ELO-weighted pool, 40% latest self
        pool_threshold = random_ratio + (1.0 - random_ratio) * 0.40
        if roll < pool_threshold:
            # Most recent checkpoint (self-play)
            return self.checkpoints[-1]

        # ELO-weighted sampling: prefer opponents within +-200 of current
        elos = np.array([c["elo"] for c in self.checkpoints])
        distances = np.abs(elos - current_elo)
        # Gaussian weighting centered at current_elo, sigma=200
        weights = np.exp(-0.5 * (distances / 200.0) ** 2)
        weights = weights / weights.sum()

        idx = self._rng.choice(len(self.checkpoints), p=weights)
        return self.checkpoints[idx]

    def load_opponent(self, checkpoint_info: dict) -> ActorCritic:
        """Load a frozen opponent network from a checkpoint."""
        path = Path(checkpoint_info["path"])
        net = ActorCritic()
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        net.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        return net

    def update_elo(self, step: int, new_elo: float) -> None:
        """Update the ELO rating for a checkpoint by step number."""
        for ckpt in self.checkpoints:
            if ckpt["step"] == step:
                ckpt["elo"] = new_elo
                break

    @property
    def size(self) -> int:
        return len(self.checkpoints)

    def save_index(self, path: Path) -> None:
        """Save pool index to JSON."""
        with open(path, "w") as f:
            json.dump(self.checkpoints, f, indent=2)


# ---------------------------------------------------------------------------
# Self-play rollout collection
# ---------------------------------------------------------------------------

def collect_selfplay_rollout(
    env: TensorVecEnv,
    current_net: ActorCritic,
    opponent_net: ActorCritic | None,
    rollout_steps: int,
    device: torch.device,
    config: GameConfig,
    shaper: ContrastiveRewardShaper | None = None,
) -> tuple[RolloutBuffer, dict]:
    """Collect a rollout with explicit Team A / Team B action control.

    Team A: current_net (learning, with gradients tracked for values/logprobs)
    Team B: opponent_net (frozen) or None (random actions via env)

    Returns:
        buffer: Filled RolloutBuffer for PPO training.
        stats: Dict with episode outcomes {wins, losses, draws, episodes}.
    """
    n = env.num_envs
    n_streams = n * 3  # one stream per Team-A agent per env
    buffer = RolloutBuffer(
        buffer_size=rollout_steps * n_streams,
        obs_dim=89,
        action_dim=2,
    )
    buffer.init_structured(rollout_steps, n_streams)

    obs_a, masks_a = env.reset()  # (B, 3, 89), (B, 3)

    wins = 0
    losses = 0
    draws = 0
    episodes = 0

    for step_idx in range(rollout_steps):
        obs_a_np = np.asarray(obs_a)
        masks_a_np = np.asarray(masks_a)

        # -- Team A actions (from current policy) --
        flat_obs_a = obs_a_np.reshape(-1, 89)  # (B*3, 89)
        flat_masks_a = masks_a_np.reshape(-1)   # (B*3,)

        obs_a_t = torch.as_tensor(
            flat_obs_a, dtype=torch.float32, device=device
        )

        with torch.no_grad():
            raw_actions_a, log_probs_a, _, values_a = (
                current_net.get_action_and_value(obs_a_t)
            )

        raw_a_np = raw_actions_a.cpu().numpy()       # (B*3, 2)
        log_prob_a_np = log_probs_a.cpu().numpy()     # (B*3,)
        value_a_np = values_a.cpu().numpy()            # (B*3,)

        # Scale Team A actions: sigmoid -> [0,1] -> game range
        bounded_a = 1.0 / (1.0 + np.exp(-raw_a_np))
        scaled_a = bounded_a * np.array([360.0, config.MAX_LAUNCH_FORCE])
        team_a_actions = scaled_a.reshape(n, 3, 2).astype(np.float32)

        # -- Team B actions --
        if opponent_net is not None:
            obs_b_np = env.get_team_b_obs()  # (B, 3, 89)
            flat_obs_b = obs_b_np.reshape(-1, 89)
            obs_b_t = torch.as_tensor(
                flat_obs_b, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                raw_actions_b, _, _, _ = opponent_net.get_action_and_value(obs_b_t)
            raw_b_np = raw_actions_b.cpu().numpy()
            bounded_b = 1.0 / (1.0 + np.exp(-raw_b_np))
            scaled_b = bounded_b * np.array([360.0, config.MAX_LAUNCH_FORCE])
            team_b_actions = scaled_b.reshape(n, 3, 2).astype(np.float32)
        else:
            team_b_actions = None  # env will use random

        # -- Step --
        next_obs_a, rewards, dones, next_masks_a, infos = env.step(
            team_a_actions, team_b_actions
        )
        next_obs_a = np.asarray(next_obs_a)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        next_masks_a = np.asarray(next_masks_a)

        # Add contrastive shaped reward if available
        if shaper is not None and shaper.num_features > 0:
            obs_a_tensor = torch.as_tensor(
                obs_a_np, dtype=torch.float32, device=device
            )
            shaped = shaper.compute_reward(obs_a_tensor)  # (B, 3)
            rewards = rewards + shaped.cpu().numpy()

        # Track outcomes
        for i in range(n):
            if dones[i]:
                episodes += 1
                info = infos[i]
                winner = info.get("winner", -1)
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1

        # Store in buffer (Team A only)
        flat_rewards = rewards.reshape(-1)       # (B*3,)
        flat_dones = np.repeat(dones, 3)          # (B*3,)

        buffer.add_step(
            step=step_idx,
            obs=flat_obs_a,
            actions=raw_a_np,
            log_probs=log_prob_a_np,
            rewards=flat_rewards,
            values=value_a_np,
            dones=flat_dones,
            masks=flat_masks_a,
        )

        obs_a = next_obs_a
        masks_a = next_masks_a

    buffer.compute_gae_structured(gamma=0.99, gae_lambda=0.95)

    stats = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "episodes": episodes,
        "win_rate": wins / max(episodes, 1),
    }
    return buffer, stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_vs_opponent(
    env: TensorVecEnv,
    agent_net: ActorCritic,
    opponent_net: ActorCritic | None,
    n_games: int,
    device: torch.device,
    config: GameConfig,
) -> dict:
    """Run n_games and return {wins, losses, draws, win_rate}."""
    obs_a, masks_a = env.reset()
    wins = 0
    losses = 0
    draws = 0
    games_done = 0
    max_steps = 5000  # safety cap

    for _ in range(max_steps):
        if games_done >= n_games:
            break

        obs_a_np = np.asarray(obs_a)
        n = env.num_envs

        # Team A
        flat_a = obs_a_np.reshape(-1, 89)
        obs_a_t = torch.as_tensor(flat_a, dtype=torch.float32, device=device)
        with torch.no_grad():
            raw_a, _, _, _ = agent_net.get_action_and_value(obs_a_t)
        raw_a_np = raw_a.cpu().numpy()
        bounded_a = 1.0 / (1.0 + np.exp(-raw_a_np))
        scaled_a = bounded_a * np.array([360.0, config.MAX_LAUNCH_FORCE])
        team_a_actions = scaled_a.reshape(n, 3, 2).astype(np.float32)

        # Team B
        if opponent_net is not None:
            obs_b_np = env.get_team_b_obs()
            flat_b = obs_b_np.reshape(-1, 89)
            obs_b_t = torch.as_tensor(flat_b, dtype=torch.float32, device=device)
            with torch.no_grad():
                raw_b, _, _, _ = opponent_net.get_action_and_value(obs_b_t)
            raw_b_np = raw_b.cpu().numpy()
            bounded_b = 1.0 / (1.0 + np.exp(-raw_b_np))
            scaled_b = bounded_b * np.array([360.0, config.MAX_LAUNCH_FORCE])
            team_b_actions = scaled_b.reshape(n, 3, 2).astype(np.float32)
        else:
            team_b_actions = None

        obs_a, _, dones, masks_a, infos = env.step(team_a_actions, team_b_actions)

        for i in range(n):
            if dones[i] and games_done < n_games:
                games_done += 1
                winner = infos[i].get("winner", -1)
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
                else:
                    draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": games_done,
        "win_rate": wins / max(games_done, 1),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_self_play(args: argparse.Namespace) -> None:
    """Main self-play training loop with contrastive reward discovery."""
    global _shutdown_requested

    # -- Setup ---------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = output_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    # -- PID file (prevent duplicate launches) -------------------------------
    pid_file = output_dir / "train.pid"
    if pid_file.exists():
        old_pid = int(pid_file.read_text().strip())
        try:
            os.kill(old_pid, 0)  # signal 0 = check if alive
            print(
                f"ERROR: Training already running (PID {old_pid}). Exiting.",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)
        except OSError:
            pass  # Old process is dead, we can proceed
    pid_file.write_text(str(os.getpid()))

    def _cleanup_pid():
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_cleanup_pid)

    # -- Logging (direct file handler, no reliance on stdout redirect) -------
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # File handler with immediate flush
    file_handler = logging.FileHandler(output_dir / "train.log", mode="a")
    file_handler.setFormatter(logging.Formatter(log_fmt))
    root_logger.addHandler(file_handler)
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_fmt))
    root_logger.addHandler(console_handler)

    # -- Signal handlers (graceful shutdown) ---------------------------------
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    logger.info("Device: %s, num_envs: %d", device, args.num_envs)

    config = DEFAULTS

    # -- Environment ---------------------------------------------------------
    env = TensorVecEnv(
        num_envs=args.num_envs,
        config=config,
        device=device_str,
    )

    # -- PPO Trainer (manages learning agent + optimizer) --------------------
    ppo = PPOTrainer(
        config=config,
        lr=args.lr,
        device=device_str,
        num_envs=args.num_envs,
        backend="tensor",
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
    )

    # -- Load starting checkpoint --------------------------------------------
    start_ckpt = Path(args.start_checkpoint)
    if start_ckpt.exists():
        ppo.agent.load(start_ckpt)
        logger.info("Loaded starting checkpoint: %s", start_ckpt)
    else:
        logger.warning(
            "Starting checkpoint not found: %s. Training from scratch.",
            start_ckpt,
        )

    current_net = ppo.agent.network  # alias for readability

    # -- Checkpoint pool + ELO -----------------------------------------------
    pool = CheckpointPool(pool_dir, max_size=args.pool_size, device=device_str)
    elo_tracker = ELOTracker()
    elo_tracker.register("learner")

    # Seed the pool with the initial checkpoint
    pool.add(ppo.agent, step=0, elo=1000.0)

    # -- Contrastive reward shaping ------------------------------------------
    contrastive_cfg = ContrastiveConfig(
        collection_episodes=500,
        min_effect_size=0.3,
        max_reward_features=10,
        reward_scale=0.5,
    )
    analyzer = ContrastiveAnalyzer(contrastive_cfg)
    shaper = ContrastiveRewardShaper(contrastive_cfg)
    collector = TrajectoryCollector(config=config)

    current_elo = 1000.0
    total_steps = 0
    csv_rows: list[dict] = []

    # -- Compute steps budget per iteration ----------------------------------
    steps_per_rollout = args.rollout_steps * args.num_envs * 3
    rollouts_per_iter = max(args.steps_per_iteration // steps_per_rollout, 1)

    logger.info(
        "Starting self-play training: %d iterations, %d rollouts/iter "
        "(%d steps/rollout, ~%d steps/iter)",
        args.iterations, rollouts_per_iter, steps_per_rollout,
        rollouts_per_iter * steps_per_rollout,
    )

    # -- Main loop -----------------------------------------------------------
    for iteration in range(args.iterations):
        if _shutdown_requested:
            logger.info("Shutdown requested before iteration %d. Stopping.", iteration + 1)
            break

        t0 = time.time()
        logger.info(
            "=== Iteration %d/%d  |  ELO: %.0f  |  Pool: %d ===",
            iteration + 1, args.iterations, current_elo, pool.size,
        )

        # 1. Sample opponent from pool
        opponent_info = pool.sample_opponent(current_elo, total_steps=total_steps)
        # Log the adaptive ratio
        if total_steps < 500_000:
            rand_pct = 50
        elif total_steps < 2_000_000:
            rand_pct = int(50 - 30 * (total_steps - 5e5) / 1.5e6)
        elif total_steps < 5_000_000:
            rand_pct = int(20 - 10 * (total_steps - 2e6) / 3e6)
        else:
            rand_pct = 5

        if opponent_info is not None:
            opponent_net = pool.load_opponent(opponent_info)
            opp_label = f"pool_step_{opponent_info['step']}"
            opp_elo = opponent_info["elo"]
            logger.info(
                "Opponent: %s (elo=%.0f) [random_ratio=%d%%]",
                opp_label, opp_elo, rand_pct,
            )
        else:
            opponent_net = None
            opp_label = "random"
            opp_elo = 500.0  # nominal ELO for random
            logger.info("Opponent: random")

        # 2. Collect rollouts and train
        iter_wins = 0
        iter_losses = 0
        iter_draws = 0
        iter_episodes = 0
        iter_policy_loss = 0.0
        iter_value_loss = 0.0
        iter_entropy = 0.0

        for r in range(rollouts_per_iter):
            if _shutdown_requested:
                logger.info("Shutdown requested during rollout collection. Breaking.")
                break

            buffer, stats = collect_selfplay_rollout(
                env=env,
                current_net=current_net,
                opponent_net=opponent_net,
                rollout_steps=args.rollout_steps,
                device=device,
                config=config,
                shaper=shaper if shaper.num_features > 0 else None,
            )

            metrics = ppo.train_step(buffer)

            iter_wins += stats["wins"]
            iter_losses += stats["losses"]
            iter_draws += stats["draws"]
            iter_episodes += stats["episodes"]
            iter_policy_loss += metrics.get("policy_loss", 0)
            iter_value_loss += metrics.get("value_loss", 0)
            iter_entropy += metrics.get("entropy", 0)
            total_steps += steps_per_rollout

            if (r + 1) % max(rollouts_per_iter // 5, 1) == 0:
                wr = iter_wins / max(iter_episodes, 1)
                logger.info(
                    "  rollout %d/%d: win_rate=%.2f (%d/%d), "
                    "policy_loss=%.4f",
                    r + 1, rollouts_per_iter, wr, iter_wins, iter_episodes,
                    metrics.get("policy_loss", 0),
                )

        avg_policy_loss = iter_policy_loss / max(rollouts_per_iter, 1)
        avg_value_loss = iter_value_loss / max(rollouts_per_iter, 1)
        avg_entropy = iter_entropy / max(rollouts_per_iter, 1)
        vs_opp_winrate = iter_wins / max(iter_episodes, 1)

        # 3. Contrastive analysis (every 5 iterations)
        features_discovered = shaper.num_features
        if (iteration + 1) % 5 == 0 and iteration > 0:
            logger.info("Running contrastive analysis on recent self-play...")
            data = collector.collect(
                env=env,
                policy=current_net,
                num_episodes=500,
                device=device_str,
            )
            if len(data["observations"]) > 0:
                result = analyzer.analyze(
                    data["observations"], data["outcomes"]
                )
                if result.feature_indices:
                    shaper.update(result, device=device_str)
                    features_discovered = len(result.feature_indices)
                    logger.info(result.summary())
                else:
                    logger.info("No new significant features found.")
            else:
                logger.warning("No episodes collected for contrastive analysis.")

        # 4. Save checkpoint to pool
        pool.add(ppo.agent, step=total_steps, elo=current_elo)

        # 5. Evaluate vs random pool members for ELO update
        logger.info("Evaluating vs pool members for ELO update...")

        # Always evaluate vs random
        random_result = evaluate_vs_opponent(
            env, current_net, None, n_games=50, device=device, config=config,
        )
        vs_random_wr = random_result["win_rate"]
        logger.info(
            "  vs random: %.0f%% (%d/%d)",
            vs_random_wr * 100, random_result["wins"], random_result["games"],
        )

        # Evaluate vs 3-5 random pool members
        n_eval_opponents = min(5, pool.size)
        eval_indices = np.random.default_rng(iteration).choice(
            pool.size, size=n_eval_opponents, replace=False
        )
        vs_pool_wins = 0
        vs_pool_games = 0

        for idx in eval_indices:
            ckpt = pool.checkpoints[idx]
            eval_opp = pool.load_opponent(ckpt)
            eval_name = f"pool_step_{ckpt['step']}"
            elo_tracker.register(eval_name)

            result = evaluate_vs_opponent(
                env, current_net, eval_opp, n_games=50, device=device, config=config,
            )

            # Update ELO
            score_a = result["win_rate"]
            new_learner_elo, new_opp_elo = elo_tracker.update(
                "learner", eval_name, score_a
            )
            pool.update_elo(ckpt["step"], new_opp_elo)

            vs_pool_wins += result["wins"]
            vs_pool_games += result["games"]

            logger.info(
                "  vs %s: %.0f%% (%d/%d) -> learner_elo=%.0f",
                eval_name, result["win_rate"] * 100,
                result["wins"], result["games"], new_learner_elo,
            )

        current_elo = elo_tracker.get_rating("learner")
        vs_pool_wr = vs_pool_wins / max(vs_pool_games, 1)

        elapsed = time.time() - t0

        # 6. Log metrics
        row = {
            "step": total_steps,
            "elo": current_elo,
            "vs_random_winrate": vs_random_wr,
            "vs_pool_winrate": vs_pool_wr,
            "vs_opponent_winrate": vs_opp_winrate,
            "features_discovered": features_discovered,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "pool_size": pool.size,
            "opponent": opp_label,
            "iteration": iteration,
            "episodes": iter_episodes,
            "elapsed_seconds": elapsed,
        }
        csv_rows.append(row)

        # Write CSV after each iteration (so partial results are saved)
        csv_path = output_dir / "training_log.csv"
        _write_csv(csv_rows, csv_path)

        # Save pool index
        pool.save_index(output_dir / "pool_index.json")

        logger.info(
            "Iteration %d complete in %.1fs: ELO=%.0f, "
            "vs_random=%.0f%%, vs_pool=%.0f%%, "
            "features=%d, episodes=%d",
            iteration + 1, elapsed, current_elo,
            vs_random_wr * 100, vs_pool_wr * 100,
            features_discovered, iter_episodes,
        )

    # -- Final save -----------------------------------------------------------
    final_path = output_dir / "final_agent.pt"
    ppo.agent.save(final_path)
    logger.info("Saved final agent to %s", final_path)

    # Leaderboard
    leaderboard = elo_tracker.get_leaderboard()
    logger.info("Final ELO leaderboard:")
    for name, elo in leaderboard[:10]:
        logger.info("  %s: %.0f", name, elo)

    env.close()
    _cleanup_pid()
    logger.info("Training complete. %d iterations, final ELO=%.0f", args.iterations, current_elo)


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write training log rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "step", "elo", "vs_random_winrate", "vs_pool_winrate",
        "vs_opponent_winrate", "features_discovered",
        "policy_loss", "value_loss", "entropy",
        "pool_size", "opponent", "iteration", "episodes", "elapsed_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-play training with contrastive reward discovery"
    )
    parser.add_argument(
        "--num-envs", type=int, default=64,
        help="Number of parallel environments (default: 64)",
    )
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Number of self-play iterations (default: 20)",
    )
    parser.add_argument(
        "--steps-per-iteration", type=int, default=200_000,
        help="Approximate training steps per iteration (default: 200000)",
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=128,
        help="PPO rollout length per collection (default: 128)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="PPO mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=20,
        help="Maximum checkpoints in opponent pool (default: 20)",
    )
    parser.add_argument(
        "--start-checkpoint", type=str,
        default="runs/contrastive/final_agent.pt",
        help="Path to pre-trained checkpoint to start from",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/self_play",
        help="Output directory for checkpoints and logs (default: runs/self_play)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Torch device: cpu, cuda, or auto (default: auto)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_self_play(parse_args())
