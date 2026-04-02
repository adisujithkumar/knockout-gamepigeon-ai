"""Monitored training harness with evaluation, stall detection, and auto-intervention.

Wraps PPO or MAPPO trainers with periodic evaluation against baselines,
curriculum learning, automatic hyperparameter adjustment when training
stalls, and full checkpoint/logging support.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from knockout.agents.base import Agent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.agents.random_agent import RandomAgent
from knockout.agents.rl_agent import RLAgent
from knockout.agents.mappo_agent import MAPPOAgent, MAPPOEvalAgent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.training.evaluation import run_match


# -- Curriculum stages --------------------------------------------------------

STAGE_RANDOM = "random"
STAGE_MIXED = "mixed"
STAGE_HEURISTIC = "heuristic"

STAGE_ORDER = [STAGE_RANDOM, STAGE_MIXED, STAGE_HEURISTIC]


# -- Intervention names -------------------------------------------------------

INTERVENTION_ENTROPY = "entropy_boost"
INTERVENTION_LR = "lr_reduction"
INTERVENTION_CURRICULUM = "curriculum_switch"
INTERVENTION_ROLLOUT = "rollout_increase"
INTERVENTION_RESET = "reset_to_best"


class MonitoredTrainer:
    """Training harness with periodic evaluation, stall detection, and auto-intervention.

    Monitors training health by:
    1. Evaluating against baselines every N steps
    2. Tracking learning curves (win rate, loss, entropy)
    3. Detecting stalls (no improvement for K evaluations)
    4. Auto-adjusting hyperparameters or curriculum when stalled
    5. Saving checkpoints at every evaluation point
    6. Logging everything to CSV for post-analysis
    """

    def __init__(
        self,
        algorithm: str = "ppo",
        config: GameConfig = DEFAULTS,
        device: str = "cuda",
        num_envs: int = 4096,
        backend: str = "tensor",
        eval_interval: int = 50000,
        eval_games: int = 50,
        stall_patience: int = 5,
        checkpoint_dir: str = "checkpoints/monitored/",
        log_path: str = "logs/training_monitor.csv",
        lr: float = 3e-4,
        rollout_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 4,
        entropy_coef: float = 0.01,
    ):
        self.algorithm = algorithm.lower()
        self.config = config
        self.device = device
        self.num_envs = num_envs
        self.backend = backend
        self.eval_interval = eval_interval
        self.eval_games = eval_games
        self.stall_patience = stall_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_path = Path(log_path)

        # Hyperparameters (mutable -- interventions may change them)
        self.lr = lr
        self.rollout_steps = rollout_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef

        # Curriculum state
        self.curriculum_stage = STAGE_RANDOM
        self.curriculum_stage_index = 0

        # Tracking state
        self.best_wr_heuristic: float = -1.0
        self.best_wr_random: float = -1.0
        self.best_checkpoint_path: str | None = None
        self.evals_since_improvement: int = 0
        self.interventions: list[dict] = []
        self.eval_log: list[dict] = []
        self.intervention_index: int = 0  # which intervention to try next

        # Timing
        self._train_start: float = 0.0

        # Lazy-created trainer/env (built in train())
        self._trainer = None
        self._vec_env = None

    # ------------------------------------------------------------------
    # Trainer creation helpers
    # ------------------------------------------------------------------

    def _make_opponent_factory(self) -> Callable[[], Agent]:
        """Return opponent factory matching current curriculum stage."""
        if self.curriculum_stage == STAGE_RANDOM:
            return lambda: RandomAgent("opp")
        elif self.curriculum_stage == STAGE_HEURISTIC:
            return lambda: HeuristicAgent("opp")
        else:
            # Mixed: 50/50
            counter = {"n": 0}

            def _factory():
                counter["n"] += 1
                if counter["n"] % 2 == 0:
                    return HeuristicAgent("opp")
                return RandomAgent("opp")

            return _factory

    def _build_trainer(self):
        """Create trainer + vectorized env according to current settings."""
        if self.algorithm == "mappo":
            return self._build_mappo_trainer()
        return self._build_ppo_trainer()

    def _build_ppo_trainer(self):
        from knockout.training.ppo import PPOTrainer

        trainer = PPOTrainer(
            config=self.config,
            lr=self.lr,
            rollout_steps=self.rollout_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            entropy_coef=self.entropy_coef,
            device=self.device,
            num_envs=self.num_envs,
            opponent_factory=self._make_opponent_factory(),
            backend=self.backend,
        )
        return trainer

    def _build_mappo_trainer(self):
        from knockout.training.mappo import MAPPOTrainer

        trainer = MAPPOTrainer(
            config=self.config,
            num_envs=self.num_envs,
            lr=self.lr,
            rollout_steps=self.rollout_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            entropy_coef=self.entropy_coef,
            device=self.device,
            backend=self.backend,
            opponent_factory=self._make_opponent_factory(),
        )
        return trainer

    def _get_vec_env(self, trainer):
        """Return the vectorized env from the trainer."""
        if self.algorithm == "mappo":
            return trainer.vec_env
        # PPO: create it the same way _train_vec does
        if self.num_envs > 1:
            if self.backend == "tensor":
                from knockout.env.tensor_env import TensorVecEnv

                return TensorVecEnv(
                    num_envs=self.num_envs,
                    config=self.config,
                    device=self.device,
                )
            else:
                from knockout.training.vec_env import SingleTeamVecEnv

                return SingleTeamVecEnv(
                    num_envs=self.num_envs,
                    opponent_factory=self._make_opponent_factory(),
                    config=self.config,
                )
        return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, n_games: int | None = None) -> dict:
        """Run evaluation games vs Random and Heuristic using Pymunk backend.

        Returns dict with win_rate_vs_random, win_rate_vs_heuristic, avg_steps.
        """
        if n_games is None:
            n_games = self.eval_games

        wins_random = 0
        wins_heuristic = 0
        total_steps_random = 0
        total_steps_heuristic = 0
        total_reward_random = 0.0
        total_reward_heuristic = 0.0

        # Build Team A from trained agent
        team_a = self._make_eval_team_a()

        for game_idx in range(n_games):
            # vs Random
            team_b_random = {
                f"penguin_{k}": RandomAgent(f"opp_{k}", seed=game_idx * 100 + k)
                for k in range(3, 6)
            }
            result = run_match(
                team_a, team_b_random, config=self.config,
                seed=game_idx, max_steps=200,
            )
            if result["winner"] == 0:
                wins_random += 1
            total_steps_random += result["steps"]
            # Proxy reward: +1 win, -1 loss, 0 draw
            total_reward_random += (
                1.0 if result["winner"] == 0
                else (-1.0 if result["winner"] == 1 else 0.0)
            )

            # vs Heuristic
            team_b_heuristic = {
                f"penguin_{k}": HeuristicAgent(f"opp_{k}", seed=game_idx * 100 + k)
                for k in range(3, 6)
            }
            result = run_match(
                team_a, team_b_heuristic, config=self.config,
                seed=game_idx + 10000, max_steps=200,
            )
            if result["winner"] == 0:
                wins_heuristic += 1
            total_steps_heuristic += result["steps"]
            total_reward_heuristic += (
                1.0 if result["winner"] == 0
                else (-1.0 if result["winner"] == 1 else 0.0)
            )

        return {
            "win_rate_vs_random": wins_random / max(n_games, 1),
            "win_rate_vs_heuristic": wins_heuristic / max(n_games, 1),
            "avg_steps_vs_random": total_steps_random / max(n_games, 1),
            "avg_steps_vs_heuristic": total_steps_heuristic / max(n_games, 1),
            "avg_reward_random": total_reward_random / max(n_games, 1),
            "avg_reward_heuristic": total_reward_heuristic / max(n_games, 1),
        }

    def _make_eval_team_a(self) -> dict[str, Agent]:
        """Build team A dict for evaluation from the current trained agent."""
        if self.algorithm == "mappo":
            mappo_agent = self._trainer.agent
            mappo_agent.eval()
            team_a = {}
            for k in range(3):
                team_a[f"penguin_{k}"] = MAPPOEvalAgent(
                    f"penguin_{k}", mappo_agent, agent_index=k,
                )
            return team_a
        else:
            rl_agent = self._trainer.agent
            rl_agent.network.eval()
            team_a = {f"penguin_{k}": rl_agent for k in range(3)}
            return team_a

    # ------------------------------------------------------------------
    # Stall detection
    # ------------------------------------------------------------------

    def _check_stall(self, eval_result: dict, train_metrics: dict) -> str | None:
        """Check for training pathologies.

        Returns a string describing the problem, or None if healthy.
        """
        # Entropy collapse
        entropy = train_metrics.get("entropy", 1.0)
        if entropy < 0.1:
            return "entropy_collapse"

        # Loss divergence
        ploss = train_metrics.get("policy_loss", 0.0)
        vloss = train_metrics.get("value_loss", 0.0)
        if not math.isfinite(ploss) or not math.isfinite(vloss):
            return "loss_divergence"
        if abs(ploss) > 10.0 or abs(vloss) > 100.0:
            return "loss_divergence"

        # Win rate regression (compare to best)
        wr = eval_result["win_rate_vs_heuristic"]
        if self.best_wr_heuristic > 0 and wr < self.best_wr_heuristic - 0.15:
            return "win_rate_regression"

        # No improvement
        if wr > self.best_wr_heuristic:
            self.best_wr_heuristic = wr
            self.evals_since_improvement = 0
            return None

        # Also track random win rate
        wr_rand = eval_result["win_rate_vs_random"]
        if wr_rand > self.best_wr_random:
            self.best_wr_random = wr_rand
            # Still count as improvement if random win rate improved
            self.evals_since_improvement = 0
            return None

        self.evals_since_improvement += 1
        if self.evals_since_improvement >= self.stall_patience:
            return "stall"

        return None

    # ------------------------------------------------------------------
    # Auto-intervention
    # ------------------------------------------------------------------

    def _intervene(self, problem: str, step: int) -> str:
        """Apply the next intervention in sequence. Returns description."""
        interventions_sequence = [
            self._intervene_entropy,
            self._intervene_lr,
            self._intervene_curriculum,
            self._intervene_rollout,
            self._intervene_reset,
        ]

        # Pick intervention based on index
        idx = self.intervention_index % len(interventions_sequence)
        description = interventions_sequence[idx](problem, step)
        self.intervention_index += 1

        # Reset stall counter after any intervention
        self.evals_since_improvement = 0

        self.interventions.append({
            "step": step,
            "problem": problem,
            "action": description,
        })

        return description

    def _intervene_entropy(self, problem: str, step: int) -> str:
        old = self.entropy_coef
        self.entropy_coef = min(self.entropy_coef * 5.0, 0.1)
        self._apply_entropy_coef()
        return f"increased entropy_coef {old:.4f} -> {self.entropy_coef:.4f}"

    def _intervene_lr(self, problem: str, step: int) -> str:
        old = self.lr
        self.lr = self.lr * 0.5
        self._apply_lr()
        return f"reduced lr {old:.6f} -> {self.lr:.6f}"

    def _intervene_curriculum(self, problem: str, step: int) -> str:
        old_stage = self.curriculum_stage
        wr_random = self.best_wr_random

        # If struggling vs heuristic and haven't mastered random, go back
        if self.curriculum_stage in (STAGE_HEURISTIC, STAGE_MIXED) and wr_random < 0.8:
            self.curriculum_stage = STAGE_RANDOM
            self.curriculum_stage_index = 0
        elif self.curriculum_stage == STAGE_RANDOM:
            self.curriculum_stage = STAGE_MIXED
            self.curriculum_stage_index = 1
        elif self.curriculum_stage == STAGE_MIXED:
            self.curriculum_stage = STAGE_HEURISTIC
            self.curriculum_stage_index = 2
        else:
            self.curriculum_stage = STAGE_MIXED
            self.curriculum_stage_index = 1

        self._rebuild_env()
        return f"curriculum switch {old_stage} -> {self.curriculum_stage}"

    def _intervene_rollout(self, problem: str, step: int) -> str:
        old = self.rollout_steps
        self.rollout_steps = min(self.rollout_steps * 2, 512)
        # Need to rebuild trainer for new rollout length
        self._rebuild_trainer_preserve_weights()
        return f"increased rollout_steps {old} -> {self.rollout_steps}"

    def _intervene_reset(self, problem: str, step: int) -> str:
        if self.best_checkpoint_path and Path(self.best_checkpoint_path).exists():
            self._load_checkpoint(self.best_checkpoint_path)
            # Also reset entropy to default
            self.entropy_coef = 0.01
            self.lr = 3e-4
            self._apply_entropy_coef()
            self._apply_lr()
            return f"reset to best checkpoint {self.best_checkpoint_path}"
        return "reset skipped (no best checkpoint)"

    def _apply_entropy_coef(self):
        """Push current entropy_coef into the trainer."""
        if self._trainer is not None:
            self._trainer.entropy_coef = self.entropy_coef

    def _apply_lr(self):
        """Push current lr into the trainer optimizer."""
        if self._trainer is not None:
            for pg in self._trainer.optimizer.param_groups:
                pg["lr"] = self.lr

    def _rebuild_env(self):
        """Rebuild the vec env with new opponent factory (for curriculum switch).

        For PPO, we swap the vec_env on the trainer.
        For MAPPO, we close and replace vec_env.
        """
        if self._trainer is None:
            return

        if self.algorithm == "mappo":
            self._trainer.vec_env.close()
            if self.backend == "tensor":
                from knockout.env.tensor_env import TensorVecEnv

                self._trainer.vec_env = TensorVecEnv(
                    num_envs=self.num_envs,
                    config=self.config,
                    device=self.device,
                )
            else:
                from knockout.training.vec_env import SingleTeamVecEnv

                self._trainer.vec_env = SingleTeamVecEnv(
                    num_envs=self.num_envs,
                    opponent_factory=self._make_opponent_factory(),
                    config=self.config,
                )
        else:
            # PPO: close and recreate vec_env
            if self._vec_env is not None:
                self._vec_env.close()
            self._vec_env = self._get_vec_env(self._trainer)

    def _rebuild_trainer_preserve_weights(self):
        """Rebuild trainer but keep trained weights."""
        if self._trainer is None:
            return

        # Save weights to temp
        tmp_path = self.checkpoint_dir / "_tmp_rebuild.pt"
        self._save_checkpoint(str(tmp_path))

        # Close old env
        if self.algorithm == "mappo":
            self._trainer.vec_env.close()
        elif self._vec_env is not None:
            self._vec_env.close()

        # Recreate
        self._trainer = self._build_trainer()
        if self.algorithm == "ppo" and self.num_envs > 1:
            self._vec_env = self._get_vec_env(self._trainer)

        # Load weights back
        self._load_checkpoint(str(tmp_path))
        tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Curriculum advancement
    # ------------------------------------------------------------------

    def _check_curriculum_advance(self, eval_result: dict, step: int) -> str | None:
        """Auto-advance curriculum if thresholds met. Returns message or None."""
        wr_random = eval_result["win_rate_vs_random"]
        wr_heuristic = eval_result["win_rate_vs_heuristic"]

        if self.curriculum_stage == STAGE_RANDOM and wr_random >= 0.8:
            old = self.curriculum_stage
            self.curriculum_stage = STAGE_MIXED
            self.curriculum_stage_index = 1
            self._rebuild_env()
            return f"curriculum advanced {old} -> {self.curriculum_stage} (random WR={wr_random:.0%})"

        if self.curriculum_stage == STAGE_MIXED and wr_heuristic >= 0.4:
            old = self.curriculum_stage
            self.curriculum_stage = STAGE_HEURISTIC
            self.curriculum_stage_index = 2
            self._rebuild_env()
            return f"curriculum advanced {old} -> {self.curriculum_stage} (heuristic WR={wr_heuristic:.0%})"

        return None

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: str):
        """Save agent weights to path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.algorithm == "mappo":
            self._trainer.agent.save(p)
        else:
            self._trainer.agent.save(p)

    def _load_checkpoint(self, path: str):
        """Load agent weights from path."""
        p = Path(path)
        if not p.exists():
            return
        if self.algorithm == "mappo":
            self._trainer.agent.load(p)
        else:
            self._trainer.agent.load(p)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_eval(self, row: dict):
        """Append a row to eval_log and write CSV."""
        self.eval_log.append(row)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(row.keys())
        write_header = not self.log_path.exists() or len(self.eval_log) == 1
        mode = "w" if write_header else "a"
        with open(self.log_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                # If rewriting header, write all rows
                if len(self.eval_log) > 1:
                    for r in self.eval_log[:-1]:
                        writer.writerow(r)
            writer.writerow(row)

    def _save_config(self):
        """Save training config as JSON."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "algorithm": self.algorithm,
            "device": self.device,
            "num_envs": self.num_envs,
            "backend": self.backend,
            "eval_interval": self.eval_interval,
            "eval_games": self.eval_games,
            "stall_patience": self.stall_patience,
            "lr": self.lr,
            "rollout_steps": self.rollout_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "entropy_coef": self.entropy_coef,
        }
        with open(self.checkpoint_dir / "training_config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    # ------------------------------------------------------------------
    # Training step helpers
    # ------------------------------------------------------------------

    def _compute_effective_rollout_steps(self) -> int:
        """Choose a rollout_steps so that one rollout fits within eval_interval.

        With many parallel envs each rollout produces
        ``rollout_steps * num_envs * agents_per_env`` env steps.  If that
        exceeds eval_interval we'd never get intermediate evaluations.
        This helper caps the per-rollout step count so evaluations
        fire at the requested granularity while keeping each rollout large
        enough for meaningful PPO/GAE updates.
        """
        agents_per_env = 1 if self.algorithm == "mappo" else 3
        steps_per_tick = self.num_envs * agents_per_env
        max_rollout = max(self.eval_interval // max(steps_per_tick, 1), 1)
        # Use the smaller of the configured value and the computed cap,
        # but always at least 4 to keep updates meaningful.
        return max(min(self.rollout_steps, max_rollout), 4)

    def _compute_steps_per_rollout(self, eff_rollout_steps: int) -> int:
        """Env steps for one rollout with given effective rollout_steps."""
        if self.algorithm == "mappo":
            return eff_rollout_steps * self.num_envs
        if self.num_envs > 1:
            return eff_rollout_steps * self.num_envs * 3
        return eff_rollout_steps * 3

    def _do_rollout_and_train(self, eff_rollout_steps: int) -> dict:
        """Collect one rollout and do one PPO/MAPPO update. Returns metrics."""
        if self.algorithm == "mappo":
            buffer = self._trainer.collect_rollout(n_steps=eff_rollout_steps)
            return self._trainer.train_step(buffer)
        else:
            # Temporarily override trainer rollout_steps
            orig = self._trainer.rollout_steps
            self._trainer.rollout_steps = eff_rollout_steps
            try:
                if self.num_envs > 1 and self._vec_env is not None:
                    buffer = self._trainer.collect_rollout_vec(self._vec_env)
                else:
                    from knockout.env.penguin_env import PenguinEnv

                    env = PenguinEnv(config=self.config)
                    buffer = self._trainer.collect_rollout(env)
                    env.close()
                return self._trainer.train_step(buffer)
            finally:
                self._trainer.rollout_steps = orig

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_step(step: int) -> str:
        """Format step count for display: 1500000 -> '1.5M', 100000 -> '100K'."""
        if step >= 1_000_000:
            return f"{step / 1_000_000:.1f}M"
        if step >= 1000:
            return f"{step // 1000}K"
        return str(step)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format elapsed seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {secs}s"

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 5_000_000) -> list[dict]:
        """Main training loop with monitoring.

        Returns list of evaluation log dicts.
        """
        self._train_start = time.time()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_config()

        # Build trainer
        self._trainer = self._build_trainer()
        if self.algorithm == "ppo" and self.num_envs > 1:
            self._vec_env = self._get_vec_env(self._trainer)

        # Set train mode
        if self.algorithm == "mappo":
            self._trainer.agent.train()
        else:
            self._trainer.agent.network.train()

        # Choose effective rollout length so evaluations fire frequently
        eff_rollout_steps = self._compute_effective_rollout_steps()
        steps_per_rollout = self._compute_steps_per_rollout(eff_rollout_steps)
        total_steps = 0
        last_eval_step = 0
        last_metrics: dict = {}

        print(f"=== Monitored Training: {self.algorithm.upper()} ===")
        print(
            f"Backend: {self.backend} | Envs: {self.num_envs} | "
            f"Device: {self.device} | Curriculum: {self.curriculum_stage}"
        )
        print(
            f"Eval every {self._format_step(self.eval_interval)} steps | "
            f"{self.eval_games} games | Patience: {self.stall_patience}"
        )
        print(
            f"Effective rollout steps: {eff_rollout_steps} "
            f"({self._format_step(steps_per_rollout)} env steps/rollout)"
        )
        print()

        try:
            while total_steps < total_timesteps:
                # Train for one rollout
                metrics = self._do_rollout_and_train(eff_rollout_steps)
                total_steps += steps_per_rollout
                last_metrics = metrics

                # Check if it's time to evaluate
                if total_steps - last_eval_step >= self.eval_interval:
                    last_eval_step = total_steps
                    self._run_evaluation(total_steps, metrics, total_timesteps)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

        # Final cleanup
        self._close()
        self._print_summary(total_steps)
        return self.eval_log

    def _run_evaluation(self, total_steps: int, train_metrics: dict, total_timesteps: int):
        """Run evaluation, stall checks, curriculum checks, and logging."""
        step_label = self._format_step(total_steps)

        # 1. Evaluate
        eval_result = self._evaluate()

        wr_rand = eval_result["win_rate_vs_random"]
        wr_heur = eval_result["win_rate_vs_heuristic"]
        ploss = train_metrics.get("policy_loss", 0.0)
        vloss = train_metrics.get("value_loss", 0.0)
        entropy = train_metrics.get("entropy", 0.0)

        # 2. Check stall
        problem = self._check_stall(eval_result, train_metrics)

        # 3. Status indicator
        if problem and problem in ("stall", "entropy_collapse", "loss_divergence", "win_rate_regression"):
            if problem == "stall":
                status = "!! stalled"
            elif problem == "entropy_collapse":
                status = "!! entropy collapse"
            elif problem == "loss_divergence":
                status = "!! loss diverging"
            else:
                status = "!! regressing"
        elif self.evals_since_improvement == 0:
            status = "^ improving"
        elif self.evals_since_improvement >= self.stall_patience - 1:
            status = "~ stalling"
        else:
            status = "- plateau"

        # 4. Print progress
        print(
            f"Step {step_label:>6s} | "
            f"vs Random: {wr_rand:.0%} | vs Heuristic: {wr_heur:.0%} | "
            f"ploss: {ploss:.4f} | vloss: {vloss:.4f} | entropy: {entropy:.3f} | "
            f"stage: {self.curriculum_stage} | {status}"
        )

        # 5. Log to CSV
        row = {
            "step": total_steps,
            "win_rate_vs_random": round(wr_rand, 4),
            "win_rate_vs_heuristic": round(wr_heur, 4),
            "policy_loss": round(ploss, 6),
            "value_loss": round(vloss, 6),
            "entropy": round(entropy, 6),
            "avg_steps_vs_random": round(eval_result["avg_steps_vs_random"], 1),
            "avg_steps_vs_heuristic": round(eval_result["avg_steps_vs_heuristic"], 1),
            "curriculum_stage": self.curriculum_stage,
            "lr": self.lr,
            "entropy_coef": self.entropy_coef,
            "rollout_steps": self.rollout_steps,
            "stall_counter": self.evals_since_improvement,
        }
        self._log_eval(row)

        # 6. Save checkpoint
        ckpt_path = str(self.checkpoint_dir / f"checkpoint_{total_steps}.pt")
        self._save_checkpoint(ckpt_path)

        # Track best (by heuristic win rate, with random as tiebreaker)
        is_best = (
            wr_heur > self.best_wr_heuristic
            or (wr_heur == self.best_wr_heuristic and wr_rand > self.best_wr_random)
        )
        if is_best or self.best_checkpoint_path is None:
            self.best_checkpoint_path = ckpt_path
            best_path = str(self.checkpoint_dir / "best.pt")
            self._save_checkpoint(best_path)

        # Update best trackers (check_stall already updates these, but
        # ensure they are set even on first eval)
        if wr_heur > self.best_wr_heuristic:
            self.best_wr_heuristic = wr_heur
        if wr_rand > self.best_wr_random:
            self.best_wr_random = wr_rand

        # 7. Curriculum advancement check
        adv_msg = self._check_curriculum_advance(eval_result, total_steps)
        if adv_msg:
            print(f"  CURRICULUM: {adv_msg}")

        # 8. Intervention if stalled
        if problem in ("stall", "entropy_collapse", "loss_divergence", "win_rate_regression"):
            desc = self._intervene(problem, total_steps)
            print(f"  INTERVENTION: {desc}")

    # ------------------------------------------------------------------
    # Cleanup and summary
    # ------------------------------------------------------------------

    def _close(self):
        """Release env resources."""
        if self.algorithm == "mappo" and self._trainer is not None:
            self._trainer.vec_env.close()
        elif self._vec_env is not None:
            self._vec_env.close()
            self._vec_env = None

    def _print_summary(self, total_steps: int):
        """Print final training summary."""
        elapsed = time.time() - self._train_start
        print()
        print("=" * 50)
        print("=== Training Summary ===")
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Total steps: {total_steps:,}")
        print(f"Wall time: {self._format_time(elapsed)}")

        if self.eval_log:
            best_rand = max(r["win_rate_vs_random"] for r in self.eval_log)
            best_rand_step = next(
                r["step"] for r in self.eval_log if r["win_rate_vs_random"] == best_rand
            )
            best_heur = max(r["win_rate_vs_heuristic"] for r in self.eval_log)
            best_heur_step = next(
                r["step"] for r in self.eval_log if r["win_rate_vs_heuristic"] == best_heur
            )
            print(
                f"Best win rate vs Random: {best_rand:.0%} "
                f"(at step {self._format_step(best_rand_step)})"
            )
            print(
                f"Best win rate vs Heuristic: {best_heur:.0%} "
                f"(at step {self._format_step(best_heur_step)})"
            )

        print(f"Interventions: {len(self.interventions)}", end="")
        if self.interventions:
            descs = [
                f"{i['action']} at {self._format_step(i['step'])}"
                for i in self.interventions
            ]
            print(f" ({', '.join(descs)})")
        else:
            print()

        if self.best_checkpoint_path:
            print(f"Best checkpoint: {self.best_checkpoint_path}")
        print("=" * 50)
