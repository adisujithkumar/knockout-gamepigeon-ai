"""Self-play training with opponent pool, ELO tracking, and vectorized env."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from knockout.agents.base import Agent
from knockout.agents.rl_agent import RLAgent
from knockout.agents.random_agent import RandomAgent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.training.ppo import PPOTrainer
from knockout.training.elo_rating import ELOTracker
from knockout.training.evaluation import run_match
from knockout.training.vec_env import SingleTeamVecEnv


class OpponentPool:
    """Manages a pool of opponent agents with ELO-based eviction.

    The pool always contains a HeuristicAgent anchor that cannot be evicted.
    When the pool exceeds max_size, the lowest-ELO non-anchor opponent is
    dropped.
    """

    ANCHOR_NAME = "heuristic"

    def __init__(
        self,
        max_size: int = 10,
        elo_tracker: ELOTracker | None = None,
        seed: int | None = None,
    ):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")

        self.max_size = max_size
        self.elo_tracker = elo_tracker or ELOTracker()
        self._rng = np.random.default_rng(seed)

        # Mapping of name -> Agent
        self._agents: dict[str, Agent] = {}

        # Anchor: always present, never evicted
        anchor = HeuristicAgent(f"opponent_{self.ANCHOR_NAME}", seed=0)
        self._agents[self.ANCHOR_NAME] = anchor
        self.elo_tracker.register(self.ANCHOR_NAME)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Names of all opponents in the pool (deterministic order)."""
        return list(self._agents.keys())

    @property
    def size(self) -> int:
        return len(self._agents)

    def get(self, name: str) -> Agent:
        """Get an opponent by name."""
        return self._agents[name]

    def add(self, name: str, agent: Agent) -> str | None:
        """Add an opponent. Returns the name of the evicted agent, or None."""
        self.elo_tracker.register(name)
        self._agents[name] = agent

        evicted: str | None = None
        if len(self._agents) > self.max_size:
            evicted = self._evict_lowest_elo()
        return evicted

    def sample(self) -> tuple[str, Agent]:
        """Uniformly sample a random opponent from the pool."""
        names = self.names
        idx = int(self._rng.integers(len(names)))
        name = names[idx]
        return name, self._agents[name]

    def contains(self, name: str) -> bool:
        return name in self._agents

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_lowest_elo(self) -> str:
        """Remove the lowest-ELO non-anchor opponent. Returns evicted name."""
        candidates = [n for n in self._agents if n != self.ANCHOR_NAME]
        if not candidates:
            raise RuntimeError("Cannot evict: only anchor remains")

        worst_name = min(candidates, key=lambda n: self.elo_tracker.get_rating(n))
        del self._agents[worst_name]
        return worst_name


class SelfPlayTrainer:
    """Self-play training with opponent pool, vectorized envs, and ELO tracking.

    Training loop per generation:
      1. Sample an opponent from the pool.
      2. Train the learner for N rollouts against that opponent using
         SingleTeamVecEnv for parallelism.
      3. Evaluate the learner against every opponent in the pool.
      4. Update ELO ratings based on evaluation results.
      5. If the learner's ELO improved, save a checkpoint and add it to
         the opponent pool (with max-size eviction).
      6. Log metrics to a CSV file.
    """

    def __init__(
        self,
        config: GameConfig = DEFAULTS,
        checkpoint_dir: str = "checkpoints",
        pool_size: int = 10,
        num_envs: int = 8,
        seed: int | None = None,
        log_path: str | None = None,
        **ppo_kwargs,
    ):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.num_envs = num_envs
        self.seed = seed
        self.log_path = log_path
        self._rng = np.random.default_rng(seed)

        self.elo_tracker = ELOTracker()
        self.elo_tracker.register("learner")

        self.pool = OpponentPool(
            max_size=pool_size,
            elo_tracker=self.elo_tracker,
            seed=seed,
        )

        # PPO trainer -- we control the env ourselves, so use num_envs=1
        # in the trainer (we supply our own vec_env).
        self.trainer = PPOTrainer(config=config, num_envs=1, **ppo_kwargs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        n_generations: int = 50,
        rollouts_per_gen: int = 20,
        eval_games: int = 10,
    ) -> list[dict]:
        """Run the self-play training loop.

        Args:
            n_generations: Number of training generations.
            rollouts_per_gen: PPO rollouts collected per generation.
            eval_games: Games per opponent during evaluation.

        Returns:
            List of per-generation metric dicts.
        """
        logs: list[dict] = []
        prev_elo = self.elo_tracker.get_rating("learner")

        for gen in range(n_generations):
            # 1. Sample training opponent
            opp_name, opp_agent = self.pool.sample()

            # 2. Train against sampled opponent using vec env
            gen_metrics = self._train_generation(opp_agent, rollouts_per_gen)
            gen_metrics["generation"] = gen
            gen_metrics["opponent"] = opp_name

            # 3. Evaluate against all opponents in the pool
            win_rates = self._evaluate_all(eval_games)
            avg_win_rate = float(np.mean(list(win_rates.values()))) if win_rates else 0.0
            gen_metrics["win_rate"] = avg_win_rate
            gen_metrics["win_rates"] = win_rates

            # Find the best opponent (highest win rate for learner)
            if win_rates:
                best_opp = max(win_rates, key=lambda k: win_rates[k])
            else:
                best_opp = opp_name
            gen_metrics["best_opponent"] = best_opp

            # 4. Update ELO from evaluation results
            for opp_n, wr in win_rates.items():
                self.elo_tracker.update("learner", opp_n, wr)

            current_elo = self.elo_tracker.get_rating("learner")
            gen_metrics["elo"] = current_elo

            # 5. Checkpoint: always save on first gen (to populate pool),
            #    thereafter save only when ELO improves.
            should_save = (gen == 0) or (current_elo > prev_elo)
            gen_metrics["checkpoint_saved"] = should_save
            if should_save:
                self._save_checkpoint(gen)
                prev_elo = current_elo

            logs.append(gen_metrics)

            print(
                f"Gen {gen+1}/{n_generations}: vs {opp_name}, "
                f"avg_win_rate={avg_win_rate:.2f}, "
                f"ELO={current_elo:.0f}, "
                f"pool_size={self.pool.size}, "
                f"checkpoint={'saved' if should_save else 'skipped'}"
            )

        # Write CSV log
        if self.log_path:
            self._write_csv(logs)

        return logs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_generation(
        self, opponent: Agent, rollouts_per_gen: int
    ) -> dict[str, float]:
        """Train learner for one generation against a fixed opponent."""
        def _opp_factory() -> Agent:
            return opponent

        vec_env = SingleTeamVecEnv(
            num_envs=self.num_envs,
            opponent_factory=_opp_factory,
            config=self.config,
        )

        metrics: dict[str, float] = {}
        for _r in range(rollouts_per_gen):
            buffer = self.trainer.collect_rollout_vec(vec_env)
            metrics = self.trainer.train_step(buffer)

        vec_env.close()
        return metrics

    def _evaluate_all(self, n_games: int) -> dict[str, float]:
        """Evaluate learner against every opponent in the pool.

        Returns dict of opponent_name -> win_rate.
        """
        learner = self.trainer.agent
        win_rates: dict[str, float] = {}

        for opp_name in self.pool.names:
            opp_agent = self.pool.get(opp_name)
            wr = self._evaluate_match(learner, opp_agent, n_games)
            win_rates[opp_name] = wr

        return win_rates

    def _evaluate_match(
        self, agent_a: Agent, agent_b: Agent, n_games: int
    ) -> float:
        """Run n_games between agent_a (Team A) and agent_b (Team B).

        Returns win rate for agent_a.
        """
        team_a = {f"penguin_{k}": agent_a for k in range(3)}
        team_b = {f"penguin_{k}": agent_b for k in range(3, 6)}

        wins = 0
        for game in range(n_games):
            seed = int(self._rng.integers(0, 2**31))
            result = run_match(
                team_a, team_b,
                config=self.config,
                seed=seed,
                max_steps=200,
            )
            if result["winner"] == 0:
                wins += 1

        return wins / max(n_games, 1)

    def _save_checkpoint(self, generation: int) -> Path:
        """Save learner weights and add snapshot to opponent pool."""
        name = f"gen_{generation}"
        path = self.checkpoint_dir / f"{name}.pt"
        self.trainer.agent.save(path)

        # Create independent copy for the pool
        snapshot = RLAgent(f"opponent_{name}")
        snapshot.load(path)

        self.pool.add(name, snapshot)
        return path

    def _write_csv(self, logs: list[dict]) -> None:
        """Write training logs to CSV."""
        if not logs:
            return

        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten: exclude nested dicts (win_rates) for CSV
        csv_keys = [
            "generation", "opponent", "win_rate", "elo",
            "best_opponent", "checkpoint_saved",
            "policy_loss", "value_loss", "entropy",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(logs)
