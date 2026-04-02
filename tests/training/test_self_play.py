"""Tests for self-play training with opponent pool."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.agents.rl_agent import RLAgent
from knockout.agents.random_agent import RandomAgent
from knockout.training.elo_rating import ELOTracker
from knockout.training.self_play import OpponentPool, SelfPlayTrainer


# ---------------------------------------------------------------
# OpponentPool
# ---------------------------------------------------------------

class TestOpponentPool:
    """Tests for the OpponentPool class."""

    def test_creation_has_anchor(self):
        """Pool starts with heuristic anchor."""
        pool = OpponentPool(max_size=5)
        assert pool.size == 1
        assert pool.contains("heuristic")

    def test_anchor_is_heuristic(self):
        """The anchor agent is a HeuristicAgent."""
        pool = OpponentPool(max_size=5)
        agent = pool.get("heuristic")
        assert isinstance(agent, HeuristicAgent)

    def test_add_opponent(self):
        """Adding an opponent increases pool size."""
        pool = OpponentPool(max_size=5)
        agent = RandomAgent("test")
        evicted = pool.add("test_agent", agent)

        assert pool.size == 2
        assert pool.contains("test_agent")
        assert evicted is None

    def test_max_size_eviction(self):
        """Pool evicts lowest-ELO non-anchor when over max_size."""
        elo = ELOTracker()
        pool = OpponentPool(max_size=3, elo_tracker=elo)

        # Add 3 opponents (pool will have 4 total with anchor -> over max)
        for i in range(3):
            name = f"agent_{i}"
            elo.register(name)
            # Give agent_0 the lowest ELO
            if i == 0:
                elo.ratings[name] = 800.0
            else:
                elo.ratings[name] = 1000.0 + i * 50
            pool.add(name, RandomAgent(name))

        # Pool should be at max_size (3)
        assert pool.size == 3
        # agent_0 (lowest ELO, non-anchor) should be evicted
        assert not pool.contains("agent_0")
        # Anchor and higher-ELO agents should remain
        assert pool.contains("heuristic")
        assert pool.contains("agent_1") or pool.contains("agent_2")

    def test_anchor_never_evicted(self):
        """Heuristic anchor is never evicted, even with lowest ELO."""
        elo = ELOTracker()
        pool = OpponentPool(max_size=2, elo_tracker=elo)

        # Give heuristic the lowest possible ELO
        elo.ratings["heuristic"] = 500.0

        # Add 2 opponents to trigger eviction
        for i in range(2):
            name = f"agent_{i}"
            elo.register(name)
            elo.ratings[name] = 1200.0 + i * 100
            pool.add(name, RandomAgent(name))

        # Heuristic must survive
        assert pool.contains("heuristic")
        # Pool is at max_size
        assert pool.size == 2

    def test_sample_returns_valid_opponent(self):
        """sample() returns a (name, agent) pair from the pool."""
        pool = OpponentPool(max_size=5, seed=42)
        pool.add("extra", RandomAgent("extra"))

        for _ in range(20):
            name, agent = pool.sample()
            assert name in pool.names
            assert agent is not None

    def test_pool_size_1_only_anchor(self):
        """With max_size=1, only the anchor can remain."""
        elo = ELOTracker()
        pool = OpponentPool(max_size=1, elo_tracker=elo)

        evicted = pool.add("temp", RandomAgent("temp"))
        # temp should be evicted immediately (anchor stays)
        assert pool.size == 1
        assert pool.contains("heuristic")
        assert not pool.contains("temp")
        assert evicted == "temp"

    def test_invalid_max_size(self):
        """max_size < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            OpponentPool(max_size=0)


# ---------------------------------------------------------------
# ELO Updates via SelfPlayTrainer
# ---------------------------------------------------------------

class TestEloUpdates:
    """Tests for ELO rating updates during self-play."""

    def test_elo_changes_after_matches(self):
        """ELO should change after evaluation matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                pool_size=5,
                num_envs=1,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            initial_elo = trainer.elo_tracker.get_rating("learner")
            assert initial_elo == 1000.0

            # Manually update ELO as if learner won
            trainer.elo_tracker.update("learner", "heuristic", 1.0)
            new_elo = trainer.elo_tracker.get_rating("learner")
            assert new_elo > initial_elo

    def test_elo_decreases_on_loss(self):
        """ELO should decrease when learner loses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                pool_size=5,
                num_envs=1,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            initial_elo = trainer.elo_tracker.get_rating("learner")
            trainer.elo_tracker.update("learner", "heuristic", 0.0)
            new_elo = trainer.elo_tracker.get_rating("learner")
            assert new_elo < initial_elo


# ---------------------------------------------------------------
# Heuristic Anchor
# ---------------------------------------------------------------

class TestHeuristicAnchor:
    """Tests that the heuristic agent is always in the pool."""

    def test_heuristic_in_initial_pool(self):
        """Heuristic anchor is present after construction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                pool_size=5,
                num_envs=1,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )
            assert trainer.pool.contains("heuristic")

    def test_heuristic_survives_training(self):
        """Heuristic anchor remains after running training generations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                pool_size=3,
                num_envs=1,
                seed=42,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            trainer.train(
                n_generations=2,
                rollouts_per_gen=1,
                eval_games=2,
            )

            assert trainer.pool.contains("heuristic")

    def test_heuristic_is_heuristic_agent(self):
        """The heuristic in the pool is actually a HeuristicAgent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                num_envs=1,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )
            agent = trainer.pool.get("heuristic")
            assert isinstance(agent, HeuristicAgent)


# ---------------------------------------------------------------
# Generation Loop
# ---------------------------------------------------------------

class TestGenerationLoop:
    """Tests for the full generation training loop."""

    def test_two_generations_run(self):
        """Running 2 generations should complete without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                pool_size=5,
                num_envs=2,
                seed=42,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            logs = trainer.train(
                n_generations=2,
                rollouts_per_gen=1,
                eval_games=2,
            )

            assert len(logs) == 2

            for log in logs:
                assert "generation" in log
                assert "opponent" in log
                assert "win_rate" in log
                assert "elo" in log
                assert "policy_loss" in log
                assert "value_loss" in log
                assert "entropy" in log
                assert "best_opponent" in log
                assert "checkpoint_saved" in log

    def test_generations_produce_monotonic_gen_numbers(self):
        """Generation numbers should be sequential."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                num_envs=1,
                seed=0,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            logs = trainer.train(
                n_generations=3,
                rollouts_per_gen=1,
                eval_games=2,
            )

            gens = [log["generation"] for log in logs]
            assert gens == [0, 1, 2]


# ---------------------------------------------------------------
# Checkpoint Save/Load
# ---------------------------------------------------------------

class TestCheckpointSaveLoad:
    """Tests for saving and loading opponent checkpoints."""

    def test_save_checkpoint_creates_file(self):
        """Saving a checkpoint should create a .pt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                num_envs=1,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            path = trainer._save_checkpoint(0)
            assert path.exists()
            assert path.suffix == ".pt"

    def test_checkpoint_added_to_pool(self):
        """Saving a checkpoint should add the agent to the pool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                pool_size=5,
                num_envs=1,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            trainer._save_checkpoint(0)
            assert trainer.pool.contains("gen_0")

    def test_reload_checkpoint_produces_same_weights(self):
        """A loaded checkpoint should have identical network weights."""
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                num_envs=1,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            # Save
            path = trainer._save_checkpoint(0)

            # Load into fresh agent
            fresh = RLAgent("test_reload")
            fresh.load(path)

            # Compare state dicts (weights are deterministic; actions are stochastic)
            pool_agent = trainer.pool.get("gen_0")
            assert isinstance(pool_agent, RLAgent)

            for key in pool_agent.network.state_dict():
                torch.testing.assert_close(
                    pool_agent.network.state_dict()[key],
                    fresh.network.state_dict()[key],
                )

    def test_multiple_checkpoints(self):
        """Multiple checkpoints should all exist on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                pool_size=10,
                num_envs=1,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            paths = []
            for gen in range(5):
                p = trainer._save_checkpoint(gen)
                paths.append(p)

            for p in paths:
                assert p.exists(), f"Checkpoint {p} should exist"


# ---------------------------------------------------------------
# CSV Logging
# ---------------------------------------------------------------

class TestCSVLogging:
    """Tests for CSV log file creation and contents."""

    def test_csv_created(self):
        """Training with log_path should create a CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "test_log.csv")

            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                num_envs=1,
                seed=42,
                log_path=log_path,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            trainer.train(
                n_generations=2,
                rollouts_per_gen=1,
                eval_games=2,
            )

            assert Path(log_path).exists()

    def test_csv_has_correct_columns(self):
        """CSV should contain the expected column headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "test_log.csv")

            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                num_envs=2,
                seed=42,
                log_path=log_path,
                rollout_steps=32,
                batch_size=32,
                n_epochs=1,
                lr=1e-4,
            )

            trainer.train(
                n_generations=2,
                rollouts_per_gen=2,
                eval_games=2,
            )

            with open(log_path) as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames

            expected = [
                "generation", "opponent", "win_rate", "elo",
                "best_opponent", "checkpoint_saved",
                "policy_loss", "value_loss", "entropy",
            ]
            for col in expected:
                assert col in columns, f"Missing column: {col}"

    def test_csv_has_correct_row_count(self):
        """CSV should have one row per generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "test_log.csv")

            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                num_envs=2,
                seed=42,
                log_path=log_path,
                rollout_steps=32,
                batch_size=32,
                n_epochs=1,
                lr=1e-4,
            )

            trainer.train(
                n_generations=3,
                rollouts_per_gen=2,
                eval_games=2,
            )

            with open(log_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3

    def test_no_csv_when_log_path_none(self):
        """No CSV should be written when log_path is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SelfPlayTrainer(
                checkpoint_dir=tmpdir,
                num_envs=1,
                seed=42,
                log_path=None,
                rollout_steps=16,
                batch_size=16,
                n_epochs=1,
            )

            trainer.train(
                n_generations=1,
                rollouts_per_gen=1,
                eval_games=2,
            )

            # No CSV files should exist in tmpdir
            csv_files = list(Path(tmpdir).glob("*.csv"))
            assert len(csv_files) == 0
