"""Tests for the monitored training harness."""

import csv
import json
from pathlib import Path

import pytest

from knockout.training.monitored_trainer import (
    MonitoredTrainer,
    STAGE_RANDOM,
    STAGE_MIXED,
    STAGE_HEURISTIC,
)


# Use tiny values for speed --------------------------------------------------

FAST_KWARGS = dict(
    algorithm="ppo",
    device="cpu",
    num_envs=4,
    backend="pymunk",
    eval_interval=500,
    eval_games=3,
    stall_patience=3,
    rollout_steps=16,
    batch_size=16,
    n_epochs=2,
    entropy_coef=0.01,
    lr=3e-4,
)


class TestEvaluationRuns:
    """Test that evaluation runs at the correct intervals."""

    def test_evaluation_runs(self, tmp_path):
        """Trainer evaluates at correct intervals (roughly every eval_interval steps)."""
        ckpt_dir = tmp_path / "ckpts"
        log_path = tmp_path / "log.csv"

        trainer = MonitoredTrainer(
            **FAST_KWARGS,
            checkpoint_dir=str(ckpt_dir),
            log_path=str(log_path),
        )

        # Train for enough steps to trigger at least 2 evaluations
        # steps_per_rollout = rollout_steps * num_envs * 3 = 16 * 4 * 3 = 192
        # eval_interval = 500 -> first eval after ceil(500/192)=3 rollouts = 576 steps
        # 2 evals need ~1152 steps
        trainer.train(total_timesteps=1500)

        assert len(trainer.eval_log) >= 2, (
            f"Expected >=2 evaluations, got {len(trainer.eval_log)}"
        )

        # Each eval log entry should have win rate fields
        for row in trainer.eval_log:
            assert "win_rate_vs_random" in row
            assert "win_rate_vs_heuristic" in row
            assert "step" in row


class TestStallDetection:
    """Test that stall is detected after patience evals with no improvement."""

    def test_stall_detection(self, tmp_path):
        """Stall detected when no improvement for stall_patience evaluations."""
        trainer = MonitoredTrainer(
            **{**FAST_KWARGS, "stall_patience": 2},
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_path=str(tmp_path / "log.csv"),
        )
        trainer._trainer = trainer._build_trainer()

        # Simulate: best = 0.3, current = 0.25 (slight drop, within regression
        # tolerance of 0.15, so should count as plateau not regression)
        trainer.best_wr_heuristic = 0.3
        trainer.best_wr_random = 0.8

        eval_result = {
            "win_rate_vs_random": 0.75,
            "win_rate_vs_heuristic": 0.25,
        }
        train_metrics = {"policy_loss": 0.01, "value_loss": 0.1, "entropy": 1.0}

        # First check: no stall yet (counter goes to 1)
        problem = trainer._check_stall(eval_result, train_metrics)
        assert problem is None
        assert trainer.evals_since_improvement == 1

        # Second check: stall (counter reaches patience=2)
        problem = trainer._check_stall(eval_result, train_metrics)
        assert problem == "stall"
        assert trainer.evals_since_improvement >= 2


class TestInterventionEntropy:
    """Test that entropy coefficient increases on first stall intervention."""

    def test_intervention_entropy(self, tmp_path):
        """First intervention increases entropy coefficient."""
        trainer = MonitoredTrainer(
            **FAST_KWARGS,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_path=str(tmp_path / "log.csv"),
        )
        trainer._trainer = trainer._build_trainer()

        old_entropy = trainer.entropy_coef
        desc = trainer._intervene("stall", step=100000)

        assert "entropy" in desc.lower()
        assert trainer.entropy_coef > old_entropy
        assert trainer.evals_since_improvement == 0  # reset after intervention


class TestInterventionLR:
    """Test that learning rate decreases on second stall intervention."""

    def test_intervention_lr(self, tmp_path):
        """Second intervention reduces learning rate."""
        trainer = MonitoredTrainer(
            **FAST_KWARGS,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_path=str(tmp_path / "log.csv"),
        )
        trainer._trainer = trainer._build_trainer()

        old_lr = trainer.lr

        # First intervention (entropy)
        trainer._intervene("stall", step=100000)
        # Second intervention (lr)
        desc = trainer._intervene("stall", step=200000)

        assert "lr" in desc.lower() or "reduced" in desc.lower()
        assert trainer.lr < old_lr


class TestCurriculumAdvancement:
    """Test that curriculum auto-advances from random to mixed to heuristic."""

    def test_curriculum_advancement(self, tmp_path):
        """Curriculum advances when win rate thresholds are met."""
        trainer = MonitoredTrainer(
            **FAST_KWARGS,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_path=str(tmp_path / "log.csv"),
        )
        trainer._trainer = trainer._build_trainer()
        if trainer.algorithm == "ppo" and trainer.num_envs > 1:
            trainer._vec_env = trainer._get_vec_env(trainer._trainer)

        # Stage 1: random
        assert trainer.curriculum_stage == STAGE_RANDOM

        # High random win rate -> should advance to mixed
        eval_result = {"win_rate_vs_random": 0.85, "win_rate_vs_heuristic": 0.1}
        msg = trainer._check_curriculum_advance(eval_result, step=100000)
        assert msg is not None
        assert trainer.curriculum_stage == STAGE_MIXED

        # High heuristic win rate -> should advance to heuristic
        eval_result2 = {"win_rate_vs_random": 0.9, "win_rate_vs_heuristic": 0.45}
        msg2 = trainer._check_curriculum_advance(eval_result2, step=200000)
        assert msg2 is not None
        assert trainer.curriculum_stage == STAGE_HEURISTIC

        # Cleanup
        if trainer._vec_env is not None:
            trainer._vec_env.close()


class TestCheckpointSaved:
    """Test that checkpoints exist at evaluation points."""

    def test_checkpoint_saved(self, tmp_path):
        """Checkpoints are saved at each evaluation."""
        ckpt_dir = tmp_path / "ckpts"
        log_path = tmp_path / "log.csv"

        trainer = MonitoredTrainer(
            **FAST_KWARGS,
            checkpoint_dir=str(ckpt_dir),
            log_path=str(log_path),
        )
        trainer.train(total_timesteps=1500)

        # Should have at least 2 checkpoint files
        pt_files = list(ckpt_dir.glob("checkpoint_*.pt"))
        assert len(pt_files) >= 2, (
            f"Expected >=2 checkpoint files, got {len(pt_files)}: {pt_files}"
        )

        # Should have a best.pt
        assert (ckpt_dir / "best.pt").exists()

        # Should have training_config.json
        assert (ckpt_dir / "training_config.json").exists()


class TestCSVLogging:
    """Test that CSV log has correct columns and rows."""

    def test_csv_logging(self, tmp_path):
        """CSV has correct columns and at least as many rows as evaluations."""
        ckpt_dir = tmp_path / "ckpts"
        log_path = tmp_path / "log.csv"

        trainer = MonitoredTrainer(
            **FAST_KWARGS,
            checkpoint_dir=str(ckpt_dir),
            log_path=str(log_path),
        )
        trainer.train(total_timesteps=1500)

        assert log_path.exists()

        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) >= 2
        # Check required columns
        required = [
            "step", "win_rate_vs_random", "win_rate_vs_heuristic",
            "policy_loss", "value_loss", "entropy",
            "curriculum_stage", "lr", "entropy_coef",
        ]
        for col in required:
            assert col in rows[0], f"Missing column: {col}"


class TestBestCheckpointTracked:
    """Test that the best checkpoint is identified correctly."""

    def test_best_checkpoint_tracked(self, tmp_path):
        """Best checkpoint path is set after evaluations."""
        ckpt_dir = tmp_path / "ckpts"
        log_path = tmp_path / "log.csv"

        trainer = MonitoredTrainer(
            **FAST_KWARGS,
            checkpoint_dir=str(ckpt_dir),
            log_path=str(log_path),
        )
        trainer.train(total_timesteps=1500)

        assert trainer.best_checkpoint_path is not None
        assert Path(trainer.best_checkpoint_path).exists()


class TestShortTraining:
    """Test that a short training run does not crash."""

    def test_short_training(self, tmp_path):
        """Run 3+ eval intervals, verify no crash and reasonable output."""
        ckpt_dir = tmp_path / "ckpts"
        log_path = tmp_path / "log.csv"

        trainer = MonitoredTrainer(
            **FAST_KWARGS,
            checkpoint_dir=str(ckpt_dir),
            log_path=str(log_path),
        )
        result = trainer.train(total_timesteps=2000)

        assert isinstance(result, list)
        assert len(result) >= 3

        # All win rates should be valid probabilities
        for row in result:
            assert 0.0 <= row["win_rate_vs_random"] <= 1.0
            assert 0.0 <= row["win_rate_vs_heuristic"] <= 1.0


class TestConfigSaved:
    """Test that training config JSON is saved."""

    def test_config_saved(self, tmp_path):
        """training_config.json is written with correct fields."""
        ckpt_dir = tmp_path / "ckpts"
        log_path = tmp_path / "log.csv"

        trainer = MonitoredTrainer(
            **FAST_KWARGS,
            checkpoint_dir=str(ckpt_dir),
            log_path=str(log_path),
        )
        trainer.train(total_timesteps=700)

        cfg_path = ckpt_dir / "training_config.json"
        assert cfg_path.exists()

        with open(cfg_path) as f:
            cfg = json.load(f)

        assert cfg["algorithm"] == "ppo"
        assert cfg["num_envs"] == 4
        assert cfg["eval_interval"] == 500


class TestMAPPOAlgorithm:
    """Test that MAPPO algorithm works with the monitored trainer."""

    def test_mappo_short_training(self, tmp_path):
        """MAPPO runs without crashing in monitored mode."""
        ckpt_dir = tmp_path / "ckpts"
        log_path = tmp_path / "log.csv"

        trainer = MonitoredTrainer(
            **{**FAST_KWARGS, "algorithm": "mappo"},
            checkpoint_dir=str(ckpt_dir),
            log_path=str(log_path),
        )
        result = trainer.train(total_timesteps=1500)

        assert isinstance(result, list)
        assert len(result) >= 1
