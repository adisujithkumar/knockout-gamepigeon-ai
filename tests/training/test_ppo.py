"""Tests for PPO trainer."""

import math

import pytest

from knockout.core.config import DEFAULTS
from knockout.env.penguin_env import PenguinEnv
from knockout.training.ppo import PPOTrainer


class TestPPOTrainer:
    """Tests for PPOTrainer."""

    def test_trainer_creation(self):
        """Test PPOTrainer can be created with default params."""
        trainer = PPOTrainer()
        assert trainer.agent is not None
        assert trainer.opponent is not None
        assert trainer.gamma == 0.99
        assert trainer.clip_eps == 0.2
        assert trainer.n_epochs == 4
        assert trainer.batch_size == 64
        assert trainer.rollout_steps == 128

    def test_collect_rollout(self):
        """Test that collect_rollout fills the buffer with data."""
        trainer = PPOTrainer(rollout_steps=16, batch_size=32)
        env = PenguinEnv(config=DEFAULTS, seed=42)

        buffer = trainer.collect_rollout(env)
        env.close()

        # Buffer should have data (up to rollout_steps * 3 agents)
        assert buffer.pos > 0
        # Observations should not all be zero
        assert buffer.observations[: buffer.pos].any()

    def test_train_step(self):
        """Test that train_step returns metrics dict with expected keys."""
        trainer = PPOTrainer(rollout_steps=16, batch_size=16, n_epochs=2)
        env = PenguinEnv(config=DEFAULTS, seed=42)

        buffer = trainer.collect_rollout(env)
        metrics = trainer.train_step(buffer)
        env.close()

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert isinstance(metrics["policy_loss"], float)
        assert isinstance(metrics["value_loss"], float)
        assert isinstance(metrics["entropy"], float)

    def test_train_short(self):
        """Test short training run does not crash."""
        trainer = PPOTrainer(rollout_steps=16, batch_size=16, n_epochs=2)
        logs = trainer.train(total_timesteps=384, log_interval=100)

        assert len(logs) > 0
        assert "rollout" in logs[0]
        assert "policy_loss" in logs[0]

    def test_losses_are_finite(self):
        """Test that losses are finite (not NaN or Inf)."""
        trainer = PPOTrainer(rollout_steps=16, batch_size=16, n_epochs=2)
        env = PenguinEnv(config=DEFAULTS, seed=42)

        buffer = trainer.collect_rollout(env)
        metrics = trainer.train_step(buffer)
        env.close()

        assert math.isfinite(metrics["policy_loss"])
        assert math.isfinite(metrics["value_loss"])
        assert math.isfinite(metrics["entropy"])
