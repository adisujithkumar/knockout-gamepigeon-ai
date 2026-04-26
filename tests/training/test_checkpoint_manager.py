"""Tests for checkpoint manager (save/load/resume training state)."""

import json
import time

import numpy as np
import pytest
import torch

from knockout.training.checkpoint_manager import (
    CheckpointManager,
    TrainingState,
    capture_rng_state,
    restore_rng_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(step: int = 1000, iteration: int = 5, **overrides) -> TrainingState:
    """Create a TrainingState with realistic dummy data."""
    from knockout.agents.rl_agent import ActorCritic

    net = ActorCritic(obs_dim=89, action_dim=2, hidden_sizes=(32, 16))
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    defaults = dict(
        step=step,
        iteration=iteration,
        agent_state_dict=net.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        pool_metadata=[
            {"path": "pool/pool_step_0000000.pt", "elo": 1000.0, "step": 0},
            {"path": "pool/pool_step_0500000.pt", "elo": 1050.0, "step": 500_000},
        ],
        elo_ratings={"learner": 1150.0, "pool_step_0": 1000.0},
        reward_shaper_state={
            "feature_names": ["ego.dist_to_edge", "enemy.speed"],
            "weights": [0.3, -0.2],
        },
        self_play_ratio=0.65,
        metrics_history=[
            {"step": 0, "elo": 1000.0, "policy_loss": 0.5},
            {"step": 500_000, "elo": 1050.0, "policy_loss": 0.3},
        ],
        rng_state=capture_rng_state(),
        config={"lr": 3e-4, "num_envs": 64, "rollout_steps": 128},
    )
    defaults.update(overrides)
    return TrainingState(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainingState:
    """Test TrainingState dataclass basics."""

    def test_default_construction(self):
        """TrainingState can be constructed with just step/iteration/dicts."""
        state = TrainingState(
            step=0,
            iteration=0,
            agent_state_dict={},
            optimizer_state_dict={},
        )
        assert state.step == 0
        assert state.pool_metadata == []
        assert state.elo_ratings == {}
        assert state.self_play_ratio == 0.0


class TestCheckpointManagerSaveLoad:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        """Saving then loading should preserve all fields."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        original = _make_state(step=1000, iteration=5)

        saved_path = manager.save(original)
        assert saved_path.exists()
        assert (saved_path / "training_state.pt").exists()
        assert (saved_path / "config.json").exists()
        assert (saved_path / "metrics.csv").exists()
        assert (saved_path / "checkpoint_info.json").exists()

        loaded = manager.load()

        assert loaded.step == original.step
        assert loaded.iteration == original.iteration
        assert loaded.self_play_ratio == pytest.approx(original.self_play_ratio)
        assert loaded.pool_metadata == original.pool_metadata
        assert loaded.elo_ratings == original.elo_ratings
        assert loaded.reward_shaper_state == original.reward_shaper_state
        assert loaded.config == original.config
        assert len(loaded.metrics_history) == len(original.metrics_history)

        # Check agent weights match
        for key in original.agent_state_dict:
            torch.testing.assert_close(
                loaded.agent_state_dict[key],
                original.agent_state_dict[key],
            )

    def test_save_with_label(self, tmp_path):
        """Save with a custom label creates correctly named directory."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        state = _make_state(step=5000)

        saved_path = manager.save(state, label="milestone_1")
        assert saved_path.name == "checkpoint_milestone_1"
        assert saved_path.exists()

    def test_save_default_name(self, tmp_path):
        """Save without label uses step-based name."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        state = _make_state(step=1234567)

        saved_path = manager.save(state)
        assert saved_path.name == "checkpoint_step_1234567"

    def test_latest_symlink(self, tmp_path):
        """Latest symlink points to most recently saved checkpoint."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        state1 = _make_state(step=1000)
        manager.save(state1)

        state2 = _make_state(step=2000)
        manager.save(state2)

        # Load from latest should get step=2000
        loaded = manager.load()
        assert loaded.step == 2000

    def test_load_specific_checkpoint(self, tmp_path):
        """Loading a specific checkpoint by name works."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        state1 = _make_state(step=1000)
        manager.save(state1)

        state2 = _make_state(step=2000)
        manager.save(state2)

        # Load the first one explicitly
        loaded = manager.load("checkpoint_step_0001000")
        assert loaded.step == 1000

    def test_load_best(self, tmp_path):
        """Loading with 'best' finds the highest-ELO checkpoint."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        state_low = _make_state(
            step=1000, elo_ratings={"learner": 1050.0}
        )
        manager.save(state_low)

        state_high = _make_state(
            step=2000, elo_ratings={"learner": 1200.0}
        )
        manager.save(state_high)

        state_mid = _make_state(
            step=3000, elo_ratings={"learner": 1100.0}
        )
        manager.save(state_mid)

        loaded = manager.load("best")
        assert loaded.step == 2000
        assert loaded.elo_ratings["learner"] == 1200.0

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading a nonexistent checkpoint raises FileNotFoundError."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent_checkpoint")

    def test_load_no_latest_raises(self, tmp_path):
        """Loading latest when no checkpoints exist raises FileNotFoundError."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        with pytest.raises(FileNotFoundError, match="No 'latest'"):
            manager.load()


class TestListCheckpoints:
    """Test listing available checkpoints."""

    def test_list_empty(self, tmp_path):
        """Empty checkpoint directory returns empty list."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        assert manager.list_checkpoints() == []

    def test_list_multiple(self, tmp_path):
        """Lists all checkpoints sorted by step."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        for step in [3000, 1000, 2000]:
            state = _make_state(step=step, iteration=step // 1000)
            manager.save(state)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
        assert checkpoints[0]["step"] == 1000
        assert checkpoints[1]["step"] == 2000
        assert checkpoints[2]["step"] == 3000

    def test_list_includes_metadata(self, tmp_path):
        """Each checkpoint entry has expected metadata fields."""
        manager = CheckpointManager(tmp_path / "checkpoints")
        state = _make_state(step=5000, iteration=10)
        manager.save(state)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 1
        ckpt = checkpoints[0]
        assert ckpt["step"] == 5000
        assert ckpt["iteration"] == 10
        assert "timestamp" in ckpt
        assert "path" in ckpt
        assert ckpt["elo"] is not None


class TestAutoSave:
    """Test auto_save interval logic."""

    def test_auto_save_respects_interval(self, tmp_path):
        """auto_save only saves when interval has been exceeded."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        state1 = _make_state(step=100)
        result = manager.auto_save(state1, interval_steps=500)
        # First call: _last_save_step is -1, so 100 - (-1) = 101 < 500
        # But wait, 100 - (-1) = 101 < 500... actually let's check:
        # _last_save_step starts at -1, so 100 - (-1) = 101 < 500
        # So this should NOT save yet. But it's close to 0 vs 500.
        # Actually the first save should happen because
        # step(100) - last_save(-1) = 101 < 500, so no save.
        assert result is None

        state2 = _make_state(step=600)
        result = manager.auto_save(state2, interval_steps=500)
        # 600 - (-1) = 601 >= 500 -> should save
        assert result is not None
        assert result.exists()

        state3 = _make_state(step=800)
        result = manager.auto_save(state3, interval_steps=500)
        # 800 - 600 = 200 < 500 -> no save
        assert result is None

        state4 = _make_state(step=1200)
        result = manager.auto_save(state4, interval_steps=500)
        # 1200 - 600 = 600 >= 500 -> save
        assert result is not None

    def test_auto_save_updates_last_state(self, tmp_path):
        """auto_save always updates _last_state for signal handler."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        state = _make_state(step=50)
        manager.auto_save(state, interval_steps=1000)
        assert manager._last_state is state


class TestCleanup:
    """Test checkpoint cleanup logic."""

    def test_cleanup_keeps_last_n(self, tmp_path):
        """Cleanup removes old checkpoints, keeping the last N."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        for step in range(0, 5000, 1000):
            state = _make_state(step=step, iteration=step // 1000)
            manager.save(state)

        # Should have 5 checkpoints: 0, 1000, 2000, 3000, 4000
        assert len(manager.list_checkpoints()) == 5

        removed = manager.cleanup(keep_last_n=2, keep_best=False)
        remaining = manager.list_checkpoints()

        assert len(remaining) == 2
        steps = [c["step"] for c in remaining]
        assert 3000 in steps
        assert 4000 in steps
        assert len(removed) == 3

    def test_cleanup_keeps_best(self, tmp_path):
        """Cleanup keeps the best ELO checkpoint even if old."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        # Best ELO is at step=1000
        for step, elo in [(0, 1000.0), (1000, 1300.0), (2000, 1100.0),
                          (3000, 1150.0), (4000, 1200.0)]:
            state = _make_state(
                step=step, iteration=step // 1000,
                elo_ratings={"learner": elo},
            )
            manager.save(state)

        manager.cleanup(keep_last_n=2, keep_best=True)
        remaining = manager.list_checkpoints()
        steps = [c["step"] for c in remaining]

        # Should keep: 3000, 4000 (last 2) + 1000 (best ELO)
        assert 1000 in steps, "Best ELO checkpoint should be kept"
        assert 3000 in steps
        assert 4000 in steps

    def test_cleanup_no_removal_needed(self, tmp_path):
        """Cleanup does nothing when under the limit."""
        manager = CheckpointManager(tmp_path / "checkpoints")

        state = _make_state(step=1000)
        manager.save(state)

        removed = manager.cleanup(keep_last_n=5)
        assert removed == []
        assert len(manager.list_checkpoints()) == 1


class TestRNGState:
    """Test RNG capture and restore."""

    def test_rng_roundtrip(self):
        """Captured RNG state produces same sequence after restore."""
        torch.manual_seed(42)
        np.random.seed(42)

        state = capture_rng_state()

        # Generate some random numbers
        expected_np = np.random.rand(5)
        expected_torch = torch.rand(5)

        # Restore and generate again
        restore_rng_state(state)
        actual_np = np.random.rand(5)
        actual_torch = torch.rand(5)

        np.testing.assert_array_equal(actual_np, expected_np)
        torch.testing.assert_close(actual_torch, expected_torch)
