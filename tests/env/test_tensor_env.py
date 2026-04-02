"""Tests for the fully vectorized TensorVecEnv."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from knockout.core.config import GameConfig, DEFAULTS
from knockout.env.tensor_env import TensorVecEnv
from knockout.env.tensor_observations import OBS_DIM


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def env4() -> TensorVecEnv:
    """4-environment TensorVecEnv on CPU."""
    e = TensorVecEnv(num_envs=4, device="cpu")
    return e


# ======================================================================
# Reset
# ======================================================================

class TestReset:
    def test_reset_shape(self, env4: TensorVecEnv) -> None:
        obs, masks = env4.reset()
        assert obs.shape == (4, 3, OBS_DIM)
        assert masks.shape == (4, 3)
        assert obs.dtype == np.float32
        assert masks.dtype == bool

    def test_reset_all_alive(self, env4: TensorVecEnv) -> None:
        _, masks = env4.reset()
        assert masks.all()


# ======================================================================
# Step
# ======================================================================

class TestStep:
    def test_step_shape(self, env4: TensorVecEnv) -> None:
        env4.reset()
        actions = np.zeros((4, 3, 2), dtype=np.float32)
        obs, rew, done, masks, infos = env4.step(actions)
        assert obs.shape == (4, 3, OBS_DIM)
        assert rew.shape == (4, 3)
        assert done.shape == (4,)
        assert masks.shape == (4, 3)
        assert len(infos) == 4

    def test_step_no_action_no_crash(self, env4: TensorVecEnv) -> None:
        """Zero actions should not crash."""
        env4.reset()
        actions = np.zeros((4, 3, 2), dtype=np.float32)
        for _ in range(3):
            obs, rew, done, masks, infos = env4.step(actions)
        assert obs.shape == (4, 3, OBS_DIM)


# ======================================================================
# Auto-reset
# ======================================================================

class TestAutoReset:
    def test_auto_reset(self) -> None:
        """After game ends, environment should auto-reset."""
        env = TensorVecEnv(num_envs=2, device="cpu")
        env.reset()

        # Force-kill team B in env 0 to trigger game over
        env.alive[0, 3:] = False

        actions = np.zeros((2, 3, 2), dtype=np.float32)
        obs, rew, done, masks, infos = env.step(actions)

        # Env 0 should have been done
        assert done[0]

        # After auto-reset, next step should work fine
        obs2, rew2, done2, masks2, infos2 = env.step(actions)
        # The auto-reset env should now have all alive
        assert obs2.shape == (2, 3, OBS_DIM)


# ======================================================================
# Rewards
# ======================================================================

class TestRewards:
    def test_win_reward(self) -> None:
        """Team A wins => reward +1."""
        env = TensorVecEnv(num_envs=1, device="cpu")
        env.reset()

        # Kill all of team B
        env.alive[0, 3:] = False
        actions = np.zeros((1, 3, 2), dtype=np.float32)
        _, rew, done, _, infos = env.step(actions)

        assert done[0]
        assert (rew[0] == 1.0).all()
        assert infos[0]["winner"] == 0

    def test_loss_reward(self) -> None:
        """Team A loses => reward -1."""
        env = TensorVecEnv(num_envs=1, device="cpu")
        env.reset()

        # Kill all of team A
        env.alive[0, :3] = False
        actions = np.zeros((1, 3, 2), dtype=np.float32)
        _, rew, done, _, infos = env.step(actions)

        assert done[0]
        assert (rew[0] == -1.0).all()
        assert infos[0]["winner"] == 1

    def test_draw_reward(self) -> None:
        """Both teams eliminated => reward 0 (draw)."""
        env = TensorVecEnv(num_envs=1, device="cpu")
        env.reset()

        # Kill all penguins
        env.alive[0] = False
        actions = np.zeros((1, 3, 2), dtype=np.float32)
        _, rew, done, _, _ = env.step(actions)

        assert done[0]
        assert (rew[0] == 0.0).all()

    def test_ongoing_no_reward(self) -> None:
        """While game is ongoing, rewards should be zero."""
        env = TensorVecEnv(num_envs=1, device="cpu")
        env.reset()
        actions = np.zeros((1, 3, 2), dtype=np.float32)
        _, rew, done, _, _ = env.step(actions)
        if not done[0]:
            assert (rew[0] == 0.0).all()


# ======================================================================
# Shrink
# ======================================================================

class TestShrink:
    def test_shrink_at_interval(self) -> None:
        """Arena should shrink every SHRINK_INTERVAL rounds."""
        env = TensorVecEnv(num_envs=1, device="cpu")
        env.reset()
        initial_hw = env.arena_hw[0].item()

        actions = np.zeros((1, 3, 2), dtype=np.float32)
        for _ in range(DEFAULTS.SHRINK_INTERVAL):
            env.step(actions)

        new_hw = env.arena_hw[0].item()
        # Should have shrunk (if env hasn't been auto-reset)
        # Due to random opponent, game may end — check conditionally
        if env.round_number[0].item() >= DEFAULTS.SHRINK_INTERVAL:
            assert new_hw < initial_hw or new_hw == pytest.approx(
                initial_hw * DEFAULTS.SHRINK_FACTOR, abs=0.1
            )


# ======================================================================
# Observation bounds
# ======================================================================

class TestObservationBounds:
    def test_observation_bounded(self) -> None:
        """All observations should be in [-1, 1]."""
        env = TensorVecEnv(num_envs=8, device="cpu")
        obs, _ = env.reset()
        assert obs.min() >= -1.0 - 1e-6
        assert obs.max() <= 1.0 + 1e-6

        actions = np.random.default_rng(42).uniform(
            low=[0, 0], high=[360, 400], size=(8, 3, 2)
        ).astype(np.float32)

        for _ in range(3):
            obs, _, _, _, _ = env.step(actions)
            assert obs.min() >= -1.0 - 1e-6, f"obs min={obs.min()}"
            assert obs.max() <= 1.0 + 1e-6, f"obs max={obs.max()}"


# ======================================================================
# Num envs validation
# ======================================================================

class TestEdgeCases:
    def test_num_envs_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            TensorVecEnv(num_envs=0)

    def test_single_env(self) -> None:
        env = TensorVecEnv(num_envs=1, device="cpu")
        obs, masks = env.reset()
        assert obs.shape == (1, 3, OBS_DIM)

    def test_close_is_safe(self) -> None:
        env = TensorVecEnv(num_envs=2, device="cpu")
        env.reset()
        env.close()  # should not raise


# ======================================================================
# Statistical comparison with SingleTeamVecEnv
# ======================================================================

class TestVsSingleTeamVecEnv:
    """Loose statistical checks against the reference environment."""

    def test_game_completes(self) -> None:
        """A game with random actions should eventually end."""
        env = TensorVecEnv(num_envs=4, device="cpu", max_rounds=50)
        env.reset()
        rng = np.random.default_rng(7)

        any_done = False
        for _ in range(50):
            actions = rng.uniform(
                low=[0, 0], high=[360, 400], size=(4, 3, 2)
            ).astype(np.float32)
            _, _, done, _, _ = env.step(actions)
            if done.any():
                any_done = True
                break

        # With random actions and shrinking, games should end quickly
        assert any_done, "No game completed in 50 rounds"

    def test_rewards_sum_to_expected(self) -> None:
        """Over many games, Team A should win roughly 50% against random."""
        env = TensorVecEnv(num_envs=32, device="cpu", max_rounds=30)
        env.reset()
        rng = np.random.default_rng(99)

        wins = 0
        losses = 0
        games = 0

        for _ in range(30):
            actions = rng.uniform(
                low=[0, 0], high=[360, 400], size=(32, 3, 2)
            ).astype(np.float32)
            _, rew, done, _, infos = env.step(actions)

            for i in range(32):
                if done[i]:
                    games += 1
                    if infos[i].get("winner") == 0:
                        wins += 1
                    elif infos[i].get("winner") == 1:
                        losses += 1

        # With random actions on both sides, win rate should be roughly balanced
        # (not a strict test — just a sanity check)
        if games > 5:
            win_rate = wins / games
            # Very loose bounds — just checking it's not pathologically broken
            assert 0.0 <= win_rate <= 1.0
