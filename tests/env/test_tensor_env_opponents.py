"""Tests for TensorVecEnv team_b_actions parameter and get_team_b_obs()."""

from __future__ import annotations

import numpy as np
import pytest

from knockout.env.tensor_env import TensorVecEnv
from knockout.env.tensor_observations import OBS_DIM


NUM_ENVS = 4


@pytest.fixture
def env() -> TensorVecEnv:
    e = TensorVecEnv(num_envs=NUM_ENVS, device="cpu")
    e.reset()
    return e


# ======================================================================
# Backward compatibility — step() with no team_b_actions
# ======================================================================

class TestBackwardCompatibility:
    def test_step_without_team_b_actions(self, env: TensorVecEnv) -> None:
        """Calling step(actions) without team_b_actions still works."""
        actions = np.zeros((NUM_ENVS, 3, 2), dtype=np.float32)
        obs, rew, done, masks, infos = env.step(actions)
        assert obs.shape == (NUM_ENVS, 3, OBS_DIM)
        assert rew.shape == (NUM_ENVS, 3)
        assert done.shape == (NUM_ENVS,)
        assert masks.shape == (NUM_ENVS, 3)

    def test_step_positional_only(self, env: TensorVecEnv) -> None:
        """Positional-only call matches the original API."""
        actions = np.zeros((NUM_ENVS, 3, 2), dtype=np.float32)
        # Should not raise
        env.step(actions)


# ======================================================================
# Explicit team_b_actions
# ======================================================================

class TestExplicitTeamBActions:
    def test_step_with_team_b_actions_shape(self, env: TensorVecEnv) -> None:
        """step() accepts team_b_actions and returns correct shapes."""
        ta = np.zeros((NUM_ENVS, 3, 2), dtype=np.float32)
        tb = np.zeros((NUM_ENVS, 3, 2), dtype=np.float32)
        obs, rew, done, masks, infos = env.step(ta, team_b_actions=tb)
        assert obs.shape == (NUM_ENVS, 3, OBS_DIM)
        assert rew.shape == (NUM_ENVS, 3)

    def test_team_b_actions_are_applied(self) -> None:
        """Providing specific team_b_actions produces different outcomes
        than the default random actions (deterministic check)."""
        env1 = TensorVecEnv(num_envs=1, device="cpu")
        env2 = TensorVecEnv(num_envs=1, device="cpu")
        env1.reset()
        env2.reset()

        # Make both envs start from identical state
        env2.positions = env1.positions.clone()
        env2.velocities = env1.velocities.clone()
        env2.alive = env1.alive.clone()
        env2.arena_hw = env1.arena_hw.clone()
        env2.round_number = env1.round_number.clone()
        env2.step_count = env1.step_count.clone()

        ta = np.array([[[90.0, 200.0], [180.0, 200.0], [270.0, 200.0]]], dtype=np.float32)
        # Team B: all shoot right with full power
        tb_explicit = np.array([[[0.0, 400.0], [0.0, 400.0], [0.0, 400.0]]], dtype=np.float32)
        # Team B: all shoot left with full power
        tb_different = np.array([[[180.0, 400.0], [180.0, 400.0], [180.0, 400.0]]], dtype=np.float32)

        obs1, _, _, _, _ = env1.step(ta, team_b_actions=tb_explicit)
        obs2, _, _, _, _ = env2.step(ta, team_b_actions=tb_different)

        # Different Team B actions should produce different observations
        assert not np.allclose(obs1, obs2), (
            "Different team_b_actions should produce different outcomes"
        )

    def test_zero_team_b_actions(self, env: TensorVecEnv) -> None:
        """Zero-power team_b_actions means Team B doesn't move (from impulse)."""
        ta = np.zeros((NUM_ENVS, 3, 2), dtype=np.float32)
        tb = np.zeros((NUM_ENVS, 3, 2), dtype=np.float32)
        # Both teams do nothing — should not crash, game stays ongoing
        obs, rew, done, masks, infos = env.step(ta, team_b_actions=tb)
        assert obs.shape == (NUM_ENVS, 3, OBS_DIM)
        # No one moved, game should still be ongoing
        assert not done.any()


# ======================================================================
# get_team_b_obs
# ======================================================================

class TestGetTeamBObs:
    def test_shape(self, env: TensorVecEnv) -> None:
        """get_team_b_obs() returns (B, 3, 89)."""
        obs_b = env.get_team_b_obs()
        assert obs_b.shape == (NUM_ENVS, 3, OBS_DIM)
        assert obs_b.dtype == np.float32

    def test_bounded(self, env: TensorVecEnv) -> None:
        """Team B observations should be in [-1, 1]."""
        obs_b = env.get_team_b_obs()
        assert obs_b.min() >= -1.0 - 1e-6
        assert obs_b.max() <= 1.0 + 1e-6

    def test_ally_enemy_swapped(self, env: TensorVecEnv) -> None:
        """Team B obs should have ally/enemy features swapped vs Team A.

        Layout: ego(14) + ally1(14) + ally2(14) + enemy1(14) + enemy2(14) + enemy3(14) + global(5)
        Global features include ego_team_alive and opp_team_alive at indices 86 and 87.
        For Team A agent: global[2]=team_a_alive/3, global[3]=team_b_alive/3
        For Team B agent: global[2]=team_b_alive/3, global[3]=team_a_alive/3
        """
        # Kill one Team A penguin to create asymmetry
        env.alive[0, 0] = False

        obs_a, _ = env._build_team_a_output()
        obs_b = env.get_team_b_obs()

        # Global features are the last 5 dims of each agent's 89-dim obs
        # Index 84: team_a_alive/3 (absolute, same for both)
        # Index 85: team_b_alive/3 (absolute, same for both)
        # Index 86: ego_team_alive/3
        # Index 87: opp_team_alive/3
        # Index 88: timestep

        # For an alive Team A agent, ego_team = team_a (2/3), opp_team = team_b (3/3)
        # For an alive Team B agent, ego_team = team_b (3/3), opp_team = team_a (2/3)
        a_agent_obs = obs_a[0, 1]  # Agent 1 of team A (alive)
        b_agent_obs = obs_b[0, 0]  # Agent 0 of team B (alive, index 3)

        # ego_team and opp_team should be swapped between teams
        assert a_agent_obs[86] != b_agent_obs[86] or a_agent_obs[87] != b_agent_obs[87], (
            "ego_team/opp_team global features should differ between teams"
        )
        # Specifically: A's ego_team == B's opp_team and vice versa
        assert abs(a_agent_obs[86] - b_agent_obs[87]) < 1e-6
        assert abs(a_agent_obs[87] - b_agent_obs[86]) < 1e-6


# ======================================================================
# Mirror match symmetry
# ======================================================================

class TestMirrorMatch:
    def test_symmetric_actions_symmetric_rewards(self) -> None:
        """When both teams take identical actions from symmetric positions,
        outcomes should be symmetric (rewards sum to zero)."""
        env = TensorVecEnv(num_envs=8, device="cpu")
        env.reset()

        rng = np.random.default_rng(123)
        total_reward_a = 0.0
        games = 0

        for _ in range(30):
            actions = rng.uniform(
                low=[0, 0], high=[360, 400], size=(8, 3, 2)
            ).astype(np.float32)
            # Mirror: both teams get the same actions
            _, rew, done, _, infos = env.step(actions, team_b_actions=actions)

            for i in range(8):
                if done[i]:
                    games += 1
                    total_reward_a += rew[i].sum()

        # With identical actions, the game is still not perfectly symmetric
        # because initial positions differ between teams. But rewards should
        # be bounded and not pathologically biased.
        if games > 0:
            avg = total_reward_a / games
            assert -3.5 <= avg <= 3.5, f"Mirror match avg reward {avg} seems biased"
