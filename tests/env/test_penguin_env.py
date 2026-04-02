"""Tests for PettingZoo environment."""

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from knockout.env.penguin_env import PenguinEnv
from knockout.core.config import GameConfig, DEFAULTS


class TestPenguinEnv:
    """Test PettingZoo environment implementation."""

    def test_environment_creation(self):
        """Test that environment can be created."""
        env = PenguinEnv(seed=42)
        assert env is not None
        assert env.max_steps == 1000
        assert env.physics_steps_per_action == 10

    def test_possible_agents(self):
        """Test that possible_agents is correctly defined."""
        env = PenguinEnv(seed=42)
        assert len(env.possible_agents) == 6
        assert env.possible_agents == list(DEFAULTS.POSSIBLE_AGENTS)
        assert "penguin_0" in env.possible_agents
        assert "penguin_5" in env.possible_agents

    def test_observation_space(self):
        """Test observation space definition."""
        env = PenguinEnv(seed=42)

        # Check observation spaces dict
        obs_spaces = env.observation_spaces
        assert len(obs_spaces) == 6

        # Check individual observation space
        for agent_id in env.possible_agents:
            obs_space = env.observation_space(agent_id)
            assert obs_space.shape == (89,)
            assert obs_space.dtype == np.float32
            assert obs_space.low[0] == -1.0
            assert obs_space.high[0] == 1.0

    def test_action_space(self):
        """Test action space definition."""
        env = PenguinEnv(seed=42)

        # Check action spaces dict
        action_spaces = env.action_spaces
        assert len(action_spaces) == 6

        # Check individual action space
        for agent_id in env.possible_agents:
            action_space = env.action_space(agent_id)
            assert action_space.shape == (2,)
            assert action_space.dtype == np.float32
            assert action_space.low[0] == 0.0  # angle
            assert action_space.high[0] == 360.0  # angle
            assert action_space.low[1] == 0.0  # power
            assert action_space.high[1] == DEFAULTS.MAX_LAUNCH_FORCE  # power

    def test_reset(self):
        """Test environment reset."""
        env = PenguinEnv(seed=42)
        observations, infos = env.reset()

        # Check observations
        assert len(observations) == 6
        for agent_id in env.possible_agents:
            assert agent_id in observations
            assert observations[agent_id].shape == (89,)
            assert observations[agent_id].dtype == np.float32

        # Check infos
        assert len(infos) == 6
        for agent_id in env.possible_agents:
            assert agent_id in infos
            assert isinstance(infos[agent_id], dict)

        # Check that agents are active
        assert len(env.agents) == 6

    def test_reset_determinism(self):
        """Test that reset with same seed produces same initial state."""
        env1 = PenguinEnv(seed=42)
        obs1, _ = env1.reset()

        env2 = PenguinEnv(seed=42)
        obs2, _ = env2.reset()

        # Observations should be identical
        for agent_id in env1.possible_agents:
            assert np.allclose(obs1[agent_id], obs2[agent_id])

    def test_step_with_actions(self):
        """Test stepping with actions."""
        env = PenguinEnv(seed=42, settle_mode=False)
        env.reset()

        # Create actions for all agents
        actions = {
            agent_id: np.array([float(i * 45), 300.0], dtype=np.float32)
            for i, agent_id in enumerate(env.agents)
        }

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check observations
        assert len(observations) == 6
        for agent_id in env.agents:
            assert observations[agent_id].shape == (89,)

        # Check rewards (should be 0 during game)
        assert len(rewards) == 6
        for agent_id in env.agents:
            assert rewards[agent_id] == 0.0  # Sparse rewards

        # Check terminations
        assert len(terminations) == 6
        assert all(not term for term in terminations.values())  # Game not over yet

        # Check truncations
        assert len(truncations) == 6
        assert all(not trunc for trunc in truncations.values())  # Not truncated yet

        # Check infos
        assert len(infos) == 6

    def test_step_without_actions_for_some_agents(self):
        """Test stepping when some agents don't provide actions."""
        env = PenguinEnv(seed=42)
        env.reset()

        # Only provide actions for half the agents
        actions = {
            "penguin_0": np.array([45.0, 300.0], dtype=np.float32),
            "penguin_3": np.array([135.0, 300.0], dtype=np.float32),
        }

        # Should not crash (missing agents get default no-op action)
        observations, rewards, terminations, truncations, infos = env.step(actions)

        assert len(observations) == 6
        assert len(rewards) == 6

    def test_simultaneous_actions(self):
        """Test that all actions are applied simultaneously."""
        # Use settle_mode=False so penguins are still moving after a fixed step count
        env = PenguinEnv(seed=42, settle_mode=False)
        env.reset()

        # All agents launch in different directions
        actions = {
            f"penguin_{i}": np.array([float(i * 60), 400.0], dtype=np.float32) for i in range(6)
        }

        # Step once
        env.step(actions)

        # Check that all penguins moved (simultaneity)
        for i in range(6):
            agent_id = f"penguin_{i}"
            penguin = env.physics_engine.penguins[agent_id]
            velocity = penguin.velocity

            # All penguins should have non-zero velocity
            speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
            assert speed > 1.0  # Should be moving

    def test_sparse_rewards_during_game(self):
        """Test that rewards are 0 during the game."""
        env = PenguinEnv(seed=42, settle_mode=False)
        env.reset()

        # Take several steps
        for _ in range(10):
            actions = {
                agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env.agents
            }
            _, rewards, terminations, _, _ = env.step(actions)

            if not any(terminations.values()):
                # Game not over, all rewards should be 0
                assert all(r == 0.0 for r in rewards.values())

    def test_game_termination(self):
        """Test that game terminates when one team is eliminated."""
        env = PenguinEnv(seed=42)
        env.reset()

        # Manually eliminate all of Team A
        for i in range(3):
            agent_id = f"penguin_{i}"
            env.physics_engine.penguins[agent_id].eliminate(env.physics_engine.space)

        # Step environment
        actions = {
            agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env.agents
        }
        _, rewards, terminations, _, infos = env.step(actions)

        # Game should be over
        assert all(terminations.values())

        # Team B should have won (team_id=1)
        for i, agent_id in enumerate(env.possible_agents):
            if i in DEFAULTS.TEAM_A_INDICES:
                assert rewards[agent_id] == -1.0  # Lost
            else:
                assert rewards[agent_id] == 1.0  # Won

        # Check info has winner
        for info in infos.values():
            assert "winner" in info
            assert info["winner"] == 1  # Team B

    def test_max_steps_truncation(self):
        """Test that environment truncates after max_steps."""
        env = PenguinEnv(seed=42, max_steps=10)
        env.reset()

        # Step exactly max_steps times
        for step in range(10):
            actions = {
                agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env.agents
            }
            _, _, terminations, truncations, _ = env.step(actions)

            if step < 9:
                # Not truncated yet
                assert all(not trunc for trunc in truncations.values())
            else:
                # Should be truncated
                assert all(truncations.values())

    def test_agents_list_updates(self):
        """Test that agents list is cleared after game end."""
        env = PenguinEnv(seed=42)
        env.reset()

        # Initially all agents active
        assert len(env.agents) == 6

        # Eliminate Team A
        for i in range(3):
            agent_id = f"penguin_{i}"
            env.physics_engine.penguins[agent_id].eliminate(env.physics_engine.space)

        # Step environment
        actions = {
            agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env.agents
        }
        env.step(actions)

        # Agents list should be cleared
        assert len(env.agents) == 0

    def test_cumulative_rewards(self):
        """Test that cumulative rewards are tracked correctly."""
        env = PenguinEnv(seed=42)
        env.reset()

        # Initially all cumulative rewards are 0
        for agent_id in env.possible_agents:
            assert env._cumulative_rewards[agent_id] == 0.0

        # Eliminate Team A and end game
        for i in range(3):
            agent_id = f"penguin_{i}"
            env.physics_engine.penguins[agent_id].eliminate(env.physics_engine.space)

        actions = {
            agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env.agents
        }
        env.step(actions)

        # Cumulative rewards should be updated
        for i, agent_id in enumerate(env.possible_agents):
            if i in DEFAULTS.TEAM_A_INDICES:
                assert env._cumulative_rewards[agent_id] == -1.0
            else:
                assert env._cumulative_rewards[agent_id] == 1.0

    def test_info_contains_step_count(self):
        """Test that info dicts contain step counts."""
        env = PenguinEnv(seed=42)
        env.reset()

        actions = {
            agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env.agents
        }
        _, _, _, _, infos = env.step(actions)

        for agent_id in env.agents:
            assert "step_count" in infos[agent_id]
            assert "current_step" in infos[agent_id]
            assert infos[agent_id]["current_step"] == 1

    def test_draw_game(self):
        """Test behavior when game ends in a draw."""
        env = PenguinEnv(seed=42)
        env.reset()

        # Eliminate all penguins (draw)
        for i in range(6):
            agent_id = f"penguin_{i}"
            env.physics_engine.penguins[agent_id].eliminate(env.physics_engine.space)

        actions = {
            agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env.agents
        }
        _, rewards, terminations, _, infos = env.step(actions)

        # Game should be over
        assert all(terminations.values())

        # All rewards should be 0 (draw)
        assert all(r == 0.0 for r in rewards.values())

        # Check winner is -1 (draw)
        for info in infos.values():
            assert info["winner"] == -1

    def test_reset_clears_state(self):
        """Test that reset properly clears previous game state."""
        env = PenguinEnv(seed=42, settle_mode=False)

        # Play a game
        env.reset()
        actions = {
            agent_id: np.array([float(i * 45), 300.0], dtype=np.float32)
            for i, agent_id in enumerate(env.agents)
        }
        for _ in range(10):
            env.step(actions)

        # Reset
        observations, infos = env.reset(seed=43)

        # Should be back to initial state
        assert len(env.agents) == 6
        assert env.current_step == 0
        assert all(env._cumulative_rewards[a] == 0.0 for a in env.possible_agents)

    def test_physics_steps_per_action(self):
        """Test that physics_steps_per_action controls simulation granularity."""
        # Test with different physics steps (settle_mode=False for fixed-step behavior)
        env1 = PenguinEnv(seed=42, physics_steps_per_action=1, settle_mode=False)
        env1.reset()

        env2 = PenguinEnv(seed=42, physics_steps_per_action=10, settle_mode=False)
        env2.reset()

        # Same action
        actions = {
            agent_id: np.array([45.0, 300.0], dtype=np.float32) for agent_id in env1.agents
        }

        env1.step(actions)
        env2.step(actions)

        # env2 should have more physics steps
        assert env2.physics_engine.step_count == 10
        assert env1.physics_engine.step_count == 1

    def test_action_clipping(self):
        """Test that out-of-bounds actions are clipped."""
        env = PenguinEnv(seed=42)
        env.reset()

        # Out of bounds actions
        actions = {
            "penguin_0": np.array([500.0, 1000.0], dtype=np.float32),  # Way too high
            "penguin_1": np.array([-100.0, -50.0], dtype=np.float32),  # Negative
        }

        # Should not crash (clipped internally)
        observations, rewards, terminations, truncations, infos = env.step(actions)

        assert len(observations) == 6
        assert len(rewards) == 6


class TestSettleMode:
    """Tests for settle-based stepping mode."""

    def test_settle_mode_penguins_at_rest_after_step(self):
        """In settle mode, after env.step(), all alive penguins should be at rest
        (unless the game ended due to elimination first)."""
        env = PenguinEnv(seed=42, settle_mode=True)
        env.reset()

        # Use low power so nobody gets knocked off the ice
        actions = {
            agent_id: np.array([float(i * 60), 50.0], dtype=np.float32)
            for i, agent_id in enumerate(env.agents)
        }
        env.step(actions)

        # Game should NOT be over with such low power
        assert not env.physics_engine.is_game_over(), "Game ended unexpectedly"

        # After step, every alive penguin should have speed below threshold
        threshold = env.config.SETTLE_SPEED_THRESHOLD
        for penguin in env.physics_engine.penguins.values():
            if penguin.alive:
                vx, vy = penguin.velocity
                speed = (vx**2 + vy**2) ** 0.5
                assert speed <= threshold, (
                    f"{penguin.agent_id} still moving at speed {speed:.3f} "
                    f"(threshold={threshold})"
                )

    def test_legacy_fixed_step_mode(self):
        """When settle_mode=False, the old fixed-step behavior is preserved."""
        env = PenguinEnv(seed=42, physics_steps_per_action=10, settle_mode=False)
        env.reset()

        actions = {
            agent_id: np.array([45.0, 300.0], dtype=np.float32)
            for agent_id in env.agents
        }
        env.step(actions)

        # Should have exactly 10 physics steps
        assert env.physics_engine.step_count == 10


class TestPettingZooAPICompliance:
    """Test PettingZoo API compliance."""

    def test_parallel_api_test(self):
        """Run PettingZoo's official parallel API test."""
        # This test verifies full PettingZoo Parallel API compliance
        # Use settle_mode=False to keep the test fast and predictable
        env = PenguinEnv(seed=42, max_steps=100, settle_mode=False)

        # Run official PettingZoo test
        # This will raise an error if the environment doesn't comply
        parallel_api_test(env, num_cycles=10)

    def test_parallel_api_test_with_different_seeds(self):
        """Test API compliance with different seeds."""
        for seed in [0, 42, 123, 999]:
            env = PenguinEnv(seed=seed, max_steps=100, settle_mode=False)
            parallel_api_test(env, num_cycles=5)


class TestRewardSystem:
    """Test reward system in detail."""

    def test_team_a_win_rewards(self):
        """Test rewards when Team A wins."""
        env = PenguinEnv(seed=42)
        env.reset()

        # Eliminate Team B
        for i in range(3, 6):
            agent_id = f"penguin_{i}"
            env.physics_engine.penguins[agent_id].eliminate(env.physics_engine.space)

        actions = {
            agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env.agents
        }
        _, rewards, _, _, _ = env.step(actions)

        # Team A should have +1, Team B should have -1
        for i in range(6):
            agent_id = f"penguin_{i}"
            if i < 3:
                assert rewards[agent_id] == 1.0
            else:
                assert rewards[agent_id] == -1.0

    def test_team_b_win_rewards(self):
        """Test rewards when Team B wins."""
        env = PenguinEnv(seed=42)
        env.reset()

        # Eliminate Team A
        for i in range(3):
            agent_id = f"penguin_{i}"
            env.physics_engine.penguins[agent_id].eliminate(env.physics_engine.space)

        actions = {
            agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in env.agents
        }
        _, rewards, _, _, _ = env.step(actions)

        # Team B should have +1, Team A should have -1
        for i in range(6):
            agent_id = f"penguin_{i}"
            if i < 3:
                assert rewards[agent_id] == -1.0
            else:
                assert rewards[agent_id] == 1.0

    def test_no_rewards_until_game_end(self):
        """Test that no rewards are given until game ends."""
        env = PenguinEnv(seed=42, max_steps=50, settle_mode=False)
        env.reset()

        # Play for many steps without ending game
        total_rewards = {agent_id: 0.0 for agent_id in env.possible_agents}

        for _ in range(20):
            actions = {
                agent_id: np.array([0.0, 50.0], dtype=np.float32) for agent_id in env.agents
            }
            _, rewards, terminations, _, _ = env.step(actions)

            # Accumulate rewards
            for agent_id in rewards:
                total_rewards[agent_id] += rewards[agent_id]

            # If game not over, all rewards should be 0
            if not any(terminations.values()):
                assert all(r == 0.0 for r in rewards.values())

        # If game never ended, all total rewards should still be 0
        if len(env.agents) > 0:
            assert all(r == 0.0 for r in total_rewards.values())
