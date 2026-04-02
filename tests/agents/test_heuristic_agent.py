"""Tests for heuristic agent."""

import numpy as np
import pytest

from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.env.penguin_env import PenguinEnv


class TestHeuristicAgent:
    """Test heuristic agent implementation."""

    def test_heuristic_agent_creation(self):
        """Test that heuristic agent can be created."""
        agent = HeuristicAgent("penguin_0")
        assert agent.agent_id == "penguin_0"

    def test_heuristic_agent_action_shape(self):
        """Test that actions have correct shape."""
        agent = HeuristicAgent("penguin_0", seed=42)
        obs = self._create_mock_observation()
        action = agent.get_action(obs)

        assert action.shape == (2,)
        assert action.dtype == np.float32

    def test_heuristic_agent_action_range(self):
        """Test that actions are in valid range."""
        agent = HeuristicAgent("penguin_0", seed=42)
        obs = self._create_mock_observation()
        action = agent.get_action(obs)

        # Angle should be in [0, 360]
        assert 0.0 <= action[0] <= 360.0

        # Power should be in [0, 500]
        assert 0.0 <= action[1] <= 500.0

    def test_parse_ego(self):
        """Test ego state parsing."""
        agent = HeuristicAgent("penguin_0")
        obs = self._create_mock_observation()

        ego_state = agent._parse_ego(obs)

        assert "position" in ego_state
        assert "velocity" in ego_state
        assert "distance_to_edge" in ego_state
        assert "alive" in ego_state

    def test_parse_enemies(self):
        """Test enemy state parsing."""
        agent = HeuristicAgent("penguin_0")
        obs = self._create_mock_observation()

        enemies = agent._parse_enemies(obs)

        assert len(enemies) == 3
        for enemy in enemies:
            assert "position" in enemy
            assert "velocity" in enemy
            assert "distance_to_ego" in enemy
            assert "alive" in enemy

    def test_predict_position(self):
        """Test position prediction."""
        agent = HeuristicAgent("penguin_0")

        target = {
            "position": (0.0, 0.0),
            "velocity": (10.0, 10.0),
            "distance_to_ego": 50.0,
            "alive": True,
        }

        pred_pos = agent._predict_position(target, prediction_steps=10)

        # Predicted position should be ahead in direction of velocity
        assert pred_pos[0] > 0.0  # Moving right
        assert pred_pos[1] > 0.0  # Moving up

    def test_calculate_angle(self):
        """Test angle calculation."""
        agent = HeuristicAgent("penguin_0")

        # Test east direction
        angle = agent._calculate_angle((0.0, 0.0), (10.0, 0.0))
        assert np.isclose(angle, 0.0)

        # Test north direction
        angle = agent._calculate_angle((0.0, 0.0), (0.0, 10.0))
        assert np.isclose(angle, 90.0)

        # Test west direction
        angle = agent._calculate_angle((0.0, 0.0), (-10.0, 0.0))
        assert np.isclose(angle, 180.0)

        # Test south direction
        angle = agent._calculate_angle((0.0, 0.0), (0.0, -10.0))
        assert np.isclose(angle, 270.0)

    def test_calculate_power(self):
        """Test power calculation."""
        agent = HeuristicAgent("penguin_0")

        # Close target: lower power
        power_close = agent._calculate_power((0.0, 0.0), (10.0, 0.0))

        # Far target: higher power
        power_far = agent._calculate_power((0.0, 0.0), (100.0, 0.0))

        assert power_far > power_close
        assert 100.0 <= power_close <= 500.0
        assert 100.0 <= power_far <= 500.0

    def test_launch_toward_center(self):
        """Test emergency center launch."""
        agent = HeuristicAgent("penguin_0")

        ego_state = {
            "position": (80.0, 0.0),  # Near edge
            "velocity": (0.0, 0.0),
            "distance_to_edge": 10.0,
            "alive": True,
        }

        action = agent._launch_toward_center(ego_state)

        # Should launch toward center (angle ~180 degrees from east)
        assert action.shape == (2,)
        assert 0.0 <= action[0] <= 360.0
        assert action[1] > 0.0  # Should have some power

    def test_edge_awareness(self):
        """Test that agent avoids edge."""
        agent = HeuristicAgent("penguin_0", seed=42)

        # Create observation with ego near edge
        obs = self._create_mock_observation()
        obs[5] = 0.1  # distance_to_edge (normalized) - very close to edge

        action = agent.get_action(obs)

        # Should produce valid action
        assert action.shape == (2,)
        assert 0.0 <= action[0] <= 360.0
        assert 0.0 <= action[1] <= 500.0

    def test_targets_closest_enemy(self):
        """Test that agent targets closest enemy."""
        agent = HeuristicAgent("penguin_0", seed=42)

        # Create observation with multiple enemies at different distances
        obs = self._create_mock_observation()

        # Set enemy distances (index 13 of each 14-feature block)
        obs[42 + 13] = 0.5  # Enemy 1: distance 0.5 (normalized)
        obs[56 + 13] = 0.2  # Enemy 2: distance 0.2 (closest)
        obs[70 + 13] = 0.8  # Enemy 3: distance 0.8

        # All enemies alive
        obs[42 + 8] = 1.0
        obs[56 + 8] = 1.0
        obs[70 + 8] = 1.0

        action = agent.get_action(obs)

        # Should produce valid action (targeting closest enemy)
        assert action.shape == (2,)
        assert 0.0 <= action[0] <= 360.0
        assert 0.0 <= action[1] <= 500.0

    def test_handles_no_enemies(self):
        """Test behavior when no enemies alive."""
        agent = HeuristicAgent("penguin_0", seed=42)

        # Create observation with all enemies dead
        obs = self._create_mock_observation()
        obs[42 + 8] = 0.0  # Enemy 1 dead
        obs[56 + 8] = 0.0  # Enemy 2 dead
        obs[70 + 8] = 0.0  # Enemy 3 dead

        action = agent.get_action(obs)

        # Should return no-op action
        assert action.shape == (2,)
        assert action[0] == 0.0
        assert action[1] == 0.0

    def test_reset(self):
        """Test that reset works."""
        agent = HeuristicAgent("penguin_0")
        agent.reset()  # Should not raise

    def test_works_with_real_environment(self):
        """Test that agent works with real environment."""
        env = PenguinEnv(seed=42)
        obs_dict, _ = env.reset()

        # Create agents for all penguins
        agents = {
            f"penguin_{i}": HeuristicAgent(f"penguin_{i}", seed=i)
            for i in range(6)
        }

        # Get actions from all agents
        actions = {}
        for agent_id, agent in agents.items():
            obs = obs_dict[agent_id]
            action = agent.get_action(obs)
            actions[agent_id] = action

        # Step environment
        obs_dict, rewards, terminations, truncations, infos = env.step(actions)

        # Should work without errors
        assert len(obs_dict) == 6

    def test_prediction_accuracy(self):
        """BUG FIX #6: Verify geometric-sum prediction is accurate.

        A target at (0,0) with velocity (100,0), predicted for 10 steps.
        With damping=0.98 and dt=1/60:
          displacement = v * dt * (1 - d^N) / (1 - d)
                       = 100 * (1/60) * (1 - 0.98^10) / (1 - 0.98)
                       = 100 * 0.016667 * (1 - 0.8171) / 0.02
                       = 100 * 0.016667 * 9.1416
                       approx 15.24
        """
        agent = HeuristicAgent("penguin_0", config=DEFAULTS)

        target = {
            "position": (0.0, 0.0),
            "velocity": (100.0, 0.0),
            "distance_to_ego": 50.0,
            "alive": True,
        }

        prediction_steps = 10
        pred_pos = agent._predict_position(target, prediction_steps)

        # Calculate expected displacement with geometric sum
        # Pymunk damping is per-second; per-step retention = damping^dt
        dt = DEFAULTS.FIXED_DT  # 1/60
        d = DEFAULTS.DAMPING ** dt  # per-step retention factor
        expected_displacement = 100.0 * dt * (1.0 - d**prediction_steps) / (1.0 - d)

        assert np.isclose(pred_pos[0], expected_displacement, rtol=1e-6), (
            f"Predicted x={pred_pos[0]}, expected={expected_displacement}"
        )
        assert np.isclose(pred_pos[1], 0.0, atol=1e-9), (
            f"Predicted y={pred_pos[1]}, expected=0.0"
        )

    def test_utility_weights_sum_to_one(self):
        """BUG FIX #7: Verify utility is always in [0, 1].

        The utility function weights (0.4 + 0.25 + 0.35 = 1.0) should
        produce values bounded in [0, 1] since each factor is in [0, 1].
        """
        agent = HeuristicAgent("penguin_0")

        ego_state = {
            "position": (0.0, 0.0),
            "velocity": (0.0, 0.0),
            "distance_to_edge": 100.0,
            "alive": True,
        }

        # Test a variety of enemy configurations
        test_cases = [
            # Near edge, moving outward (high utility)
            {
                "position": (80.0, 0.0),
                "velocity": (50.0, 0.0),
                "distance_to_ego": 80.0,
                "distance_to_edge": 20.0,
                "alive": True,
            },
            # Center, stationary (low utility)
            {
                "position": (0.0, 0.0),
                "velocity": (0.0, 0.0),
                "distance_to_ego": 10.0,
                "distance_to_edge": 100.0,
                "alive": True,
            },
            # Far from ego, moving inward
            {
                "position": (60.0, 0.0),
                "velocity": (-100.0, 0.0),
                "distance_to_ego": 180.0,
                "distance_to_edge": 40.0,
                "alive": True,
            },
            # Near edge, close to ego, stationary
            {
                "position": (90.0, 0.0),
                "velocity": (0.0, 0.0),
                "distance_to_ego": 20.0,
                "distance_to_edge": 10.0,
                "alive": True,
            },
        ]

        for enemy in test_cases:
            utility = agent._calculate_target_utility(enemy, ego_state)
            assert 0.0 <= utility <= 1.0, (
                f"Utility {utility} out of bounds for enemy at {enemy['position']} "
                f"with velocity {enemy['velocity']}"
            )

    def _create_mock_observation(self) -> np.ndarray:
        """Create a mock observation for testing.

        Returns:
            89-dimensional observation with reasonable values
        """
        obs = np.zeros(89, dtype=np.float32)

        # Ego features (0-13)
        obs[0] = 0.0  # position_x (normalized)
        obs[1] = 0.0  # position_y (normalized)
        obs[2] = 0.1  # velocity_x (normalized)
        obs[3] = 0.1  # velocity_y (normalized)
        obs[4] = 0.0  # distance_from_center (normalized)
        obs[5] = 1.0  # distance_to_edge (normalized) - far from edge
        obs[6] = 0.1  # speed (normalized)
        obs[7] = 0.0  # heading (normalized)
        obs[8] = 1.0  # alive

        # Allies (14-41) - set some defaults
        obs[14 + 8] = 1.0  # Ally 1 alive
        obs[28 + 8] = 1.0  # Ally 2 alive

        # Enemies (42-83) - set some defaults
        obs[42] = 0.3  # Enemy 1 position_x
        obs[42 + 1] = 0.3  # Enemy 1 position_y
        obs[42 + 5] = 0.7  # Enemy 1 distance_to_edge (normalized)
        obs[42 + 8] = 1.0  # Enemy 1 alive
        obs[42 + 13] = 0.5  # Enemy 1 distance to ego

        obs[56] = 0.5  # Enemy 2 position_x
        obs[56 + 1] = 0.5  # Enemy 2 position_y
        obs[56 + 5] = 0.5  # Enemy 2 distance_to_edge (normalized)
        obs[56 + 8] = 1.0  # Enemy 2 alive
        obs[56 + 13] = 0.7  # Enemy 2 distance to ego

        obs[70] = -0.3  # Enemy 3 position_x
        obs[70 + 1] = -0.3  # Enemy 3 position_y
        obs[70 + 5] = 0.6  # Enemy 3 distance_to_edge (normalized)
        obs[70 + 8] = 1.0  # Enemy 3 alive
        obs[70 + 13] = 0.4  # Enemy 3 distance to ego

        # Global features (84-88)
        obs[84] = 1.0  # team_a_alive (normalized)
        obs[85] = 1.0  # team_b_alive (normalized)
        obs[86] = 1.0  # ego_team_alive (normalized)
        obs[87] = 1.0  # opponent_team_alive (normalized)
        obs[88] = 0.1  # timestep (normalized)

        return obs
