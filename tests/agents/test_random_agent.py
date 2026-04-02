"""Tests for random agent."""

import numpy as np

from knockout.agents.random_agent import RandomAgent


class TestRandomAgent:
    """Test random agent implementation."""

    def test_random_agent_creation(self):
        """Test that random agent can be created."""
        agent = RandomAgent("penguin_0")
        assert agent.agent_id == "penguin_0"

    def test_random_agent_with_seed(self):
        """Test that random agent with seed is deterministic."""
        agent1 = RandomAgent("penguin_0", seed=42)
        agent2 = RandomAgent("penguin_1", seed=42)

        obs = np.zeros(89, dtype=np.float32)

        action1 = agent1.get_action(obs)
        action2 = agent2.get_action(obs)

        # Same seed should produce same action
        assert np.allclose(action1, action2)

    def test_random_agent_action_shape(self):
        """Test that actions have correct shape."""
        agent = RandomAgent("penguin_0", seed=42)
        obs = np.zeros(89, dtype=np.float32)
        action = agent.get_action(obs)

        assert action.shape == (2,)
        assert action.dtype == np.float32

    def test_random_agent_action_range(self):
        """Test that actions are in valid range."""
        agent = RandomAgent("penguin_0", seed=42)
        obs = np.zeros(89, dtype=np.float32)

        # Sample multiple actions
        for _ in range(100):
            action = agent.get_action(obs)

            # Angle should be in [0, 360]
            assert 0.0 <= action[0] <= 360.0

            # Power should be in [0, 500]
            assert 0.0 <= action[1] <= 500.0

    def test_random_agent_different_seeds_produce_different_actions(self):
        """Test that different seeds produce different actions."""
        agent1 = RandomAgent("penguin_0", seed=42)
        agent2 = RandomAgent("penguin_0", seed=123)

        obs = np.zeros(89, dtype=np.float32)

        action1 = agent1.get_action(obs)
        action2 = agent2.get_action(obs)

        # Different seeds should produce different actions
        assert not np.allclose(action1, action2)

    def test_random_agent_reset(self):
        """Test that reset works."""
        agent = RandomAgent("penguin_0", seed=42)
        agent.reset()  # Should not raise

    def test_random_agent_observation_ignored(self):
        """Test that observation doesn't affect action (for random agent)."""
        agent = RandomAgent("penguin_0", seed=42)

        obs1 = np.zeros(89, dtype=np.float32)
        obs2 = np.ones(89, dtype=np.float32)

        action1 = agent.get_action(obs1)
        agent = RandomAgent("penguin_0", seed=42)  # Reset with same seed
        action2 = agent.get_action(obs2)

        # Random agent ignores observation, so with same seed should get same action
        assert np.allclose(action1, action2)
