"""Tests for base agent interface."""

import numpy as np
import pytest

from knockout.agents.base import Agent


class ConcreteAgent(Agent):
    """Concrete implementation for testing."""

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Return fixed action for testing."""
        return np.array([45.0, 200.0], dtype=np.float32)


class TestAgent:
    """Test abstract Agent base class."""

    def test_agent_creation(self):
        """Test that concrete agent can be created."""
        agent = ConcreteAgent("penguin_0")
        assert agent.agent_id == "penguin_0"

    def test_agent_get_action(self):
        """Test that get_action works."""
        agent = ConcreteAgent("penguin_0")
        obs = np.zeros(89, dtype=np.float32)
        action = agent.get_action(obs)

        assert action.shape == (2,)
        assert action.dtype == np.float32
        assert action[0] == 45.0
        assert action[1] == 200.0

    def test_agent_reset(self):
        """Test that reset is callable."""
        agent = ConcreteAgent("penguin_0")
        agent.reset()  # Should not raise

    def test_cannot_instantiate_abstract_agent(self):
        """Test that Agent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Agent("penguin_0")  # type: ignore[abstract]
