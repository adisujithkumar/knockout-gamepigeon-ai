"""Base agent interface for penguin knockout game."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Agent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Select action based on observation.

        Args:
            observation: 89-dimensional observation vector
        Returns:
            Action array [angle_degrees, power_newtons]
        """
        ...

    def reset(self) -> None:
        """Reset agent state (called at episode start)."""
        pass

    def save(self, path: Path) -> None:
        """Save agent state to file. Override in subclasses with learnable params."""
        pass

    def load(self, path: Path) -> None:
        """Load agent state from file. Override in subclasses with learnable params."""
        pass
