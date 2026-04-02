"""Random agent for baseline comparison."""

import numpy as np

from knockout.agents.base import Agent


class RandomAgent(Agent):
    """Agent that selects random actions."""

    def __init__(self, agent_id: str, seed: int | None = None):
        super().__init__(agent_id)
        self.rng = np.random.default_rng(seed)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        angle = self.rng.uniform(0.0, 360.0)
        power = self.rng.uniform(0.0, 500.0)
        return np.array([angle, power], dtype=np.float32)

    def reset(self) -> None:
        pass
