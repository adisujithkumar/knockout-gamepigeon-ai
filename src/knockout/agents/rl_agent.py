"""Neural network agent with actor-critic architecture for PPO."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from knockout.agents.base import Agent


class ActorCritic(nn.Module):
    """Actor-critic network with shared feature extractor."""

    def __init__(
        self,
        obs_dim: int = 89,
        action_dim: int = 2,
        hidden_sizes: Sequence[int] = (128, 64),
    ):
        super().__init__()

        # Shared feature extractor
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.features = nn.Sequential(*layers)

        # Actor head: outputs mean for each action dimension
        self.actor_mean = nn.Linear(input_dim, action_dim)
        # Log std as learnable parameter
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head: outputs state value
        self.critic = nn.Linear(input_dim, 1)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean, log_std, and value."""
        features = self.features(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, self.actor_log_std.expand_as(action_mean), value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value.

        If action is None, sample a new one. If action is provided, compute log_prob for it.
        """
        action_mean, action_log_std, value = self(obs)
        action_std = action_log_std.exp()

        dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value.squeeze(-1)


class RLAgent(Agent):
    """RL agent wrapping ActorCritic network."""

    def __init__(
        self,
        agent_id: str,
        obs_dim: int = 89,
        action_dim: int = 2,
        hidden_sizes: Sequence[int] = (128, 64),
        device: str = "cpu",
        max_force: float | None = None,
    ):
        super().__init__(agent_id)
        self.device = torch.device(device)
        self.network = ActorCritic(obs_dim, action_dim, hidden_sizes).to(self.device)
        # Action scaling: network outputs ~ N(0,1), sigmoid -> [0,1], then scale.
        # Must match the scaling used during training (config.MAX_LAUNCH_FORCE).
        from knockout.core.config import DEFAULTS

        _max_force = max_force if max_force is not None else DEFAULTS.MAX_LAUNCH_FORCE
        self.action_low = np.array([0.0, 0.0], dtype=np.float32)
        self.action_high = np.array([360.0, _max_force], dtype=np.float32)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action from policy (inference mode)."""
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            )
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            action, _, _, _ = self.network.get_action_and_value(obs_tensor)
            action = action.squeeze(0).cpu().numpy()

        # Scale from network output to action space
        # Use sigmoid to bound to [0, 1], then scale
        action_bounded = 1.0 / (1.0 + np.exp(-action))  # sigmoid
        scaled = self.action_low + action_bounded * (self.action_high - self.action_low)
        return scaled.astype(np.float32)

    def save(self, path: Path) -> None:
        """Save network weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load(self, path: Path) -> None:
        """Load network weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(state_dict)
