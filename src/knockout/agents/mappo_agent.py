"""MAPPO (Multi-Agent PPO) agent with centralized critic.

Centralized Training, Decentralized Execution (CTDE):
- Each of 3 team agents has an independent policy network
- All agents share a single critic that sees all team observations concatenated
- At inference time, each agent acts on its own observation only
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from knockout.agents.base import Agent


class PolicyNetwork(nn.Module):
    """Independent policy network for a single agent.

    Architecture: obs(89) -> hidden(128) -> hidden(64) -> action_mean(2) + learned log_std
    """

    def __init__(
        self,
        obs_dim: int = 89,
        action_dim: int = 2,
        hidden_sizes: Sequence[int] = (128, 64),
    ):
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.features = nn.Sequential(*layers)

        self.action_mean = nn.Linear(input_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and log_std.

        Args:
            obs: (batch, obs_dim) observation tensor.

        Returns:
            action_mean: (batch, action_dim)
            action_log_std: (batch, action_dim)  (broadcast from parameter)
        """
        features = self.features(obs)
        mean = self.action_mean(features)
        log_std = self.action_log_std.expand_as(mean)
        return mean, log_std

    def get_action_and_log_prob(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) an action.

        Args:
            obs: (batch, obs_dim)
            action: If provided, compute log_prob for this action instead of sampling.

        Returns:
            action: (batch, action_dim)
            log_prob: (batch,)
            entropy: (batch,)
        """
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class CriticNetwork(nn.Module):
    """Centralized critic that sees all team observations concatenated.

    Architecture: concat_obs(267) -> hidden(256) -> hidden(128) -> value(1)
    """

    def __init__(
        self,
        input_dim: int = 267,  # 3 * 89
        hidden_sizes: Sequence[int] = (256, 128),
    ):
        super().__init__()

        layers: list[nn.Module] = []
        dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.ReLU())
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, concat_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value.

        Args:
            concat_obs: (batch, num_agents * obs_dim)

        Returns:
            value: (batch,)
        """
        return self.network(concat_obs).squeeze(-1)


class MAPPOAgent:
    """Multi-Agent PPO with shared critic.

    Each of 3 team agents has an independent policy network,
    but they share a centralized value function that sees all
    team observations concatenated.
    """

    def __init__(
        self,
        obs_dim: int = 89,
        action_dim: int = 2,
        num_agents: int = 3,
        policy_hidden: Sequence[int] = (128, 64),
        critic_hidden: Sequence[int] = (256, 128),
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = torch.device(device)

        # 3 independent policy networks (one per penguin)
        self.policies = [
            PolicyNetwork(obs_dim, action_dim, policy_hidden).to(self.device)
            for _ in range(num_agents)
        ]

        # 1 shared critic seeing all team obs: (3 * 89 = 267 dims)
        self.critic = CriticNetwork(
            input_dim=obs_dim * num_agents,
            hidden_sizes=critic_hidden,
        ).to(self.device)

    def get_actions(
        self,
        team_obs: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get actions from all policies.

        Args:
            team_obs: (batch, num_agents, obs_dim) or (num_agents, obs_dim)
            actions: Optional (batch, num_agents, action_dim) for log_prob evaluation.

        Returns:
            actions: (batch, num_agents, action_dim)
            log_probs: (batch, num_agents)
            entropies: (batch, num_agents)
        """
        # Ensure batch dimension
        if team_obs.ndim == 2:
            team_obs = team_obs.unsqueeze(0)
            if actions is not None and actions.ndim == 2:
                actions = actions.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        batch_size = team_obs.shape[0]

        all_actions = []
        all_log_probs = []
        all_entropies = []

        for i, policy in enumerate(self.policies):
            agent_obs = team_obs[:, i, :]  # (batch, obs_dim)
            agent_action = actions[:, i, :] if actions is not None else None

            act, lp, ent = policy.get_action_and_log_prob(agent_obs, agent_action)
            all_actions.append(act)
            all_log_probs.append(lp)
            all_entropies.append(ent)

        # Stack along agent dimension
        result_actions = torch.stack(all_actions, dim=1)   # (batch, num_agents, action_dim)
        result_log_probs = torch.stack(all_log_probs, dim=1)  # (batch, num_agents)
        result_entropies = torch.stack(all_entropies, dim=1)  # (batch, num_agents)

        if squeeze:
            result_actions = result_actions.squeeze(0)
            result_log_probs = result_log_probs.squeeze(0)
            result_entropies = result_entropies.squeeze(0)

        return result_actions, result_log_probs, result_entropies

    def get_value(self, team_obs: torch.Tensor) -> torch.Tensor:
        """Get centralized value estimate for the team.

        Args:
            team_obs: (batch, num_agents, obs_dim) or (num_agents, obs_dim)

        Returns:
            value: (batch,) or scalar
        """
        if team_obs.ndim == 2:
            # Single sample: (num_agents, obs_dim) -> (1, num_agents * obs_dim)
            concat = team_obs.reshape(1, -1)
            return self.critic(concat).squeeze(0)
        else:
            # Batch: (batch, num_agents, obs_dim) -> (batch, num_agents * obs_dim)
            concat = team_obs.reshape(team_obs.shape[0], -1)
            return self.critic(concat)

    def get_action(self, obs: np.ndarray, agent_index: int = 0) -> np.ndarray:
        """Single agent action for decentralized execution (evaluation).

        Args:
            obs: (obs_dim,) numpy observation for one agent.
            agent_index: Which policy to use (0, 1, or 2).

        Returns:
            Scaled action [angle_degrees, power_newtons] as numpy array.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            act, _, _ = self.policies[agent_index].get_action_and_log_prob(obs_t)
            act_np = act.squeeze(0).cpu().numpy()

        # Scale from raw network output to action space via sigmoid
        bounded = 1.0 / (1.0 + np.exp(-act_np))
        scaled = bounded * np.array([360.0, 500.0], dtype=np.float32)
        return scaled

    def parameters(self):
        """Yield all parameters (policies + critic) for a single optimizer."""
        for policy in self.policies:
            yield from policy.parameters()
        yield from self.critic.parameters()

    def policy_parameters(self):
        """Yield only policy parameters."""
        for policy in self.policies:
            yield from policy.parameters()

    def critic_parameters(self):
        """Yield only critic parameters."""
        yield from self.critic.parameters()

    def train(self) -> None:
        """Set all networks to training mode."""
        for policy in self.policies:
            policy.train()
        self.critic.train()

    def eval(self) -> None:
        """Set all networks to evaluation mode."""
        for policy in self.policies:
            policy.eval()
        self.critic.eval()

    def save(self, path: Path) -> None:
        """Save all network weights to a single checkpoint.

        Args:
            path: File path for the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "num_agents": self.num_agents,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
        for i, policy in enumerate(self.policies):
            checkpoint[f"policy_{i}"] = policy.state_dict()
        checkpoint["critic"] = self.critic.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        """Load all network weights from a checkpoint.

        Args:
            path: File path to the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        for i, policy in enumerate(self.policies):
            policy.load_state_dict(checkpoint[f"policy_{i}"])
        self.critic.load_state_dict(checkpoint["critic"])


class MAPPOEvalAgent(Agent):
    """Wraps a single policy from MAPPOAgent for use as a PettingZoo Agent.

    Used during evaluation to plug a trained MAPPO policy into the
    standard Agent interface expected by the environment and evaluation code.
    """

    def __init__(
        self,
        agent_id: str,
        mappo_agent: MAPPOAgent,
        agent_index: int = 0,
    ):
        super().__init__(agent_id)
        self.mappo_agent = mappo_agent
        self.agent_index = agent_index

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action from the assigned policy."""
        return self.mappo_agent.get_action(observation, self.agent_index)
