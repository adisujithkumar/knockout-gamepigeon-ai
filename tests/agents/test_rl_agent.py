"""Tests for RL agent with actor-critic network."""

import numpy as np
import pytest
import torch

from knockout.agents.rl_agent import ActorCritic, RLAgent


class TestRLAgent:
    """Test RL agent and ActorCritic network."""

    def test_creation(self):
        """Test that RL agent can be created with defaults."""
        agent = RLAgent("penguin_0")
        assert agent.agent_id == "penguin_0"
        assert agent.device == torch.device("cpu")

    def test_action_shape_and_range(self):
        """Test that actions have correct shape and are within valid range."""
        agent = RLAgent("penguin_0")
        obs = np.zeros(89, dtype=np.float32)

        # Sample multiple actions to check range consistently
        for _ in range(50):
            action = agent.get_action(obs)

            assert action.shape == (2,), f"Expected shape (2,), got {action.shape}"
            assert action.dtype == np.float32, f"Expected float32, got {action.dtype}"

            # Angle should be in [0, 360] (sigmoid-bounded)
            assert 0.0 <= action[0] <= 360.0, f"Angle {action[0]} out of [0, 360]"

            # Power should be in [0, 500] (sigmoid-bounded)
            assert 0.0 <= action[1] <= 500.0, f"Power {action[1]} out of [0, 500]"

    def test_save_load_roundtrip(self, tmp_path):
        """Test that saving and loading preserves weights (deterministic network output)."""
        # Create agent and get deterministic network output for fixed observation
        torch.manual_seed(42)
        agent1 = RLAgent("penguin_0")
        obs = np.random.default_rng(123).standard_normal(89).astype(np.float32)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean1, log_std1, value1 = agent1.network(obs_tensor)

        # Save weights
        save_path = tmp_path / "test_weights.pt"
        agent1.save(save_path)

        # Create new agent with different random init, then load saved weights
        torch.manual_seed(999)
        agent2 = RLAgent("penguin_1")

        # Before loading, outputs should (almost certainly) differ
        with torch.no_grad():
            mean_before, _, _ = agent2.network(obs_tensor)
        assert not torch.allclose(mean1, mean_before), "Different inits should differ"

        agent2.load(save_path)

        # After loading, deterministic outputs should match exactly
        with torch.no_grad():
            mean2, log_std2, value2 = agent2.network(obs_tensor)

        torch.testing.assert_close(mean1, mean2, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(log_std1, log_std2, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(value1, value2, rtol=1e-5, atol=1e-6)

    def test_actor_critic_forward(self):
        """Test that ActorCritic forward produces correct tensor shapes."""
        network = ActorCritic(obs_dim=89, action_dim=2, hidden_sizes=(128, 64))

        # Single observation
        obs = torch.randn(1, 89)
        action_mean, log_std, value = network(obs)

        assert action_mean.shape == (1, 2), f"action_mean shape: {action_mean.shape}"
        assert log_std.shape == (1, 2), f"log_std shape: {log_std.shape}"
        assert value.shape == (1, 1), f"value shape: {value.shape}"

        # Batch of observations
        obs_batch = torch.randn(16, 89)
        action_mean_b, log_std_b, value_b = network(obs_batch)

        assert action_mean_b.shape == (16, 2)
        assert log_std_b.shape == (16, 2)
        assert value_b.shape == (16, 1)

        # get_action_and_value
        action, log_prob, entropy, val = network.get_action_and_value(obs_batch)

        assert action.shape == (16, 2)
        assert log_prob.shape == (16,)
        assert entropy.shape == (16,)
        assert val.shape == (16,)
