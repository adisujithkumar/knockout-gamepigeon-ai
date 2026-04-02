"""Tests for MAPPO (Multi-Agent PPO) training."""

import math

import numpy as np
import pytest
import torch

from knockout.agents.mappo_agent import (
    CriticNetwork,
    MAPPOAgent,
    MAPPOEvalAgent,
    PolicyNetwork,
)
from knockout.agents.random_agent import RandomAgent
from knockout.core.config import DEFAULTS
from knockout.training.mappo import MAPPORolloutBuffer, MAPPOTrainer


# ---------------------------------------------------------------------------
# MAPPOAgent network tests
# ---------------------------------------------------------------------------


class TestMAPPOAgent:
    """Tests for the MAPPO agent networks."""

    def test_mappo_agent_creation(self):
        """Test MAPPOAgent creates correct number of networks with right shapes."""
        agent = MAPPOAgent(obs_dim=89, action_dim=2, num_agents=3)

        assert len(agent.policies) == 3
        assert agent.obs_dim == 89
        assert agent.action_dim == 2
        assert agent.num_agents == 3

        # Check each policy is an independent PolicyNetwork
        for p in agent.policies:
            assert isinstance(p, PolicyNetwork)

        # Check critic exists
        assert isinstance(agent.critic, CriticNetwork)

    def test_policy_forward(self):
        """Test each policy produces valid actions with correct shapes."""
        agent = MAPPOAgent()

        obs = torch.randn(1, 89)
        for i, policy in enumerate(agent.policies):
            action, log_prob, entropy = policy.get_action_and_log_prob(obs)

            assert action.shape == (1, 2), f"Policy {i}: action shape {action.shape}"
            assert log_prob.shape == (1,), f"Policy {i}: log_prob shape {log_prob.shape}"
            assert entropy.shape == (1,), f"Policy {i}: entropy shape {entropy.shape}"
            assert torch.isfinite(action).all(), f"Policy {i}: non-finite action"
            assert torch.isfinite(log_prob).all(), f"Policy {i}: non-finite log_prob"

    def test_policy_forward_batch(self):
        """Test policy forward with batch dimension."""
        policy = PolicyNetwork(obs_dim=89, action_dim=2)
        obs_batch = torch.randn(16, 89)

        action, log_prob, entropy = policy.get_action_and_log_prob(obs_batch)
        assert action.shape == (16, 2)
        assert log_prob.shape == (16,)
        assert entropy.shape == (16,)

    def test_critic_forward(self):
        """Test critic accepts concatenated obs and outputs scalar."""
        critic = CriticNetwork(input_dim=267)  # 3 * 89

        # Single sample
        concat_obs = torch.randn(1, 267)
        value = critic(concat_obs)
        assert value.shape == (1,), f"Critic single shape: {value.shape}"
        assert torch.isfinite(value).all()

        # Batch
        concat_obs_batch = torch.randn(16, 267)
        values = critic(concat_obs_batch)
        assert values.shape == (16,), f"Critic batch shape: {values.shape}"
        assert torch.isfinite(values).all()

    def test_get_actions_shape(self):
        """Test get_actions returns (3, 2) actions for single sample."""
        agent = MAPPOAgent()

        # Single sample: (3, 89)
        team_obs = torch.randn(3, 89)
        actions, log_probs, entropies = agent.get_actions(team_obs)

        assert actions.shape == (3, 2), f"Actions shape: {actions.shape}"
        assert log_probs.shape == (3,), f"Log probs shape: {log_probs.shape}"
        assert entropies.shape == (3,), f"Entropies shape: {entropies.shape}"

    def test_get_actions_shape_batch(self):
        """Test get_actions returns correct batch shapes."""
        agent = MAPPOAgent()

        # Batch: (8, 3, 89)
        team_obs = torch.randn(8, 3, 89)
        actions, log_probs, entropies = agent.get_actions(team_obs)

        assert actions.shape == (8, 3, 2)
        assert log_probs.shape == (8, 3)
        assert entropies.shape == (8, 3)

    def test_get_actions_with_given_actions(self):
        """Test get_actions evaluates log_prob for given actions."""
        agent = MAPPOAgent()
        team_obs = torch.randn(4, 3, 89)
        given_actions = torch.randn(4, 3, 2)

        actions, log_probs, entropies = agent.get_actions(team_obs, given_actions)

        # Should return the given actions, not new samples
        torch.testing.assert_close(actions, given_actions)
        assert log_probs.shape == (4, 3)
        assert torch.isfinite(log_probs).all()

    def test_get_value_shape(self):
        """Test get_value returns scalar for single and batch inputs."""
        agent = MAPPOAgent()

        # Single: (3, 89) -> scalar
        team_obs = torch.randn(3, 89)
        value = agent.get_value(team_obs)
        assert value.ndim == 0, f"Single value ndim: {value.ndim}"
        assert torch.isfinite(value)

        # Batch: (8, 3, 89) -> (8,)
        team_obs_batch = torch.randn(8, 3, 89)
        values = agent.get_value(team_obs_batch)
        assert values.shape == (8,), f"Batch values shape: {values.shape}"
        assert torch.isfinite(values).all()

    def test_get_action_single_agent(self):
        """Test decentralized get_action for evaluation."""
        agent = MAPPOAgent()
        obs = np.zeros(89, dtype=np.float32)

        for idx in range(3):
            action = agent.get_action(obs, agent_index=idx)
            assert action.shape == (2,)
            assert action.dtype == np.float32
            # Sigmoid-scaled: angle in [0, 360], power in [0, 500]
            assert 0.0 <= action[0] <= 360.0
            assert 0.0 <= action[1] <= 500.0

    def test_save_load(self, tmp_path):
        """Test round-trip serialization preserves weights."""
        torch.manual_seed(42)
        agent1 = MAPPOAgent()

        obs = torch.randn(3, 89)
        with torch.no_grad():
            actions1, lp1, _ = agent1.get_actions(obs)
            value1 = agent1.get_value(obs)

        # Save
        save_path = tmp_path / "mappo_test.pt"
        agent1.save(save_path)

        # Load into fresh agent
        torch.manual_seed(999)
        agent2 = MAPPOAgent()

        # Before load, outputs should differ
        with torch.no_grad():
            _, lp_before, _ = agent2.get_actions(obs)
        assert not torch.allclose(lp1, lp_before), "Different inits should differ"

        agent2.load(save_path)

        # After load, outputs should match
        with torch.no_grad():
            actions2, lp2, _ = agent2.get_actions(obs)
            value2 = agent2.get_value(obs)

        # Note: actions are sampled, so we compare deterministic outputs (value + log_prob of same action)
        torch.testing.assert_close(value1, value2, rtol=1e-5, atol=1e-6)

        # Evaluate same action under both to confirm policy weights match
        with torch.no_grad():
            _, lp1_eval, _ = agent1.get_actions(obs, actions1)
            _, lp2_eval, _ = agent2.get_actions(obs, actions1)
        torch.testing.assert_close(lp1_eval, lp2_eval, rtol=1e-5, atol=1e-6)

    def test_independent_policies(self):
        """Test that different policies produce different actions (not weight-sharing bug).

        After random init, the 3 policies should have different weights and
        therefore produce different outputs for the same input.
        """
        agent = MAPPOAgent()

        # Use a distinctive observation (not zeros)
        obs = torch.randn(1, 89)

        means = []
        for policy in agent.policies:
            mean, _ = policy(obs)
            means.append(mean.detach())

        # At least two policies should produce different means
        all_same = True
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                if not torch.allclose(means[i], means[j], atol=1e-6):
                    all_same = False
                    break
            if not all_same:
                break

        assert not all_same, (
            "All 3 policies produced identical outputs -- "
            "they may be sharing weights unintentionally"
        )

    def test_eval_agent_wrapping(self):
        """Test MAPPOEvalAgent wraps correctly for PettingZoo Agent interface."""
        mappo = MAPPOAgent()
        eval_agent = MAPPOEvalAgent("penguin_0", mappo, agent_index=0)

        obs = np.zeros(89, dtype=np.float32)
        action = eval_agent.get_action(obs)

        assert action.shape == (2,)
        assert 0.0 <= action[0] <= 360.0
        assert 0.0 <= action[1] <= 500.0

    def test_train_eval_mode(self):
        """Test train/eval mode switching."""
        agent = MAPPOAgent()

        agent.eval()
        for p in agent.policies:
            assert not p.training
        assert not agent.critic.training

        agent.train()
        for p in agent.policies:
            assert p.training
        assert agent.critic.training


# ---------------------------------------------------------------------------
# MAPPORolloutBuffer tests
# ---------------------------------------------------------------------------


class TestMAPPORolloutBuffer:
    """Tests for the MAPPO rollout buffer."""

    def test_buffer_creation(self):
        """Test buffer allocates arrays with correct shapes."""
        buf = MAPPORolloutBuffer(buffer_size=32, obs_dim=89, action_dim=2, num_agents=3)

        assert buf.observations.shape == (32, 3, 89)
        assert buf.actions.shape == (32, 3, 2)
        assert buf.log_probs.shape == (32, 3)
        assert buf.rewards.shape == (32, 3)
        assert buf.values.shape == (32,)
        assert buf.dones.shape == (32,)
        assert buf.masks.shape == (32, 3)

    def test_buffer_add_and_size(self):
        """Test adding transitions to buffer."""
        buf = MAPPORolloutBuffer(buffer_size=4)

        for i in range(3):
            buf.add(
                obs=np.zeros((3, 89), dtype=np.float32),
                actions=np.zeros((3, 2), dtype=np.float32),
                log_probs=np.zeros(3, dtype=np.float32),
                rewards=np.ones(3, dtype=np.float32) * i,
                value=float(i),
                done=False,
                masks=np.ones(3, dtype=np.float32),
            )

        assert buf.pos == 3
        assert buf.size == 3
        assert not buf.full

    def test_buffer_compute_gae(self):
        """Test GAE computation doesn't produce NaN."""
        buf = MAPPORolloutBuffer(buffer_size=16)

        for i in range(16):
            buf.add(
                obs=np.random.randn(3, 89).astype(np.float32),
                actions=np.random.randn(3, 2).astype(np.float32),
                log_probs=np.random.randn(3).astype(np.float32),
                rewards=np.random.randn(3).astype(np.float32),
                value=float(np.random.randn()),
                done=(i == 15),
                masks=np.ones(3, dtype=np.float32),
            )

        buf.compute_gae(gamma=0.99, gae_lambda=0.95)

        assert np.isfinite(buf.advantages[:buf.pos]).all()
        assert np.isfinite(buf.returns[:buf.pos]).all()

    def test_buffer_get_batches(self):
        """Test batch generation yields correct shapes."""
        buf = MAPPORolloutBuffer(buffer_size=16)

        for _ in range(16):
            buf.add(
                obs=np.random.randn(3, 89).astype(np.float32),
                actions=np.random.randn(3, 2).astype(np.float32),
                log_probs=np.random.randn(3).astype(np.float32),
                rewards=np.zeros(3, dtype=np.float32),
                value=0.0,
                done=False,
                masks=np.ones(3, dtype=np.float32),
            )

        buf.compute_gae()
        batches = list(buf.get_batches(batch_size=8))

        assert len(batches) == 2
        for batch in batches:
            B = batch["observations"].shape[0]
            assert batch["observations"].shape == (B, 3, 89)
            assert batch["actions"].shape == (B, 3, 2)
            assert batch["log_probs"].shape == (B, 3)
            assert batch["advantages"].shape == (B,)
            assert batch["returns"].shape == (B,)
            assert batch["masks"].shape == (B, 3)


# ---------------------------------------------------------------------------
# MAPPOTrainer tests
# ---------------------------------------------------------------------------


class TestMAPPOTrainer:
    """Tests for the MAPPO trainer."""

    def test_trainer_creation(self):
        """Test trainer creates with expected defaults."""
        trainer = MAPPOTrainer(num_envs=2)

        assert trainer.agent is not None
        assert len(trainer.agent.policies) == 3
        assert trainer.num_envs == 2
        assert trainer.gamma == 0.99
        assert trainer.clip_eps == 0.2
        trainer.close()

    def test_trainer_collect_rollout(self):
        """Test collect_rollout fills buffer with correct shapes."""
        trainer = MAPPOTrainer(num_envs=2, rollout_steps=8)

        buffer = trainer.collect_rollout()

        # Should have num_envs * rollout_steps entries
        assert buffer.pos == 2 * 8
        assert buffer.observations[:buffer.pos].shape == (16, 3, 89)
        assert buffer.actions[:buffer.pos].shape == (16, 3, 2)

        # Should have real data (not all zeros)
        assert buffer.observations[:buffer.pos].any()
        assert np.isfinite(buffer.advantages[:buffer.pos]).all()
        assert np.isfinite(buffer.returns[:buffer.pos]).all()

        trainer.close()

    def test_trainer_train_step(self):
        """Test train_step returns metrics with no NaN."""
        trainer = MAPPOTrainer(num_envs=2, rollout_steps=8, batch_size=8, n_epochs=2)

        buffer = trainer.collect_rollout()
        metrics = trainer.train_step(buffer)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert math.isfinite(metrics["policy_loss"]), f"policy_loss: {metrics['policy_loss']}"
        assert math.isfinite(metrics["value_loss"]), f"value_loss: {metrics['value_loss']}"
        assert math.isfinite(metrics["entropy"]), f"entropy: {metrics['entropy']}"

        trainer.close()

    def test_trainer_smoke(self):
        """Run 3 rollouts of training, verify no crash and finite metrics."""
        trainer = MAPPOTrainer(
            num_envs=2,
            rollout_steps=8,
            batch_size=8,
            n_epochs=2,
        )

        for rollout_idx in range(3):
            buffer = trainer.collect_rollout()
            metrics = trainer.train_step(buffer)

            assert math.isfinite(metrics["policy_loss"]), (
                f"Rollout {rollout_idx}: NaN policy_loss"
            )
            assert math.isfinite(metrics["value_loss"]), (
                f"Rollout {rollout_idx}: NaN value_loss"
            )
            assert math.isfinite(metrics["entropy"]), (
                f"Rollout {rollout_idx}: NaN entropy"
            )

        trainer.close()

    def test_trainer_train_loop(self):
        """Test the full train() loop returns logs."""
        trainer = MAPPOTrainer(
            num_envs=2,
            rollout_steps=8,
            batch_size=8,
            n_epochs=2,
        )

        logs = trainer.train(total_timesteps=64, log_interval=100)

        assert len(logs) > 0
        assert "rollout" in logs[0]
        assert "policy_loss" in logs[0]
        assert "timesteps" in logs[0]

        for entry in logs:
            assert math.isfinite(entry["policy_loss"])
            assert math.isfinite(entry["value_loss"])

        trainer.close()

    def test_trainer_with_heuristic_opponent(self):
        """Test training against heuristic opponent doesn't crash."""
        from knockout.agents.heuristic_agent import HeuristicAgent

        trainer = MAPPOTrainer(
            num_envs=2,
            rollout_steps=8,
            batch_size=8,
            n_epochs=2,
            opponent_factory=lambda: HeuristicAgent("opp"),
        )

        buffer = trainer.collect_rollout()
        metrics = trainer.train_step(buffer)

        assert math.isfinite(metrics["policy_loss"])
        trainer.close()

    def test_trainer_save_load_checkpoint(self, tmp_path):
        """Test that trained agent can be saved and loaded."""
        trainer = MAPPOTrainer(
            num_envs=2,
            rollout_steps=8,
            batch_size=8,
            n_epochs=2,
        )

        # Do one training step
        buffer = trainer.collect_rollout()
        trainer.train_step(buffer)

        # Save
        path = tmp_path / "mappo_ckpt.pt"
        trainer.agent.save(path)

        assert path.exists()
        assert path.stat().st_size > 0

        # Load into fresh agent
        new_agent = MAPPOAgent()
        new_agent.load(path)

        # Verify loaded agent can produce actions
        obs = np.zeros(89, dtype=np.float32)
        action = new_agent.get_action(obs, agent_index=0)
        assert action.shape == (2,)
        assert np.isfinite(action).all()

        trainer.close()
