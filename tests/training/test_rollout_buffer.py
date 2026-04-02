"""Tests for PPO rollout buffer with GAE computation."""

import numpy as np
import pytest
import torch

from knockout.training.rollout_buffer import RolloutBuffer


class TestRolloutBuffer:
    """Tests for RolloutBuffer."""

    def test_creation(self):
        """Test buffer can be created with correct dimensions."""
        buf = RolloutBuffer(buffer_size=100, obs_dim=89, action_dim=2)
        assert buf.buffer_size == 100
        assert buf.obs_dim == 89
        assert buf.action_dim == 2
        assert buf.pos == 0
        assert buf.full is False
        assert buf.observations.shape == (100, 89)
        assert buf.actions.shape == (100, 2)
        assert buf.log_probs.shape == (100,)
        assert buf.rewards.shape == (100,)
        assert buf.values.shape == (100,)
        assert buf.dones.shape == (100,)

    def test_add_transition(self):
        """Test adding transitions increments position."""
        buf = RolloutBuffer(buffer_size=10, obs_dim=4, action_dim=2)

        obs = np.array([1.0, 2.0, 3.0, 4.0])
        action = np.array([0.5, 0.3])
        buf.add(obs=obs, action=action, log_prob=-1.5, reward=1.0, value=0.5, done=False)

        assert buf.pos == 1
        np.testing.assert_array_almost_equal(buf.observations[0], obs)
        np.testing.assert_array_almost_equal(buf.actions[0], action)
        assert buf.log_probs[0] == pytest.approx(-1.5)
        assert buf.rewards[0] == pytest.approx(1.0)
        assert buf.values[0] == pytest.approx(0.5)
        assert buf.dones[0] == pytest.approx(0.0)

        # Add until full
        for i in range(9):
            buf.add(obs=obs, action=action, log_prob=-1.0, reward=0.0, value=0.0, done=False)

        assert buf.pos == 10
        assert buf.full is True

    def test_compute_gae_simple(self):
        """Test GAE computation with known values.

        Setup: 3-step trajectory, no terminals, gamma=0.99, lambda=0.95
        rewards = [1, 2, 3], values = [0.5, 1.0, 1.5], last_value = 2.0

        Expected deltas:
        delta_2 = r_2 + gamma * V_next * 1.0 - V_2 = 3 + 0.99*2.0 - 1.5 = 3.48
        delta_1 = r_1 + gamma * V_2 * 1.0 - V_1 = 2 + 0.99*1.5 - 1.0 = 2.485
        delta_0 = r_0 + gamma * V_1 * 1.0 - V_0 = 1 + 0.99*1.0 - 0.5 = 1.49

        GAE (reversed):
        A_2 = delta_2 = 3.48
        A_1 = delta_1 + gamma*lambda*A_2 = 2.485 + 0.99*0.95*3.48 = 5.75706
        A_0 = delta_0 + gamma*lambda*A_1 = 1.49 + 0.99*0.95*5.75706 = 6.9028..
        """
        buf = RolloutBuffer(buffer_size=10, obs_dim=2, action_dim=1)

        # Add 3 transitions
        buf.add(obs=[0, 0], action=[0], log_prob=0, reward=1.0, value=0.5, done=False)
        buf.add(obs=[0, 0], action=[0], log_prob=0, reward=2.0, value=1.0, done=False)
        buf.add(obs=[0, 0], action=[0], log_prob=0, reward=3.0, value=1.5, done=False)

        buf.compute_gae(gamma=0.99, gae_lambda=0.95, last_value=2.0, last_done=False)

        # Verify advantages
        delta_2 = 3.0 + 0.99 * 2.0 - 1.5
        delta_1 = 2.0 + 0.99 * 1.5 - 1.0
        delta_0 = 1.0 + 0.99 * 1.0 - 0.5

        a_2 = delta_2
        a_1 = delta_1 + 0.99 * 0.95 * a_2
        a_0 = delta_0 + 0.99 * 0.95 * a_1

        assert buf.advantages[0] == pytest.approx(a_0, abs=1e-5)
        assert buf.advantages[1] == pytest.approx(a_1, abs=1e-5)
        assert buf.advantages[2] == pytest.approx(a_2, abs=1e-5)

        # Returns = advantages + values
        assert buf.returns[0] == pytest.approx(a_0 + 0.5, abs=1e-5)
        assert buf.returns[1] == pytest.approx(a_1 + 1.0, abs=1e-5)
        assert buf.returns[2] == pytest.approx(a_2 + 1.5, abs=1e-5)

    def test_compute_gae_terminal(self):
        """Test that done=True at step t cuts bootstrapping FROM step t.

        The done flag at step t means the episode ended at step t.
        Therefore step t does NOT bootstrap from V(s_{t+1}), and its
        advantage does NOT propagate past the episode boundary.

        However, step t-1 (which is still in the same episode) DOES
        bootstrap from V(s_t) and carries the advantage from step t.
        """
        buf = RolloutBuffer(buffer_size=10, obs_dim=2, action_dim=1)

        # Step 0: normal (done=False), Step 1: terminal (done=True)
        buf.add(obs=[0, 0], action=[0], log_prob=0, reward=1.0, value=0.5, done=False)
        buf.add(obs=[0, 0], action=[0], log_prob=0, reward=5.0, value=1.0, done=True)

        buf.compute_gae(gamma=0.99, gae_lambda=0.95, last_value=10.0, last_done=False)

        # Step 1 (done=True): non_terminal = 0, no bootstrap from last_value
        # delta_1 = 5.0 + 0.99 * 10.0 * 0 - 1.0 = 4.0
        delta_1 = 5.0 + 0.0 - 1.0  # = 4.0
        a_1 = delta_1  # = 4.0

        # Step 0 (done=False): non_terminal = 1, bootstraps from V(s_1)
        # delta_0 = 1.0 + 0.99 * 1.0 * 1.0 - 0.5 = 1.49
        # A_0 = delta_0 + gamma * lambda * 1.0 * A_1 = 1.49 + 0.99*0.95*4.0
        delta_0 = 1.0 + 0.99 * 1.0 - 0.5  # = 1.49
        a_0 = delta_0 + 0.99 * 0.95 * a_1  # = 1.49 + 3.762 = 5.252

        assert buf.advantages[0] == pytest.approx(a_0, abs=1e-5)
        assert buf.advantages[1] == pytest.approx(a_1, abs=1e-5)

    def test_get_batches_covers_all_data(self):
        """Test that get_batches yields all indices exactly once."""
        buf = RolloutBuffer(buffer_size=20, obs_dim=4, action_dim=2)

        # Add 10 transitions
        for i in range(10):
            buf.add(
                obs=np.ones(4) * i,
                action=np.ones(2) * i,
                log_prob=float(i),
                reward=float(i),
                value=float(i),
                done=False,
            )
        buf.compute_gae()

        # Collect all batch observations
        all_obs = []
        for batch in buf.get_batches(batch_size=3):
            assert "observations" in batch
            assert "actions" in batch
            assert "log_probs" in batch
            assert "advantages" in batch
            assert "returns" in batch
            assert "values" in batch
            all_obs.append(batch["observations"])

        # Concatenate and check all 10 samples are present
        all_obs_cat = torch.cat(all_obs, dim=0)
        assert all_obs_cat.shape[0] == 10

        # Each observation sum should appear exactly once (obs = [i,i,i,i], sum=4i)
        sums = sorted(all_obs_cat.sum(dim=1).tolist())
        expected_sums = sorted([4.0 * i for i in range(10)])
        for a, b in zip(sums, expected_sums):
            assert a == pytest.approx(b, abs=1e-5)

    def test_reset(self):
        """Test that reset clears the buffer state."""
        buf = RolloutBuffer(buffer_size=10, obs_dim=4, action_dim=2)

        for i in range(5):
            buf.add(
                obs=np.ones(4),
                action=np.ones(2),
                log_prob=0.0,
                reward=1.0,
                value=0.5,
                done=False,
            )

        assert buf.pos == 5
        buf.reset()
        assert buf.pos == 0
        assert buf.full is False
