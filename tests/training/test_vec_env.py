"""Tests for SingleTeamVecEnv."""

from __future__ import annotations

import numpy as np
import pytest

from knockout.agents.random_agent import RandomAgent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.core.config import DEFAULTS
from knockout.env.penguin_env import PenguinEnv
from knockout.training.vec_env import SingleTeamVecEnv


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_vec(num_envs: int = 4, seeds: list[int] | None = None, **kw):
    """Shorthand to create a vec env with deterministic seeds."""
    if seeds is None:
        seeds = list(range(num_envs))
    return SingleTeamVecEnv(num_envs=num_envs, seeds=seeds, **kw)


def _step_until_done(vec: SingleTeamVecEnv, max_steps: int = 2000):
    """Step a vec env with random actions until at least one env finishes."""
    rng = np.random.default_rng(99)
    obs, masks = vec.reset()
    for _ in range(max_steps):
        actions = np.stack([
            np.stack([
                np.array([rng.uniform(0, 360), rng.uniform(0, 400)], dtype=np.float32)
                for _ in range(3)
            ])
            for _ in range(vec.num_envs)
        ])
        obs, rewards, dones, masks, infos = vec.step(actions)
        if dones.any():
            return obs, rewards, dones, masks, infos
    pytest.skip("No env finished within max_steps (rare with random actions)")


# ---------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------

class TestShapes:
    def test_reset_shape(self):
        vec = _make_vec(4)
        obs, masks = vec.reset()
        assert obs.shape == (4, 3, 89)
        assert obs.dtype == np.float32
        assert masks.shape == (4, 3)
        assert masks.dtype == bool
        vec.close()

    def test_reset_masks_all_true(self):
        """After reset every Team-A agent should be alive."""
        vec = _make_vec(4)
        _, masks = vec.reset()
        assert masks.all()
        vec.close()

    def test_step_shapes(self):
        vec = _make_vec(4)
        obs, masks = vec.reset()
        actions = np.zeros((4, 3, 2), dtype=np.float32)
        next_obs, rewards, dones, next_masks, infos = vec.step(actions)

        assert next_obs.shape == (4, 3, 89)
        assert rewards.shape == (4, 3)
        assert dones.shape == (4,)
        assert next_masks.shape == (4, 3)
        assert isinstance(infos, list) and len(infos) == 4
        vec.close()

    def test_step_bad_action_shape_raises(self):
        vec = _make_vec(2)
        vec.reset()
        with pytest.raises(ValueError, match="Expected actions shape"):
            vec.step(np.zeros((3, 3, 2), dtype=np.float32))
        vec.close()

    def test_single_env_shapes(self):
        """N=1 should still produce correct batch dimensions."""
        vec = _make_vec(1, seeds=[42])
        obs, masks = vec.reset()
        assert obs.shape == (1, 3, 89)

        actions = np.zeros((1, 3, 2), dtype=np.float32)
        next_obs, rewards, dones, next_masks, infos = vec.step(actions)
        assert next_obs.shape == (1, 3, 89)
        assert dones.shape == (1,)
        vec.close()


# ---------------------------------------------------------------
# Auto-reset
# ---------------------------------------------------------------

class TestAutoReset:
    def test_auto_reset_returns_fresh_obs(self):
        """When an env episode ends, auto-reset should give non-zero obs."""
        vec = _make_vec(4)
        obs, rewards, dones, masks, infos = _step_until_done(vec)
        done_indices = np.where(dones)[0]

        for idx in done_indices:
            # After auto-reset, masks should all be True (all alive)
            assert masks[idx].all(), "After auto-reset all Team-A agents must be alive"
            # Observations should not be all-zero
            assert obs[idx].any(), "Post-reset obs should not be all zeros"
        vec.close()

    def test_done_env_rewards_terminal(self):
        """Terminal step should carry game-end rewards before auto-reset."""
        vec = _make_vec(4)
        _obs, rewards, dones, _masks, _infos = _step_until_done(vec)
        done_indices = np.where(dones)[0]

        # At least one env should have non-zero rewards at terminal step
        # (sparse +1/-1 for winner/loser)
        for idx in done_indices:
            # Reward is per-agent; the whole team gets the same sign
            team_rewards = rewards[idx]
            if team_rewards.any():
                # All agents on a team should get the same reward sign
                nonzero = team_rewards[team_rewards != 0.0]
                if len(nonzero) > 0:
                    assert (nonzero > 0).all() or (nonzero < 0).all()
        vec.close()


# ---------------------------------------------------------------
# Dead agent masking
# ---------------------------------------------------------------

class TestDeadAgentMasking:
    def test_dead_agent_zero_obs_and_mask(self):
        """If a Team-A agent dies mid-game, its obs should be zero and mask False."""
        vec = _make_vec(8)
        obs, masks = vec.reset()
        rng = np.random.default_rng(123)

        # Step many times; eventually a penguin may be knocked off
        found_dead = False
        for _ in range(500):
            actions = np.stack([
                np.stack([
                    np.array([rng.uniform(0, 360), rng.uniform(100, 400)], dtype=np.float32)
                    for _ in range(3)
                ])
                for _ in range(vec.num_envs)
            ])
            obs, rewards, dones, masks, infos = vec.step(actions)

            # Check for any env with a dead Team-A agent but game not over
            for i in range(vec.num_envs):
                if not dones[i] and not masks[i].all():
                    dead_idx = np.where(~masks[i])[0]
                    for j in dead_idx:
                        assert np.allclose(obs[i, j], 0.0), (
                            f"Dead agent obs should be zero, got {obs[i, j]}"
                        )
                    found_dead = True
                    break
            if found_dead:
                break

        # This test is probabilistic; skip rather than fail if never observed
        if not found_dead:
            pytest.skip("No Team-A agent died mid-game within test budget")
        vec.close()


# ---------------------------------------------------------------
# Opponent actions
# ---------------------------------------------------------------

class TestOpponentActions:
    def test_opponent_factory_is_used(self):
        """Opponent should receive real observations, not zeros."""
        call_log: list[np.ndarray] = []

        class SpyAgent(RandomAgent):
            def get_action(self, observation: np.ndarray) -> np.ndarray:
                call_log.append(observation.copy())
                return super().get_action(observation)

        vec = SingleTeamVecEnv(
            num_envs=2,
            opponent_factory=lambda: SpyAgent("spy"),
            seeds=[10, 11],
        )
        vec.reset()
        actions = np.zeros((2, 3, 2), dtype=np.float32)
        vec.step(actions)

        # Each env has up to 3 opponent agents -> up to 6 calls
        assert len(call_log) > 0, "Opponent agent was never called"
        # Observations should contain non-zero data (positions etc.)
        any_nonzero = any(obs.any() for obs in call_log)
        assert any_nonzero, "Opponent received only zero observations"
        vec.close()

    def test_heuristic_opponent(self):
        """Vec env should work with HeuristicAgent as opponent."""
        vec = SingleTeamVecEnv(
            num_envs=2,
            opponent_factory=lambda: HeuristicAgent("heur", seed=0),
            seeds=[20, 21],
        )
        obs, masks = vec.reset()
        actions = np.zeros((2, 3, 2), dtype=np.float32)
        # Should not raise
        next_obs, rewards, dones, next_masks, infos = vec.step(actions)
        assert next_obs.shape == (2, 3, 89)
        vec.close()


# ---------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------

class TestDeterminism:
    def test_same_seeds_same_results(self):
        """Two vec envs with the same seeds and same actions => identical results.

        Opponents must also be seeded so their actions are deterministic.
        """
        seeds = [100, 101, 102, 103]

        # Counter-based factory so each opponent gets a unique but
        # reproducible seed across runs.
        def _make_seeded_opp_factory():
            counter = [0]
            def factory():
                s = counter[0]
                counter[0] += 1
                return RandomAgent("opp", seed=s)
            return factory

        def run():
            rng = np.random.default_rng(0)
            vec = SingleTeamVecEnv(
                num_envs=4,
                seeds=seeds,
                opponent_factory=_make_seeded_opp_factory(),
            )
            obs, masks = vec.reset()
            all_obs = [obs.copy()]
            for _ in range(10):
                actions = rng.uniform(0, 1, (4, 3, 2)).astype(np.float32)
                actions[:, :, 0] *= 360.0
                actions[:, :, 1] *= 400.0
                obs, rewards, dones, masks, _ = vec.step(actions)
                all_obs.append(obs.copy())
            vec.close()
            return np.stack(all_obs)

        run1 = run()
        run2 = run()

        np.testing.assert_array_equal(run1, run2)


# ---------------------------------------------------------------
# Consistency with single env
# ---------------------------------------------------------------

class TestConsistency:
    def test_vec1_matches_single_env(self):
        """SingleTeamVecEnv(N=1) should produce the same Team-A obs/rewards
        as manually driving a PenguinEnv with the same seed."""
        seed = 77

        # -- Manual single env run --
        env = PenguinEnv(config=DEFAULTS, seed=seed)
        opp = RandomAgent("opp", seed=0)

        obs_dict, _ = env.reset()
        manual_obs = np.stack([obs_dict[a] for a in SingleTeamVecEnv.TEAM_A])

        # zero actions for learner, random for opponent
        actions_single: dict[str, np.ndarray] = {}
        for a in SingleTeamVecEnv.TEAM_A:
            actions_single[a] = np.zeros(2, dtype=np.float32)
        for a in SingleTeamVecEnv.TEAM_B:
            if a in env.agents:
                actions_single[a] = opp.get_action(obs_dict[a])

        next_obs_dict, rewards_dict, terms, truncs, infos = env.step(actions_single)
        manual_next_obs = np.stack(
            [next_obs_dict.get(a, np.zeros(89, dtype=np.float32))
             for a in SingleTeamVecEnv.TEAM_A]
        )
        manual_rewards = np.array(
            [rewards_dict.get(a, 0.0) for a in SingleTeamVecEnv.TEAM_A],
            dtype=np.float32,
        )
        env.close()

        # -- Vec env run --
        vec = SingleTeamVecEnv(
            num_envs=1,
            opponent_factory=lambda: RandomAgent("opp", seed=0),
            seeds=[seed],
        )
        vec_obs, _ = vec.reset()

        np.testing.assert_allclose(vec_obs[0], manual_obs, atol=1e-6)

        vec_actions = np.zeros((1, 3, 2), dtype=np.float32)
        vec_next_obs, vec_rewards, vec_dones, vec_masks, _ = vec.step(vec_actions)

        done = any(terms.values()) or any(truncs.values())
        if not done:
            np.testing.assert_allclose(vec_next_obs[0], manual_next_obs, atol=1e-6)
        np.testing.assert_allclose(vec_rewards[0], manual_rewards, atol=1e-6)
        vec.close()


# ---------------------------------------------------------------
# Rewards consistency
# ---------------------------------------------------------------

class TestRewardsMatch:
    def test_step_rewards_match_manual(self):
        """Rewards returned by vec env must match PenguinEnv.step() exactly."""
        seed = 55
        vec = SingleTeamVecEnv(
            num_envs=1,
            opponent_factory=lambda: RandomAgent("opp", seed=7),
            seeds=[seed],
        )
        env = PenguinEnv(config=DEFAULTS, seed=seed)
        opp = RandomAgent("opp", seed=7)

        vec_obs, _ = vec.reset()
        obs_dict, _ = env.reset()

        for _ in range(20):
            if not env.agents:
                break

            # Use zero actions for Team A
            vec_actions = np.zeros((1, 3, 2), dtype=np.float32)

            # Build manual actions
            manual_actions: dict[str, np.ndarray] = {}
            for a in SingleTeamVecEnv.TEAM_A:
                if a in env.agents:
                    manual_actions[a] = np.zeros(2, dtype=np.float32)
            for a in SingleTeamVecEnv.TEAM_B:
                if a in env.agents:
                    manual_actions[a] = opp.get_action(obs_dict[a])

            next_obs_dict, rewards_dict, terms, truncs, _ = env.step(manual_actions)
            vec_next_obs, vec_rewards, vec_dones, vec_masks, _ = vec.step(vec_actions)

            manual_rewards = np.array(
                [rewards_dict.get(a, 0.0) for a in SingleTeamVecEnv.TEAM_A],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                vec_rewards[0], manual_rewards, atol=1e-6,
                err_msg=f"Reward mismatch at some step",
            )

            done = any(terms.values()) or any(truncs.values())
            if done:
                break
            obs_dict = next_obs_dict

        vec.close()
        env.close()


# ---------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------

class TestEdgeCases:
    def test_num_envs_zero_raises(self):
        with pytest.raises(ValueError, match="num_envs must be >= 1"):
            SingleTeamVecEnv(num_envs=0)

    def test_seeds_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="len\\(seeds\\)"):
            SingleTeamVecEnv(num_envs=2, seeds=[1])

    def test_close_is_idempotent(self):
        vec = _make_vec(2)
        vec.reset()
        vec.close()
        vec.close()  # Should not raise
