"""Integration tests for knockout-v2."""

import numpy as np
import pytest

from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.physics_engine import PhysicsEngine
from knockout.env.penguin_env import PenguinEnv
from knockout.env.observations import ObservationBuilder
from knockout.agents.random_agent import RandomAgent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.agents.rl_agent import RLAgent
from knockout.training.rollout_buffer import RolloutBuffer
from knockout.training.elo_rating import ELOTracker


class TestFullGameIntegration:
    """Test complete game flow from env creation to game over."""

    def test_full_game_random_vs_random(self):
        """Run a full game with random agents."""
        env = PenguinEnv(seed=42, max_steps=200)
        agents = {f"penguin_{i}": RandomAgent(f"penguin_{i}", seed=i) for i in range(6)}

        obs_dict, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            actions = {
                agent_id: agents[agent_id].get_action(obs_dict[agent_id])
                for agent_id in env.agents
            }
            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            done = any(terms.values()) or any(truncs.values())
            steps += 1

        # Game should complete (either terminated or truncated)
        assert done
        env.close()

    def test_full_game_heuristic_vs_random(self):
        """Run a full game with heuristic vs random."""
        env = PenguinEnv(seed=42, max_steps=200)
        team_a = {f"penguin_{i}": HeuristicAgent(f"penguin_{i}", seed=i) for i in range(3)}
        team_b = {f"penguin_{i}": RandomAgent(f"penguin_{i}", seed=i) for i in range(3, 6)}
        agents = {**team_a, **team_b}

        obs_dict, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            actions = {
                agent_id: agents[agent_id].get_action(obs_dict[agent_id])
                for agent_id in env.agents
            }
            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            done = any(terms.values()) or any(truncs.values())
            steps += 1

        assert done
        env.close()

    def test_full_game_rl_vs_random(self):
        """Run a full game with RL agent vs random."""
        env = PenguinEnv(seed=42, max_steps=100)
        team_a = {f"penguin_{i}": RLAgent(f"penguin_{i}") for i in range(3)}
        team_b = {f"penguin_{i}": RandomAgent(f"penguin_{i}", seed=i) for i in range(3, 6)}
        agents = {**team_a, **team_b}

        obs_dict, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            actions = {
                agent_id: agents[agent_id].get_action(obs_dict[agent_id])
                for agent_id in env.agents
            }
            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            done = any(terms.values()) or any(truncs.values())
            steps += 1

        env.close()

    def test_multiple_games_deterministic_replay(self):
        """Run same game twice with same seed, verify identical results."""
        def run_game(seed):
            env = PenguinEnv(seed=seed, max_steps=50)
            agents = {f"penguin_{i}": RandomAgent(f"penguin_{i}", seed=i) for i in range(6)}

            obs_dict, _ = env.reset()
            all_rewards = []

            for _ in range(50):
                if not env.agents:
                    break
                actions = {
                    agent_id: agents[agent_id].get_action(obs_dict[agent_id])
                    for agent_id in env.agents
                }
                obs_dict, rewards, terms, truncs, _ = env.step(actions)
                all_rewards.append(dict(rewards))
                if any(terms.values()) or any(truncs.values()):
                    break

            env.close()
            return all_rewards

        result1 = run_game(42)
        result2 = run_game(42)

        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            for key in r1:
                assert r1[key] == r2[key], f"Determinism violation: {r1} != {r2}"


class TestTrainThenEvaluate:
    """Test the train-then-evaluate pipeline."""

    def test_ppo_collect_and_train(self):
        """Collect rollout and train one step."""
        from knockout.training.ppo import PPOTrainer

        trainer = PPOTrainer(rollout_steps=32, batch_size=16)
        env = PenguinEnv(seed=42, max_steps=50)

        buffer = trainer.collect_rollout(env)
        assert buffer.pos > 0

        metrics = trainer.train_step(buffer)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])

        env.close()

    def test_evaluation_after_training(self):
        """Train briefly then evaluate."""
        from knockout.training.ppo import PPOTrainer
        from knockout.training.evaluation import run_match

        trainer = PPOTrainer(rollout_steps=32, batch_size=16)
        env = PenguinEnv(seed=42, max_steps=50)

        # Quick train
        buffer = trainer.collect_rollout(env)
        trainer.train_step(buffer)
        env.close()

        # Evaluate
        team_a = {f"penguin_{k}": trainer.agent for k in range(3)}
        team_b = {f"penguin_{k}": RandomAgent(f"r_{k}", seed=k) for k in range(3, 6)}

        result = run_match(team_a, team_b, seed=42, max_steps=50)
        assert "winner" in result
        assert result["winner"] in [0, 1, -1]


class TestCrossModuleBoundaries:
    """Test that modules interact correctly at boundaries."""

    def test_physics_engine_to_observations(self):
        """Test that observations correctly reflect physics state."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Before any actions
        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)
        assert obs.shape == (89,)

        # After launching penguin
        engine.apply_actions({"penguin_0": (45.0, 300.0)})
        engine.step(10)

        obs_after = builder.build_observation("penguin_0", engine.penguins, timestep=10)
        assert not np.allclose(obs, obs_after)

    def test_observation_bounds_contract(self):
        """Verify observation bounds contract holds under stress."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Launch all penguins with max force
        for i in range(6):
            engine.apply_actions({f"penguin_{i}": (float(i * 60), 500.0)})
        engine.step(200)

        for agent_id in [
            "penguin_0", "penguin_1", "penguin_2",
            "penguin_3", "penguin_4", "penguin_5",
        ]:
            if engine.penguins[agent_id].alive:
                obs = builder.build_observation(agent_id, engine.penguins, timestep=200)
                assert np.all(obs >= -1.0), f"Obs below -1 for {agent_id}: min={obs.min()}"
                assert np.all(obs <= 1.0), f"Obs above 1 for {agent_id}: max={obs.max()}"

    def test_env_to_agent_interface(self):
        """Test that env observations work with all agent types."""
        env = PenguinEnv(seed=42)
        obs_dict, _ = env.reset()

        random_agent = RandomAgent("test", seed=42)
        heuristic_agent = HeuristicAgent("test", seed=42)
        rl_agent = RLAgent("test")

        obs = obs_dict["penguin_0"]

        # All agents should produce valid actions
        for agent in [random_agent, heuristic_agent, rl_agent]:
            action = agent.get_action(obs)
            assert action.shape == (2,), f"{type(agent).__name__} bad shape: {action.shape}"
            assert action.dtype == np.float32

        env.close()
