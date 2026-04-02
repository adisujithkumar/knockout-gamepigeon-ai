"""Evaluation harness for agent matchups."""

from __future__ import annotations

from typing import Optional

import numpy as np

from knockout.agents.base import Agent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.env.penguin_env import PenguinEnv


def run_match(
    team_a: dict[str, Agent],
    team_b: dict[str, Agent],
    config: GameConfig = DEFAULTS,
    seed: int = 42,
    max_steps: int = 200,
) -> dict:
    """Run a single match between two teams of agents.

    Args:
        team_a: dict mapping agent_id -> Agent for team A (penguin_0,1,2)
        team_b: dict mapping agent_id -> Agent for team B (penguin_3,4,5)
        config: game config
        seed: random seed
        max_steps: max environment steps

    Returns:
        dict with 'winner' (0, 1, or -1), 'steps', 'team_a_alive', 'team_b_alive'
    """
    env = PenguinEnv(seed=seed, max_steps=max_steps, config=config)
    obs_dict, _ = env.reset()

    for step in range(max_steps):
        if not env.agents:
            break

        actions = {}
        for agent_id in env.agents:
            obs = obs_dict[agent_id]
            idx = int(agent_id.split("_")[1])
            if idx < 3 and agent_id in team_a:
                actions[agent_id] = team_a[agent_id].get_action(obs)
            elif idx >= 3 and agent_id in team_b:
                actions[agent_id] = team_b[agent_id].get_action(obs)
            else:
                actions[agent_id] = np.array([0.0, 0.0], dtype=np.float32)

        obs_dict, rewards, terms, truncs, infos = env.step(actions)

        if any(terms.values()) or any(truncs.values()):
            break

    winner = env.physics_engine.get_winner()
    result = {
        "winner": winner if winner is not None else -1,
        "steps": step + 1,
        "team_a_alive": env.physics_engine.get_alive_count(0),
        "team_b_alive": env.physics_engine.get_alive_count(1),
    }
    env.close()
    return result


def round_robin(
    agents: dict[str, Agent],
    n_games: int = 10,
    config: GameConfig = DEFAULTS,
) -> dict[str, dict]:
    """Run round-robin tournament. Each agent plays as both team A and B.

    Args:
        agents: dict of name -> Agent (each agent will be cloned for all 3 team slots)
        n_games: games per matchup
        config: game config

    Returns:
        dict of agent_name -> {wins, losses, draws, games, win_rate}
    """
    names = list(agents.keys())
    results = {name: {"wins": 0, "losses": 0, "draws": 0, "games": 0} for name in names}

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                continue

            agent_a = agents[name_a]
            agent_b = agents[name_b]

            # Create full teams (same agent for all 3 slots)
            team_a = {f"penguin_{k}": agent_a for k in range(3)}
            team_b = {f"penguin_{k}": agent_b for k in range(3, 6)}

            for game in range(n_games):
                result = run_match(
                    team_a, team_b, config=config, seed=i * 1000 + j * 100 + game
                )

                if result["winner"] == 0:
                    results[name_a]["wins"] += 1
                    results[name_b]["losses"] += 1
                elif result["winner"] == 1:
                    results[name_b]["wins"] += 1
                    results[name_a]["losses"] += 1
                else:
                    results[name_a]["draws"] += 1
                    results[name_b]["draws"] += 1

                results[name_a]["games"] += 1
                results[name_b]["games"] += 1

    # Compute win rates
    for name in names:
        r = results[name]
        r["win_rate"] = r["wins"] / max(r["games"], 1)

    return results


def main():
    """CLI entry point for benchmarking agents."""
    from knockout.agents.random_agent import RandomAgent
    from knockout.agents.heuristic_agent import HeuristicAgent

    agents = {
        "random": RandomAgent("bench_random", seed=42),
        "heuristic": HeuristicAgent("bench_heuristic", seed=42),
    }

    print("Running round-robin tournament...")
    results = round_robin(agents, n_games=20)

    print("\nResults:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["win_rate"], reverse=True):
        print(
            f"  {name}: {r['wins']}W/{r['losses']}L/{r['draws']}D "
            f"({r['win_rate']:.1%} win rate)"
        )
