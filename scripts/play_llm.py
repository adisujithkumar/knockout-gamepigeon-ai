#!/usr/bin/env python3
"""Run LLM-controlled Team A vs CPU Team B (heuristic or random).

Usage:
    .venv/bin/python scripts/play_llm.py --model claude-sonnet-4-20250514 --seed 42 --games 3
    .venv/bin/python scripts/play_llm.py --opponent random --games 1 --verbose

Options:
    --model       LLM model identifier (default: claude-sonnet-4-20250514)
    --opponent    Opponent type: heuristic or random (default: heuristic)
    --seed        Random seed (default: 42)
    --games       Number of games to play (default: 1)
    --verbose     Show full LLM reasoning for each round
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knockout.agents.game_state_text import obs_to_text
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.agents.llm_agent import LLMAgent
from knockout.agents.random_agent import RandomAgent
from knockout.core.config import DEFAULTS
from knockout.env.penguin_env import PenguinEnv


TEAM_A_IDS = ["penguin_0", "penguin_1", "penguin_2"]
TEAM_B_IDS = ["penguin_3", "penguin_4", "penguin_5"]


def make_opponent(kind: str, agent_id: str, seed: int):
    """Create an opponent agent."""
    if kind == "random":
        return RandomAgent(agent_id, seed=seed)
    return HeuristicAgent(agent_id, seed=seed)


def play_one_game(
    llm_agents: dict[str, LLMAgent],
    opponent_agents: dict[str, object],
    seed: int,
    verbose: bool = False,
    game_num: int = 1,
) -> dict:
    """Play a single game and return result dict."""

    env = PenguinEnv(seed=seed, max_steps=200)
    obs_dict, _ = env.reset()

    print(f"\n{'='*60}")
    print(f"  GAME {game_num}  (seed={seed})")
    print(f"{'='*60}")

    round_num = 0
    while env.agents:
        round_num += 1
        print(f"\n--- Round {round_num} ---")

        actions = {}

        # LLM team (Team A)
        for agent_id in TEAM_A_IDS:
            if agent_id not in env.agents:
                continue
            obs = obs_dict[agent_id]
            agent = llm_agents[agent_id]

            if verbose:
                state_text = obs_to_text(obs, agent_id)
                print(f"\n[{agent_id}] State:\n{state_text}")

            action = agent.get_action(obs)
            actions[agent_id] = action

            angle, power = float(action[0]), float(action[1])
            print(f"  {agent_id} -> angle={angle:.1f}, power={power:.1f}")
            if verbose and agent.last_reasoning:
                reasoning = agent.last_reasoning
                # Truncate very long responses
                if len(reasoning) > 400:
                    reasoning = reasoning[:400] + "..."
                print(f"    LLM reasoning: {reasoning}")

        # Opponent team (Team B)
        for agent_id in TEAM_B_IDS:
            if agent_id not in env.agents:
                continue
            obs = obs_dict[agent_id]
            action = opponent_agents[agent_id].get_action(obs)
            actions[agent_id] = action
            angle, power = float(action[0]), float(action[1])
            print(f"  {agent_id} -> angle={angle:.1f}, power={power:.1f}")

        # Step environment
        obs_dict, rewards, terms, truncs, infos = env.step(actions)

        # Check for eliminations
        alive_a = env.physics_engine.get_alive_count(0)
        alive_b = env.physics_engine.get_alive_count(1)
        print(f"  After round: Team A={alive_a} alive, Team B={alive_b} alive")

        if any(terms.values()) or any(truncs.values()):
            break

    # Determine winner
    winner = env.physics_engine.get_winner()
    alive_a = env.physics_engine.get_alive_count(0)
    alive_b = env.physics_engine.get_alive_count(1)

    if winner == 0:
        result_str = "Team A (LLM) WINS"
    elif winner == 1:
        result_str = "Team B (Opponent) WINS"
    else:
        result_str = "DRAW"

    print(f"\n  RESULT: {result_str}  ({alive_a}v{alive_b}, {round_num} rounds)")

    # LLM fallback stats
    total_calls = sum(a.call_count for a in llm_agents.values())
    total_fallbacks = sum(a.fallback_count for a in llm_agents.values())
    if total_fallbacks > 0:
        print(f"  LLM fallbacks: {total_fallbacks}/{total_calls}")

    env.close()
    return {
        "winner": winner if winner is not None else -1,
        "rounds": round_num,
        "alive_a": alive_a,
        "alive_b": alive_b,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM vs CPU penguin knockout")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="LLM model to use")
    parser.add_argument("--opponent", choices=["heuristic", "random"],
                        default="heuristic", help="Opponent type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--games", type=int, default=1, help="Number of games")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full LLM reasoning")
    args = parser.parse_args()

    # Verify API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Opponent: {args.opponent}")
    print(f"Games: {args.games}")

    # Create agents
    llm_agents = {}
    for agent_id in TEAM_A_IDS:
        llm_agents[agent_id] = LLMAgent(
            agent_id,
            model=args.model,
            api_key=api_key,
            seed=args.seed,
        )

    opponent_agents = {}
    for agent_id in TEAM_B_IDS:
        opponent_agents[agent_id] = make_opponent(
            args.opponent, agent_id, seed=args.seed
        )

    # Play games
    results = []
    wins_a = 0
    wins_b = 0
    draws = 0

    for game_idx in range(args.games):
        game_seed = args.seed + game_idx
        result = play_one_game(
            llm_agents, opponent_agents,
            seed=game_seed,
            verbose=args.verbose,
            game_num=game_idx + 1,
        )
        results.append(result)

        if result["winner"] == 0:
            wins_a += 1
        elif result["winner"] == 1:
            wins_b += 1
        else:
            draws += 1

        # Reset agents between games
        for a in llm_agents.values():
            a.reset()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({args.games} game{'s' if args.games > 1 else ''})")
    print(f"{'='*60}")
    print(f"  LLM (Team A) wins:  {wins_a}")
    print(f"  Opponent wins:      {wins_b}")
    print(f"  Draws:              {draws}")
    if args.games > 0:
        win_rate = wins_a / args.games
        print(f"  LLM win rate:       {win_rate:.1%}")


if __name__ == "__main__":
    main()
