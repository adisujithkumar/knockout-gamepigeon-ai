#!/usr/bin/env python3
"""Comprehensive round-robin tournament for all trained agents.

Runs every agent pair for 50 games as Team A + 50 games as Team B = 100 games
per matchup, reports win-rate matrix and ELO rankings.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from knockout.agents.random_agent import RandomAgent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.agents.rl_agent import RLAgent
from knockout.agents.mappo_agent import MAPPOAgent, MAPPOEvalAgent
from knockout.training.evaluation import run_match
from knockout.training.elo_rating import ELOTracker
from knockout.core.config import DEFAULTS

CHECKPOINT_DIR = Path("/mnt/d/projects/game-ai/knockout-v2/checkpoints")
GAMES_PER_SIDE = 50  # 50 as Team A + 50 as Team B = 100 per matchup


def make_team(agent_or_agents, team: str) -> dict:
    """Build a team dict from agent(s).

    For simple agents (Random, Heuristic, RLAgent), the same agent is used
    for all 3 penguin slots.  For MAPPO, a list of 3 MAPPOEvalAgents is
    expected (one per slot).

    Args:
        agent_or_agents: A single Agent or a list of 3 Agents.
        team: 'a' or 'b'

    Returns:
        Dict mapping penguin_id -> Agent
    """
    if team == "a":
        ids = [f"penguin_{i}" for i in range(3)]
    else:
        ids = [f"penguin_{i}" for i in range(3, 6)]

    if isinstance(agent_or_agents, list):
        assert len(agent_or_agents) == 3
        return {pid: agent for pid, agent in zip(ids, agent_or_agents)}
    else:
        return {pid: agent_or_agents for pid in ids}


def load_agents() -> dict[str, object]:
    """Load all agents, skipping any that fail.

    Returns dict of name -> agent (single Agent or list of 3 for MAPPO).
    """
    agents = {}

    # Built-in agents
    agents["Random"] = RandomAgent("random", seed=None)
    agents["Heuristic"] = HeuristicAgent("heuristic", seed=None)
    print("  [OK] Random")
    print("  [OK] Heuristic")

    # PPO agents
    ppo_checkpoints = [
        ("PPO-50K", "ppo_random_50k.pt"),
        ("PPO-200K", "ppo_heuristic_200k.pt"),
    ]
    for name, ckpt_file in ppo_checkpoints:
        path = CHECKPOINT_DIR / ckpt_file
        try:
            agent = RLAgent(agent_id=name.lower(), obs_dim=89, action_dim=2)
            agent.load(path)
            agent.network.eval()
            agents[name] = agent
            print(f"  [OK] {name} <- {ckpt_file}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    # MAPPO agents
    mappo_checkpoints = [
        ("MAPPO-50K", "mappo_random_50k.pt"),
        ("MAPPO-200K", "mappo_heuristic_200k.pt"),
    ]
    for name, ckpt_file in mappo_checkpoints:
        path = CHECKPOINT_DIR / ckpt_file
        try:
            mappo = MAPPOAgent(obs_dim=89, action_dim=2, num_agents=3)
            mappo.load(path)
            mappo.eval()
            # Wrap each policy in a MAPPOEvalAgent
            eval_agents = [
                MAPPOEvalAgent(
                    agent_id=f"{name.lower()}_slot{i}",
                    mappo_agent=mappo,
                    agent_index=i,
                )
                for i in range(3)
            ]
            agents[name] = eval_agents  # list of 3
            print(f"  [OK] {name} <- {ckpt_file} (3 policy slots)")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    return agents


def run_tournament(agents: dict) -> tuple[dict, ELOTracker]:
    """Run full round-robin tournament.

    Returns:
        (win_matrix, elo_tracker)
        win_matrix[name_a][name_b] = win count of name_a vs name_b
    """
    names = list(agents.keys())
    n = len(names)

    # win_matrix[a][b] = number of wins for a when playing against b
    win_matrix = {a: {b: 0 for b in names} for a in names}
    draw_matrix = {a: {b: 0 for b in names} for a in names}
    games_matrix = {a: {b: 0 for b in names} for a in names}

    elo = ELOTracker(initial_rating=1000.0, k=32.0)
    for name in names:
        elo.register(name)

    total_matchups = n * (n - 1)
    matchup_count = 0

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if i == j:
                continue

            matchup_count += 1
            agent_a = agents[name_a]
            agent_b = agents[name_b]

            # --- 50 games: name_a as Team A, name_b as Team B ---
            for g in range(GAMES_PER_SIDE):
                team_a = make_team(agent_a, "a")
                team_b = make_team(agent_b, "b")
                seed = i * 10000 + j * 1000 + g
                result = run_match(team_a, team_b, config=DEFAULTS, seed=seed)

                games_matrix[name_a][name_b] += 1
                games_matrix[name_b][name_a] += 1

                if result["winner"] == 0:  # Team A wins
                    win_matrix[name_a][name_b] += 1
                    elo.update(name_a, name_b, 1.0)
                elif result["winner"] == 1:  # Team B wins
                    win_matrix[name_b][name_a] += 1
                    elo.update(name_a, name_b, 0.0)
                else:  # Draw
                    draw_matrix[name_a][name_b] += 1
                    draw_matrix[name_b][name_a] += 1
                    elo.update(name_a, name_b, 0.5)

            # Progress
            sys.stdout.write(
                f"\r  Matchup {matchup_count}/{total_matchups}: "
                f"{name_a} vs {name_b} done "
                f"({GAMES_PER_SIDE} games as A)"
            )
            sys.stdout.flush()

    print()  # newline after progress
    return win_matrix, draw_matrix, games_matrix, elo


def print_results(names, win_matrix, draw_matrix, games_matrix, elo):
    """Print the win-rate matrix and ELO rankings."""
    n = len(names)

    # --- Win-rate matrix ---
    # For each (row, col), win_rate = row's wins / total games between row and col
    # Total games between a and b = games where a was Team A vs b as Team B
    #                              + games where b was Team A vs a as Team B
    # = GAMES_PER_SIDE * 2 = 100 (except self-play)

    print("\n" + "=" * 80)
    print("WIN-RATE MATRIX (row = agent, column = opponent)")
    print("Each cell = row agent's win rate against column agent (100 games)")
    print("=" * 80)

    # Column header
    col_width = 14
    header = " " * 16
    for name in names:
        header += f"{name:>{col_width}}"
    print(header)
    print("-" * len(header))

    for name_a in names:
        row = f"{name_a:<16}"
        for name_b in names:
            if name_a == name_b:
                row += f"{'---':>{col_width}}"
            else:
                total = win_matrix[name_a][name_b] + win_matrix[name_b][name_a] + draw_matrix[name_a][name_b]
                if total > 0:
                    wr = win_matrix[name_a][name_b] / total * 100
                    row += f"{wr:>{col_width - 1}.1f}%"
                else:
                    row += f"{'N/A':>{col_width}}"
        print(row)

    # --- Wins / Draws / Losses detail ---
    print(f"\n{'=' * 80}")
    print("DETAILED W/D/L  (row agent vs column agent, 100 games each)")
    print("=" * 80)
    header = " " * 16
    for name in names:
        header += f"{name:>{col_width}}"
    print(header)
    print("-" * len(header))
    for name_a in names:
        row = f"{name_a:<16}"
        for name_b in names:
            if name_a == name_b:
                row += f"{'---':>{col_width}}"
            else:
                w = win_matrix[name_a][name_b]
                d = draw_matrix[name_a][name_b]
                l = win_matrix[name_b][name_a]
                cell = f"{w}W/{d}D/{l}L"
                row += f"{cell:>{col_width}}"
        print(row)

    # --- ELO Rankings ---
    print(f"\n{'=' * 80}")
    print("ELO RANKINGS")
    print("=" * 80)
    leaderboard = elo.get_leaderboard()
    for rank, (name, rating) in enumerate(leaderboard, 1):
        print(f"  {rank}. {name:<16} {rating:.0f}")

    # --- Overall standings ---
    print(f"\n{'=' * 80}")
    print("OVERALL STANDINGS")
    print("=" * 80)
    standings = []
    for name in names:
        total_wins = sum(win_matrix[name][opp] for opp in names if opp != name)
        total_draws = sum(draw_matrix[name][opp] for opp in names if opp != name)
        total_losses = sum(win_matrix[opp][name] for opp in names if opp != name)
        total_games = total_wins + total_draws + total_losses
        overall_wr = total_wins / max(total_games, 1)
        standings.append((name, total_wins, total_draws, total_losses, total_games, overall_wr))

    standings.sort(key=lambda x: x[5], reverse=True)
    print(f"  {'Agent':<16} {'W':>5} {'D':>5} {'L':>5} {'Games':>7} {'Win%':>7}")
    print(f"  {'-' * 50}")
    for name, w, d, l, g, wr in standings:
        print(f"  {name:<16} {w:>5} {d:>5} {l:>5} {g:>7} {wr:>6.1%}")

    # --- Observations ---
    print(f"\n{'=' * 80}")
    print("OBSERVATIONS")
    print("=" * 80)

    # Find key matchup results
    def wr(a, b):
        total = win_matrix[a][b] + win_matrix[b][a] + draw_matrix[a][b]
        return win_matrix[a][b] / max(total, 1) * 100

    observations = []

    # PPO vs PPO
    if "PPO-50K" in names and "PPO-200K" in names:
        rate = wr("PPO-200K", "PPO-50K")
        observations.append(f"PPO-200K vs PPO-50K: {rate:.0f}% win rate for PPO-200K")

    # MAPPO vs MAPPO
    if "MAPPO-50K" in names and "MAPPO-200K" in names:
        rate = wr("MAPPO-200K", "MAPPO-50K")
        observations.append(f"MAPPO-200K vs MAPPO-50K: {rate:.0f}% win rate for MAPPO-200K")

    # MAPPO vs PPO (same training)
    if "PPO-50K" in names and "MAPPO-50K" in names:
        rate = wr("MAPPO-50K", "PPO-50K")
        observations.append(f"MAPPO-50K vs PPO-50K (both trained vs random): {rate:.0f}% MAPPO win rate")

    if "PPO-200K" in names and "MAPPO-200K" in names:
        rate = wr("MAPPO-200K", "PPO-200K")
        observations.append(f"MAPPO-200K vs PPO-200K (both trained vs heuristic): {rate:.0f}% MAPPO win rate")

    # RL vs baselines
    if "Heuristic" in names:
        for rl_name in ["PPO-50K", "PPO-200K", "MAPPO-50K", "MAPPO-200K"]:
            if rl_name in names:
                rate = wr(rl_name, "Heuristic")
                label = "BEATS" if rate > 50 else "LOSES TO" if rate < 50 else "TIES"
                observations.append(f"{rl_name} {label} Heuristic ({rate:.0f}% win rate)")

    for obs in observations:
        print(f"  - {obs}")


def main():
    print("=" * 80)
    print("KNOCKOUT TOURNAMENT — Round-Robin Evaluation")
    print(f"  {GAMES_PER_SIDE} games per side x 2 sides = {GAMES_PER_SIDE * 2} games per matchup")
    print("=" * 80)

    print("\nLoading agents...")
    agents = load_agents()
    print(f"\n  Loaded {len(agents)} agents: {', '.join(agents.keys())}")

    if len(agents) < 2:
        print("Need at least 2 agents to run a tournament!")
        sys.exit(1)

    total_matchups = len(agents) * (len(agents) - 1)
    total_games = total_matchups * GAMES_PER_SIDE
    print(f"  Total matchups: {total_matchups} (each = {GAMES_PER_SIDE} games one direction)")
    print(f"  Total games: {total_games}")

    print("\nRunning tournament...")
    t0 = time.time()
    win_matrix, draw_matrix, games_matrix, elo = run_tournament(agents)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s ({total_games / elapsed:.1f} games/sec)")

    names = list(agents.keys())
    print_results(names, win_matrix, draw_matrix, games_matrix, elo)


if __name__ == "__main__":
    main()
