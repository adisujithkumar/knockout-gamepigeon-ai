#!/usr/bin/env python3
"""Comprehensive tournament: RL checkpoints vs heuristic and head-to-head.

Runs 4 tournaments:
  1. Contrastive checkpoints vs Heuristic (100 games each)
  2. Attention checkpoints vs Heuristic (100 games each)
  3. LLM Architect checkpoints vs Heuristic (100 games each)
  4. Head-to-head: final agents from each training run (100 games each pair)

Usage:
    .venv/bin/python scripts/run_tournament.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from knockout.agents.base import Agent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.agents.rl_agent import RLAgent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.env.penguin_env import PenguinEnv


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(
    agent_a: Agent,
    agent_b: Agent,
    config: GameConfig = DEFAULTS,
    seed: int = 0,
    max_steps: int = 200,
) -> dict:
    """Run a single game between two agents (each controls full team).

    agent_a controls Team A (penguin_0,1,2), agent_b controls Team B (penguin_3,4,5).
    Returns dict with 'winner' (0=A, 1=B, -1=draw), 'steps', alive counts.
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
            if idx < 3:
                actions[agent_id] = agent_a.get_action(obs)
            else:
                actions[agent_id] = agent_b.get_action(obs)

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


def run_matchup(
    agent_a: Agent,
    agent_b: Agent,
    n_games: int = 100,
    config: GameConfig = DEFAULTS,
    label: str = "",
) -> dict:
    """Run N games between two agents and collect statistics.

    Returns dict with wins_a, wins_b, draws, win_rate_a, win_rate_b.
    """
    wins_a = 0
    wins_b = 0
    draws = 0

    for i in range(n_games):
        result = run_game(agent_a, agent_b, config=config, seed=i * 7 + 13)
        if result["winner"] == 0:
            wins_a += 1
        elif result["winner"] == 1:
            wins_b += 1
        else:
            draws += 1

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": wins_a / n_games,
        "win_rate_b": wins_b / n_games,
        "n_games": n_games,
    }


def load_rl_agent(checkpoint_path: str | Path, name: str = "rl") -> RLAgent:
    """Load an RL agent from checkpoint."""
    agent = RLAgent(agent_id=name)
    agent.load(Path(checkpoint_path))
    agent.network.eval()
    return agent


# ---------------------------------------------------------------------------
# Tournament runners
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_matchup_result(label_a: str, label_b: str, result: dict) -> None:
    wr_a = result["win_rate_a"]
    wr_b = result["win_rate_b"]
    bar_len = 30
    filled = int(round(wr_a * bar_len))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(
        f"  {label_a:40s} vs {label_b:20s}  "
        f"{result['wins_a']:3d}W / {result['wins_b']:3d}L / {result['draws']:3d}D  "
        f"[{bar}] {wr_a:6.1%}"
    )


def tournament_vs_heuristic(
    name: str,
    checkpoints: list[tuple[str, str | Path]],
    n_games: int = 100,
) -> list[dict]:
    """Run each checkpoint against the heuristic agent."""
    print_header(f"Tournament: {name} vs Heuristic ({n_games} games each)")

    heuristic = HeuristicAgent("heuristic", seed=99, config=DEFAULTS)
    results = []

    for label, ckpt_path in checkpoints:
        path = Path(ckpt_path)
        if not path.exists():
            print(f"  [SKIP] {label} -- file not found: {path}")
            results.append({"label": label, "skipped": True})
            continue

        t0 = time.time()
        rl_agent = load_rl_agent(path, name=label)
        result = run_matchup(rl_agent, heuristic, n_games=n_games, label=label)
        elapsed = time.time() - t0

        result["label"] = label
        result["elapsed_s"] = elapsed
        results.append(result)

        print_matchup_result(label, "Heuristic", result)
        sys.stdout.flush()

    return results


def tournament_head_to_head(
    agents: list[tuple[str, str | Path]],
    n_games: int = 100,
) -> list[dict]:
    """Run each pair of agents against each other."""
    print_header(f"Head-to-Head Tournament ({n_games} games per pair)")

    loaded = []
    for label, path in agents:
        p = Path(path)
        if not p.exists():
            print(f"  [SKIP] {label} -- file not found: {p}")
            loaded.append((label, None))
        else:
            loaded.append((label, load_rl_agent(p, name=label)))

    results = []
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            label_a, agent_a = loaded[i]
            label_b, agent_b = loaded[j]

            if agent_a is None or agent_b is None:
                print(f"  [SKIP] {label_a} vs {label_b}")
                continue

            t0 = time.time()
            result = run_matchup(agent_a, agent_b, n_games=n_games)
            elapsed = time.time() - t0

            result["label_a"] = label_a
            result["label_b"] = label_b
            result["elapsed_s"] = elapsed
            results.append(result)

            print_matchup_result(label_a, label_b, result)
            sys.stdout.flush()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base = project_root / "runs"
    N = 100  # games per matchup

    overall_t0 = time.time()

    # --- Tournament 1: Contrastive vs Heuristic ---
    contrastive_ckpts = [
        ("contrastive/iter_000", base / "contrastive/checkpoints/agent_iter_000.pt"),
        ("contrastive/iter_001", base / "contrastive/checkpoints/agent_iter_001.pt"),
        ("contrastive/iter_002", base / "contrastive/checkpoints/agent_iter_002.pt"),
        ("contrastive/iter_003", base / "contrastive/checkpoints/agent_iter_003.pt"),
        ("contrastive/iter_004", base / "contrastive/checkpoints/agent_iter_004.pt"),
        ("contrastive/final",    base / "contrastive/final_agent.pt"),
    ]
    t1_results = tournament_vs_heuristic("Contrastive", contrastive_ckpts, n_games=N)

    # --- Tournament 2: Attention vs Heuristic ---
    attention_ckpts = [
        ("attention/step_250K",  base / "attention/checkpoints/agent_step_0245760.pt"),
        ("attention/step_500K",  base / "attention/checkpoints/agent_step_0491520.pt"),
        ("attention/step_1M",    base / "attention/checkpoints/agent_step_0983040.pt"),
        ("attention/step_1.5M",  base / "attention/checkpoints/agent_step_1474560.pt"),
        ("attention/step_2M",    base / "attention/checkpoints/agent_step_1966080.pt"),
        ("attention/step_2.5M",  base / "attention/checkpoints/agent_step_2457600.pt"),
        ("attention/final",      base / "attention/final_agent.pt"),
    ]
    t2_results = tournament_vs_heuristic("Attention", attention_ckpts, n_games=N)

    # --- Tournament 3: LLM Architect vs Heuristic ---
    llm_ckpts = [
        ("llm_architect/iter_000", base / "llm_architect/iteration_000/agent_checkpoint.pt"),
        ("llm_architect/iter_001", base / "llm_architect/iteration_001/agent_checkpoint.pt"),
        ("llm_architect/iter_002", base / "llm_architect/iteration_002/agent_checkpoint.pt"),
    ]
    t3_results = tournament_vs_heuristic("LLM Architect", llm_ckpts, n_games=N)

    # --- Tournament 4: Head-to-Head finals ---
    final_agents = [
        ("Contrastive-final", base / "contrastive/final_agent.pt"),
        ("Attention-final",   base / "attention/final_agent.pt"),
        ("LLMArchitect-final", base / "llm_architect/iteration_002/agent_checkpoint.pt"),
    ]
    t4_results = tournament_head_to_head(final_agents, n_games=N)

    # --- Summary table ---
    total_time = time.time() - overall_t0
    print()
    print("=" * 70)
    print("  FULL RESULTS SUMMARY")
    print("=" * 70)

    # vs Heuristic tables
    for tname, tresults in [
        ("Contrastive vs Heuristic", t1_results),
        ("Attention vs Heuristic", t2_results),
        ("LLM Architect vs Heuristic", t3_results),
    ]:
        print(f"\n  --- {tname} ---")
        print(f"  {'Checkpoint':<35s} {'Win%':>6s} {'W':>4s} {'L':>4s} {'D':>4s} {'Time':>7s}")
        print(f"  {'-'*35} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*7}")
        for r in tresults:
            if r.get("skipped"):
                print(f"  {r['label']:<35s} {'SKIP':>6s}")
                continue
            print(
                f"  {r['label']:<35s} "
                f"{r['win_rate_a']:5.1%} "
                f"{r['wins_a']:4d} "
                f"{r['wins_b']:4d} "
                f"{r['draws']:4d} "
                f"{r['elapsed_s']:6.1f}s"
            )

    # Head-to-head table
    print(f"\n  --- Head-to-Head (Final Agents) ---")
    print(f"  {'Agent A':<22s} {'Agent B':<22s} {'A Win%':>6s} {'W':>4s} {'L':>4s} {'D':>4s} {'Time':>7s}")
    print(f"  {'-'*22} {'-'*22} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*7}")
    for r in t4_results:
        print(
            f"  {r['label_a']:<22s} {r['label_b']:<22s} "
            f"{r['win_rate_a']:5.1%} "
            f"{r['wins_a']:4d} "
            f"{r['wins_b']:4d} "
            f"{r['draws']:4d} "
            f"{r['elapsed_s']:6.1f}s"
        )

    print(f"\n  Total tournament time: {total_time:.1f}s")
    print()


if __name__ == "__main__":
    main()
