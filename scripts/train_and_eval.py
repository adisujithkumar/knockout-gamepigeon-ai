"""Full training and evaluation pipeline for PPO and MAPPO with tensor backend.

Runs training, evaluates against random and heuristic, and does head-to-head.
"""

import time
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from knockout.training.ppo import PPOTrainer
from knockout.training.mappo import MAPPOTrainer
from knockout.training.evaluation import run_match
from knockout.agents.rl_agent import RLAgent
from knockout.agents.mappo_agent import MAPPOAgent, MAPPOEvalAgent
from knockout.agents.random_agent import RandomAgent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.core.config import DEFAULTS


def evaluate_ppo(checkpoint_path: str, n_games: int = 100, device: str = "cpu") -> dict:
    """Evaluate a PPO checkpoint against random and heuristic opponents."""
    agent = RLAgent("eval", device=device)
    agent.load(Path(checkpoint_path))
    agent.network.eval()

    results = {}
    for opp_name, opp_factory in [("random", lambda: RandomAgent("opp", seed=None)),
                                   ("heuristic", lambda: HeuristicAgent("opp", seed=None))]:
        wins = 0
        losses = 0
        draws = 0
        for game_idx in range(n_games):
            team_a = {f"penguin_{k}": agent for k in range(3)}
            opp = opp_factory()
            team_b = {f"penguin_{k}": opp for k in range(3, 6)}
            result = run_match(team_a, team_b, seed=game_idx * 7 + 13)
            if result["winner"] == 0:
                wins += 1
            elif result["winner"] == 1:
                losses += 1
            else:
                draws += 1
        results[opp_name] = {
            "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / n_games
        }
        print(f"  vs {opp_name}: {wins}W/{losses}L/{draws}D = {wins/n_games:.1%}")

    return results


def evaluate_mappo(checkpoint_path: str, n_games: int = 100, device: str = "cpu") -> dict:
    """Evaluate a MAPPO checkpoint against random and heuristic opponents."""
    mappo = MAPPOAgent(device=device)
    mappo.load(Path(checkpoint_path))
    mappo.eval()

    results = {}
    for opp_name, opp_factory in [("random", lambda: RandomAgent("opp", seed=None)),
                                   ("heuristic", lambda: HeuristicAgent("opp", seed=None))]:
        wins = 0
        losses = 0
        draws = 0
        for game_idx in range(n_games):
            team_a = {
                f"penguin_{k}": MAPPOEvalAgent(f"penguin_{k}", mappo, agent_index=k)
                for k in range(3)
            }
            opp = opp_factory()
            team_b = {f"penguin_{k}": opp for k in range(3, 6)}
            result = run_match(team_a, team_b, seed=game_idx * 7 + 13)
            if result["winner"] == 0:
                wins += 1
            elif result["winner"] == 1:
                losses += 1
            else:
                draws += 1
        results[opp_name] = {
            "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / n_games
        }
        print(f"  vs {opp_name}: {wins}W/{losses}L/{draws}D = {wins/n_games:.1%}")

    return results


def head_to_head(ppo_path: str, mappo_path: str, n_games: int = 100, device: str = "cpu") -> dict:
    """PPO vs MAPPO head-to-head."""
    ppo_agent = RLAgent("ppo", device=device)
    ppo_agent.load(Path(ppo_path))
    ppo_agent.network.eval()

    mappo = MAPPOAgent(device=device)
    mappo.load(Path(mappo_path))
    mappo.eval()

    # PPO as Team A, MAPPO as Team B
    ppo_wins = 0
    mappo_wins = 0
    draws = 0
    for game_idx in range(n_games):
        team_a = {f"penguin_{k}": ppo_agent for k in range(3)}
        team_b = {
            f"penguin_{k}": MAPPOEvalAgent(f"penguin_{k}", mappo, agent_index=k-3)
            for k in range(3, 6)
        }
        result = run_match(team_a, team_b, seed=game_idx * 11 + 7)
        if result["winner"] == 0:
            ppo_wins += 1
        elif result["winner"] == 1:
            mappo_wins += 1
        else:
            draws += 1

    # Also run with swapped sides
    for game_idx in range(n_games):
        team_a = {
            f"penguin_{k}": MAPPOEvalAgent(f"penguin_{k}", mappo, agent_index=k)
            for k in range(3)
        }
        team_b = {f"penguin_{k}": ppo_agent for k in range(3, 6)}
        result = run_match(team_a, team_b, seed=game_idx * 11 + 7 + 10000)
        if result["winner"] == 0:
            mappo_wins += 1
        elif result["winner"] == 1:
            ppo_wins += 1
        else:
            draws += 1

    total = 2 * n_games
    return {
        "ppo_wins": ppo_wins,
        "mappo_wins": mappo_wins,
        "draws": draws,
        "ppo_win_rate": ppo_wins / total,
        "mappo_win_rate": mappo_wins / total,
    }


def train_ppo(timesteps, num_envs, opponent, save_path, device, backend):
    """Train PPO and return (logs, elapsed_time)."""
    from knockout.agents.heuristic_agent import HeuristicAgent

    if opponent == "heuristic":
        opp_factory = lambda: HeuristicAgent("opp")
    else:
        opp_factory = lambda: RandomAgent("opp")

    print(f"\n{'='*60}")
    print(f"PPO Training: {timesteps} steps, {num_envs} envs, opponent={opponent}")
    print(f"  backend={backend}, device={device}")
    print(f"{'='*60}")

    trainer = PPOTrainer(
        num_envs=num_envs,
        opponent_factory=opp_factory,
        device=device,
        backend=backend,
    )

    t0 = time.time()
    logs = trainer.train(total_timesteps=timesteps, log_interval=1)
    elapsed = time.time() - t0

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.agent.save(Path(save_path))
    print(f"  Saved to {save_path} ({elapsed:.1f}s)")

    return logs, elapsed


def train_mappo(timesteps, num_envs, opponent, save_path, device, backend):
    """Train MAPPO and return (logs, elapsed_time)."""
    from knockout.agents.heuristic_agent import HeuristicAgent

    if opponent == "heuristic":
        opp_factory = lambda: HeuristicAgent("opp")
    else:
        opp_factory = lambda: RandomAgent("opp")

    print(f"\n{'='*60}")
    print(f"MAPPO Training: {timesteps} steps, {num_envs} envs, opponent={opponent}")
    print(f"  backend={backend}, device={device}")
    print(f"{'='*60}")

    trainer = MAPPOTrainer(
        num_envs=num_envs,
        opponent_factory=opp_factory,
        device=device,
        backend=backend,
    )

    t0 = time.time()
    logs = trainer.train(total_timesteps=timesteps, log_interval=1)
    elapsed = time.time() - t0

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.agent.save(Path(save_path))
    print(f"  Saved to {save_path} ({elapsed:.1f}s)")
    trainer.close()

    return logs, elapsed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "tensor"
    # Use fewer envs to get more rollouts (=more PPO updates) per run.
    # With 64 envs * 128 steps * 3 agents = 24,576 transitions per rollout.
    # 500k timesteps -> ~20 rollouts; 2M -> ~81 rollouts.
    num_envs = 64
    n_eval_games = 100

    print(f"Device: {device}")
    print(f"Backend: {backend}")
    print(f"Num envs: {num_envs}")

    results_table = {}

    # ---- PPO vs Random (quick) ----
    logs, elapsed = train_ppo(
        timesteps=500_000, num_envs=num_envs, opponent="random",
        save_path="checkpoints/ppo_random_500k.pt",
        device=device, backend=backend,
    )
    results_table["PPO vs Random (500k)"] = {"train_time": elapsed}
    print("\nEvaluating PPO (trained vs random):")
    eval_res = evaluate_ppo("checkpoints/ppo_random_500k.pt", n_eval_games, device)
    results_table["PPO vs Random (500k)"].update(eval_res)

    # ---- PPO vs Heuristic ----
    logs, elapsed = train_ppo(
        timesteps=2_000_000, num_envs=num_envs, opponent="heuristic",
        save_path="checkpoints/ppo_heuristic_2M.pt",
        device=device, backend=backend,
    )
    results_table["PPO vs Heuristic (2M)"] = {"train_time": elapsed}
    print("\nEvaluating PPO (trained vs heuristic):")
    eval_res = evaluate_ppo("checkpoints/ppo_heuristic_2M.pt", n_eval_games, device)
    results_table["PPO vs Heuristic (2M)"].update(eval_res)

    # ---- MAPPO vs Random ----
    logs, elapsed = train_mappo(
        timesteps=500_000, num_envs=num_envs, opponent="random",
        save_path="checkpoints/mappo_random_500k.pt",
        device=device, backend=backend,
    )
    results_table["MAPPO vs Random (500k)"] = {"train_time": elapsed}
    print("\nEvaluating MAPPO (trained vs random):")
    eval_res = evaluate_mappo("checkpoints/mappo_random_500k.pt", n_eval_games, device)
    results_table["MAPPO vs Random (500k)"].update(eval_res)

    # ---- MAPPO vs Heuristic ----
    logs, elapsed = train_mappo(
        timesteps=2_000_000, num_envs=num_envs, opponent="heuristic",
        save_path="checkpoints/mappo_heuristic_2M.pt",
        device=device, backend=backend,
    )
    results_table["MAPPO vs Heuristic (2M)"] = {"train_time": elapsed}
    print("\nEvaluating MAPPO (trained vs heuristic):")
    eval_res = evaluate_mappo("checkpoints/mappo_heuristic_2M.pt", n_eval_games, device)
    results_table["MAPPO vs Heuristic (2M)"].update(eval_res)

    # ---- Head-to-Head: best PPO vs best MAPPO ----
    print(f"\n{'='*60}")
    print("Head-to-Head: PPO (heuristic-trained) vs MAPPO (heuristic-trained)")
    print(f"{'='*60}")
    h2h = head_to_head(
        "checkpoints/ppo_heuristic_2M.pt",
        "checkpoints/mappo_heuristic_2M.pt",
        n_games=n_eval_games,
        device=device,
    )
    print(f"  PPO wins: {h2h['ppo_wins']}, MAPPO wins: {h2h['mappo_wins']}, "
          f"Draws: {h2h['draws']}")
    print(f"  PPO win rate: {h2h['ppo_win_rate']:.1%}, "
          f"MAPPO win rate: {h2h['mappo_win_rate']:.1%}")

    # ---- Summary Table ----
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Run':<30} {'Time':>8} {'vs Random':>10} {'vs Heuristic':>12}")
    print("-" * 64)
    for name, data in results_table.items():
        train_t = f"{data['train_time']:.0f}s"
        vs_rand = f"{data.get('random', {}).get('win_rate', 0):.0%}"
        vs_heur = f"{data.get('heuristic', {}).get('win_rate', 0):.0%}"
        print(f"{name:<30} {train_t:>8} {vs_rand:>10} {vs_heur:>12}")

    print(f"\nHead-to-Head (200 games, both sides):")
    print(f"  PPO: {h2h['ppo_win_rate']:.1%} | MAPPO: {h2h['mappo_win_rate']:.1%} | "
          f"Draw: {h2h['draws']/(2*n_eval_games):.1%}")


if __name__ == "__main__":
    main()
