"""PPO training CLI."""

from pathlib import Path
from typing import Callable

import torch

from knockout.agents.base import Agent
from knockout.agents.random_agent import RandomAgent
from knockout.training.ppo import PPOTrainer
from knockout.training.plots import plot_training_curves, save_logs_csv


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", type=str, default="checkpoints/ppo_agent.pt")
    parser.add_argument("--log-path", type=str, default="training_log.csv")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel environments (1 = legacy single-env)")
    parser.add_argument("--opponent", choices=["random", "heuristic"], default="random",
                        help="Opponent policy for Team B")
    parser.add_argument("--backend", choices=["pymunk", "tensor"], default="tensor",
                        help="Environment backend: pymunk (CPU) or tensor (GPU-accelerated)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for networks and tensor backend (cpu, cuda, auto)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Build opponent factory
    if args.opponent == "heuristic":
        from knockout.agents.heuristic_agent import HeuristicAgent
        opponent_factory: Callable[[], Agent] = lambda: HeuristicAgent("opp")
    else:
        opponent_factory = lambda: RandomAgent("opp")  # noqa: E731

    print(f"Training PPO: backend={args.backend}, device={device}, "
          f"num_envs={args.num_envs}, opponent={args.opponent}")

    trainer = PPOTrainer(
        lr=args.lr,
        num_envs=args.num_envs,
        opponent_factory=opponent_factory,
        device=device,
        backend=args.backend,
    )
    logs = trainer.train(total_timesteps=args.timesteps)

    trainer.agent.save(Path(args.save_path))
    save_logs_csv(logs, args.log_path)
    plot_training_curves(logs, save_path="training_curves.png")
    print(f"Done. Saved to {args.save_path}")


if __name__ == "__main__":
    main()
