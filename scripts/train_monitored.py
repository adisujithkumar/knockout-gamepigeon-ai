"""Monitored training CLI — PPO or MAPPO with evaluation, stall detection, and auto-intervention.

Usage:
    .venv/bin/python scripts/train_monitored.py \
        --algorithm ppo \
        --timesteps 5000000 \
        --num-envs 4096 \
        --device cuda \
        --backend tensor \
        --eval-interval 50000 \
        --eval-games 50 \
        --stall-patience 5 \
        --checkpoint-dir checkpoints/ppo_monitored/ \
        --log-path logs/ppo_monitored.csv
"""

import argparse

import torch

from knockout.training.monitored_trainer import MonitoredTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Monitored training with evaluation, stall detection, and auto-intervention",
    )
    parser.add_argument(
        "--algorithm", choices=["ppo", "mappo"], default="ppo",
        help="Training algorithm (default: ppo)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000,
        help="Total environment timesteps (default: 5M)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=4096,
        help="Number of parallel environments (default: 4096)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: cpu, cuda, auto (default: auto)",
    )
    parser.add_argument(
        "--backend", choices=["pymunk", "tensor"], default="tensor",
        help="Environment backend (default: tensor)",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=50000,
        help="Steps between evaluations (default: 50000)",
    )
    parser.add_argument(
        "--eval-games", type=int, default=50,
        help="Games per evaluation (default: 50)",
    )
    parser.add_argument(
        "--stall-patience", type=int, default=5,
        help="Evaluations without improvement before intervention (default: 5)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/monitored/",
        help="Directory for checkpoints (default: checkpoints/monitored/)",
    )
    parser.add_argument(
        "--log-path", type=str, default="logs/training_monitor.csv",
        help="Path for CSV log (default: logs/training_monitor.csv)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=128,
        help="Rollout length in steps (default: 128)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=4,
        help="PPO epochs per update (default: 4)",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01,
        help="Entropy coefficient (default: 0.01)",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    trainer = MonitoredTrainer(
        algorithm=args.algorithm,
        device=device,
        num_envs=args.num_envs,
        backend=args.backend,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        stall_patience=args.stall_patience,
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log_path,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        entropy_coef=args.entropy_coef,
    )

    trainer.train(total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
