"""Self-play training CLI entry point."""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from knockout.training.self_play import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Self-play training for Knockout penguins"
    )
    parser.add_argument(
        "--generations", type=int, default=50,
        help="Number of self-play generations (default: 50)",
    )
    parser.add_argument(
        "--rollouts-per-gen", type=int, default=20,
        help="PPO rollouts per generation of training (default: 20)",
    )
    parser.add_argument(
        "--eval-games", type=int, default=10,
        help="Games per opponent during evaluation (default: 10)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=10,
        help="Max opponents in pool (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/self_play/",
        help="Directory for model checkpoints (default: checkpoints/self_play/)",
    )
    parser.add_argument(
        "--log-path", type=str, default="logs/self_play.csv",
        help="CSV output path for training logs (default: logs/self_play.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (optional)",
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=128,
        help="PPO rollout length per collection (default: 128)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="PPO mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    args = parser.parse_args()

    trainer = SelfPlayTrainer(
        checkpoint_dir=args.checkpoint_dir,
        pool_size=args.pool_size,
        num_envs=args.num_envs,
        seed=args.seed,
        log_path=args.log_path,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    logs = trainer.train(
        n_generations=args.generations,
        rollouts_per_gen=args.rollouts_per_gen,
        eval_games=args.eval_games,
    )

    print(f"\nTraining complete. {len(logs)} generations.")
    print(f"Final ELO: {logs[-1]['elo']:.0f}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    if args.log_path:
        print(f"Logs: {args.log_path}")

    # Print final leaderboard
    leaderboard = trainer.elo_tracker.get_leaderboard()
    print("\nELO Leaderboard:")
    for name, elo in leaderboard:
        print(f"  {name}: {elo:.0f}")


if __name__ == "__main__":
    main()
