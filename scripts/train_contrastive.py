#!/usr/bin/env python3
"""CLI script for contrastive trajectory mining reward discovery.

Creates TensorVecEnv + PPO + ContrastiveTrainer, runs the outer loop,
saves discovery logs, checkpoints, and metrics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from knockout.core.config import DEFAULTS
from knockout.env.tensor_env import TensorVecEnv
from knockout.reward.contrastive import ContrastiveConfig, ContrastiveTrainer
from knockout.training.ppo import PPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contrastive trajectory mining for reward discovery"
    )
    parser.add_argument(
        "--num-envs", type=int, default=64,
        help="Number of parallel environments (default: 64)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Torch device: cpu, cuda, or auto (default: auto)",
    )
    parser.add_argument(
        "--collection-episodes", type=int, default=2000,
        help="Games to play per analysis iteration (default: 2000)",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=10,
        help="Outer loop iterations (default: 10)",
    )
    parser.add_argument(
        "--steps-per-iteration", type=int, default=1_000_000,
        help="PPO training steps per iteration (default: 1000000)",
    )
    parser.add_argument(
        "--min-effect-size", type=float, default=0.3,
        help="Cohen's d threshold for feature significance (default: 0.3)",
    )
    parser.add_argument(
        "--max-features", type=int, default=10,
        help="Max discovered reward features (default: 10)",
    )
    parser.add_argument(
        "--reward-scale", type=float, default=0.5,
        help="Scale of shaped reward vs sparse (default: 0.5)",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/contrastive",
        help="Directory for discovery logs (default: logs/contrastive)",
    )
    parser.add_argument(
        "--save-path", type=str, default="checkpoints/contrastive_agent.pt",
        help="Path to save final agent checkpoint",
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logging.info("Device: %s, num_envs: %d", device, args.num_envs)

    # Build env
    env = TensorVecEnv(
        num_envs=args.num_envs,
        config=DEFAULTS,
        device=device,
    )

    # Build PPO trainer (no opponent needed -- TensorVecEnv handles Team B)
    ppo = PPOTrainer(
        config=DEFAULTS,
        lr=args.lr,
        device=device,
        num_envs=args.num_envs,
        backend="tensor",
    )

    # Build contrastive config
    contrastive_config = ContrastiveConfig(
        collection_episodes=args.collection_episodes,
        min_effect_size=args.min_effect_size,
        max_reward_features=args.max_features,
        reward_scale=args.reward_scale,
        num_iterations=args.num_iterations,
        steps_per_iteration=args.steps_per_iteration,
    )

    # Build trainer and run
    trainer = ContrastiveTrainer(
        env=env,
        ppo_trainer=ppo,
        config=contrastive_config,
        game_config=DEFAULTS,
        device=device,
        log_dir=args.log_dir,
    )

    metrics = trainer.run()

    # Save final agent
    save_path = Path(args.save_path)
    ppo.agent.save(save_path)
    logging.info("Saved agent to %s", save_path)

    # Summary
    for m in metrics:
        n_feat = m["num_features_discovered"]
        it = m["iteration"]
        elapsed = m["elapsed_seconds"]
        logging.info(
            "Iteration %d: %d features, %.1fs",
            it, n_feat, elapsed,
        )

    env.close()
    logging.info("Done. Discovery log: %s/discovery_log.json", args.log_dir)


if __name__ == "__main__":
    main()
