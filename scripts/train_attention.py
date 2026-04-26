"""Train PPO with Feature Attention Discovery reward shaping.

Periodically runs gradient attribution on the value network and
crystallizes the top-K features into shaping reward components.
Logs what gets discovered and when.

Usage:
    python scripts/train_attention.py --timesteps 500000 --num-envs 64
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from knockout.core.config import DEFAULTS
from knockout.env.tensor_env import TensorVecEnv
from knockout.training.ppo import PPOTrainer
from knockout.reward.attention_discovery import AttentionConfig, AttentionTrainer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO + Feature Attention Discovery",
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/attention_agent.pt")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="runs/attention/checkpoints",
                        help="Directory for periodic checkpoints")
    parser.add_argument("--log-path", type=str,
                        default="attention_discovery_log.json")
    parser.add_argument("--attr-interval", type=int, default=50_000,
                        help="Run attribution every N env steps")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--shaping-weight", type=float, default=0.5)
    parser.add_argument("--decay-rate", type=float, default=0.95)
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # PPO trainer (tensor backend)
    ppo = PPOTrainer(
        lr=args.lr,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        device=device,
        backend="tensor",
    )

    # Attention discovery
    att_config = AttentionConfig(
        attribution_interval=args.attr_interval,
        top_k=args.top_k,
        initial_shaping_weight=args.shaping_weight,
        decay_rate=args.decay_rate,
    )
    att = AttentionTrainer(
        value_network=ppo.agent.network,
        config=att_config,
        device=device,
    )

    # Environment
    vec_env = TensorVecEnv(
        num_envs=args.num_envs,
        config=DEFAULTS,
        device=device,
    )

    steps_per_rollout = args.rollout_steps * args.num_envs * 3
    n_rollouts = max(args.timesteps // steps_per_rollout, 1)

    logger.info("Attention Discovery: device=%s, envs=%d, rollouts=%d, "
                "attr_interval=%d",
                device, args.num_envs, n_rollouts, args.attr_interval)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_interval = max(n_rollouts // 10, 1)  # ~10 checkpoints total

    for rollout_idx in range(n_rollouts):
        # Collect rollout with shaped reward injected
        buffer = ppo.collect_rollout_vec(vec_env)

        # Feed observations to the attention trainer
        obs_tensor = torch.as_tensor(
            buffer.observations[: buffer.pos],
            dtype=torch.float32,
            device=device,
        )
        att.on_step(obs_tensor)

        # If shaping features exist, augment rewards in the buffer
        if att.crystallized_features:
            shaped = att.compute_shaped_reward(obs_tensor)
            buffer.rewards[: buffer.pos] += shaped.cpu().numpy()
            # Recompute GAE with augmented rewards
            buffer.compute_gae(gamma=ppo.gamma, gae_lambda=ppo.gae_lambda)

        # PPO update
        metrics = ppo.train_step(buffer)

        if (rollout_idx + 1) % 10 == 0:
            n_feat = len(att.crystallized_features)
            feat_str = ", ".join(
                f"{f.feature_name}({f.sign:+.0f})"
                for f in att.crystallized_features[:4]
            )
            logger.info(
                "Rollout %d/%d: ploss=%.4f vloss=%.4f ent=%.4f "
                "| %d features: [%s]",
                rollout_idx + 1, n_rollouts,
                metrics['policy_loss'],
                metrics['value_loss'],
                metrics['entropy'],
                n_feat, feat_str,
            )

        # Periodic checkpoint
        if (rollout_idx + 1) % checkpoint_interval == 0:
            step_count = (rollout_idx + 1) * steps_per_rollout
            cp = ckpt_dir / f"agent_step_{step_count:07d}.pt"
            ppo.agent.save(cp)
            logger.info("Checkpoint: %s", cp)

    vec_env.close()

    # Save
    path = Path(args.save_path)
    ppo.agent.save(path)
    logger.info("Saved agent to %s", path)

    log_path = Path(args.log_path)
    log_path.write_text(json.dumps(att.discovery_log, indent=2))
    logger.info("Saved discovery log to %s (%d attribution cycles)",
                log_path, len(att.discovery_log))


if __name__ == "__main__":
    main()
