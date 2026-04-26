"""CLI for the LLM Reward Architect training loop.

Runs iterative reward discovery: LLM generates reward functions,
PPO trains with them, statistics feed back to the LLM.

Saves per-iteration artifacts:
    iteration_NNN/
        prompt.txt
        response.txt
        reward_function.py
        metrics.json
        full_record.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from knockout.core.config import GameConfig, DEFAULTS
from knockout.reward.llm_architect import LLMArchitectConfig, LLMArchitectTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Reward Architect: zero-knowledge reward discovery"
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of outer-loop iterations (default: 10)",
    )
    parser.add_argument(
        "--steps-per-iteration", type=int, default=1_000_000,
        help="PPO training steps per iteration (default: 1M)",
    )
    parser.add_argument(
        "--episodes-per-eval", type=int, default=500,
        help="Episodes for statistics collection (default: 500)",
    )
    parser.add_argument(
        "--num-candidates", type=int, default=3,
        help="Reward function candidates per iteration (default: 3)",
    )
    parser.add_argument(
        "--candidate-eval-steps", type=int, default=200_000,
        help="Quick-eval training steps per candidate (default: 200K)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=64,
        help="Parallel environments for training (default: 64)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Anthropic model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="LLM sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/llm_architect",
        help="Output directory for artifacts (default: runs/llm_architect)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: cpu, cuda, or auto (default: auto)",
    )
    parser.add_argument(
        "--fallback-reward", action="store_true",
        help="Use hardcoded reward templates instead of LLM API calls "
             "(no ANTHROPIC_API_KEY required)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logging.info("LLM Reward Architect starting")
    logging.info("  model=%s, device=%s, num_envs=%d", args.model, device, args.num_envs)
    logging.info("  iterations=%d, steps_per_iter=%d", args.iterations, args.steps_per_iteration)
    logging.info("  output_dir=%s", args.output_dir)
    logging.info("  fallback_reward=%s", args.fallback_reward)

    config = LLMArchitectConfig(
        num_iterations=args.iterations,
        episodes_per_eval=args.episodes_per_eval,
        steps_per_iteration=args.steps_per_iteration,
        llm_model=args.model,
        num_candidates=args.num_candidates,
        candidate_eval_steps=args.candidate_eval_steps,
        temperature=args.temperature,
        output_dir=args.output_dir,
        device=device,
        num_envs=args.num_envs,
    )

    trainer = LLMArchitectTrainer(
        architect_config=config,
        fallback_reward=args.fallback_reward,
    )
    history = trainer.run()

    # Save summary
    output_path = Path(args.output_dir)
    summary_path = output_path / "summary.json"
    summary = []
    for record in history:
        summary.append({
            "iteration": record["iteration"],
            "win_rate": record.get("game_stats", {}).get("win_rate", 0.0),
            "avg_episode_length": record.get("game_stats", {}).get("avg_episode_length", 0.0),
            "policy_loss": record.get("metrics", {}).get("policy_loss", 0.0),
        })

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Summary saved to %s", summary_path)
    logging.info("Done. %d iterations completed.", len(history))


if __name__ == "__main__":
    main()
