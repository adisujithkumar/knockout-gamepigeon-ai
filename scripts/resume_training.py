#!/usr/bin/env python3
"""Resume self-play training from a saved checkpoint.

Usage:
    # Resume from latest checkpoint
    python scripts/resume_training.py --checkpoint-dir runs/self_play/checkpoints

    # Resume from specific checkpoint
    python scripts/resume_training.py --checkpoint-dir runs/self_play/checkpoints \\
        --from checkpoint_step_5000000

    # Resume from best ELO checkpoint
    python scripts/resume_training.py --checkpoint-dir runs/self_play/checkpoints \\
        --from best

    # List available checkpoints
    python scripts/resume_training.py --checkpoint-dir runs/self_play/checkpoints --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from knockout.training.checkpoint_manager import CheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume training from a checkpoint or list available checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--from", dest="from_checkpoint", type=str, default=None,
        help=(
            "Checkpoint to resume from: a directory name (e.g. "
            "checkpoint_step_5000000), 'best' for highest ELO, "
            "or omit for latest"
        ),
    )
    parser.add_argument(
        "--list", dest="list_checkpoints", action="store_true",
        help="List available checkpoints and exit",
    )
    args = parser.parse_args()

    manager = CheckpointManager(args.checkpoint_dir)

    # -- List mode --
    if args.list_checkpoints:
        checkpoints = manager.list_checkpoints()
        if not checkpoints:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            sys.exit(0)

        print(f"Available checkpoints in {args.checkpoint_dir}:\n")
        for ckpt in checkpoints:
            elo_str = f"ELO: {ckpt['elo']:.0f}" if ckpt.get("elo") else "ELO: n/a"
            label_str = f" [{ckpt['label']}]" if ckpt.get("label") else ""
            print(
                f"  {ckpt['name']}{label_str}"
                f"  |  Step: {ckpt['step']:,}  |  Iteration: {ckpt['iteration']}"
                f"  |  {elo_str}  |  {ckpt['timestamp']}"
            )
        print(f"\nTotal: {len(checkpoints)} checkpoints")
        sys.exit(0)

    # -- Resume mode --
    try:
        state = manager.load(args.from_checkpoint)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Determine source label
    if args.from_checkpoint is None:
        source = "latest"
    elif args.from_checkpoint == "best":
        source = "best ELO checkpoint"
    else:
        source = args.from_checkpoint

    # Summary
    print(f"Resuming from: {source}")
    print(f"  Step: {state.step:,} | Iteration: {state.iteration}")

    if state.elo_ratings:
        best_name = max(state.elo_ratings, key=state.elo_ratings.get)
        best_elo = state.elo_ratings[best_name]
        print(f"  ELO: {best_elo:.0f} ({best_name})")

    if state.pool_metadata:
        elos = [p.get("elo", 0) for p in state.pool_metadata if p.get("elo")]
        elo_range = f"{min(elos):.0f}-{max(elos):.0f}" if elos else "n/a"
        print(
            f"  Pool: {len(state.pool_metadata)} checkpoints "
            f"(ELO range: {elo_range})"
        )

    shaper_state = state.reward_shaper_state
    if shaper_state.get("feature_names"):
        features = shaper_state["feature_names"]
        names_preview = ", ".join(features[:4])
        if len(features) > 4:
            names_preview += ", ..."
        print(f"  Reward features: {len(features)} ({names_preview})")

    if state.self_play_ratio > 0:
        print(f"  Self-play ratio: {state.self_play_ratio:.2f}")

    if state.metrics_history:
        print(f"  Metrics: {len(state.metrics_history)} data points")

    print("Ready to continue training.")


if __name__ == "__main__":
    main()
