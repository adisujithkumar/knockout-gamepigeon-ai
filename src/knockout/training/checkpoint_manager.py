"""Checkpoint manager for pause/resume of self-play training.

Saves and restores the complete training state: model weights, optimizer,
pool metadata, ELO ratings, contrastive reward state, RNG seeds, metrics,
and all training hyperparameters.  Designed to work with any training
script -- the ``TrainingState`` dataclass captures everything needed.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training state snapshot
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Complete snapshot of training state for pause/resume."""

    step: int
    iteration: int
    agent_state_dict: dict                      # PyTorch model weights
    optimizer_state_dict: dict                   # Optimizer state (momentum, etc.)
    pool_metadata: list[dict] = field(          # [{path, elo, step, win_rates}, ...]
        default_factory=list
    )
    elo_ratings: dict[str, float] = field(      # agent_name -> elo
        default_factory=dict
    )
    reward_shaper_state: dict = field(          # contrastive discovery state
        default_factory=dict
    )
    self_play_ratio: float = 0.0                # current random vs self-play ratio
    metrics_history: list[dict] = field(        # all logged metrics so far
        default_factory=list
    )
    rng_state: dict = field(                    # numpy + torch RNG states
        default_factory=dict
    )
    config: dict = field(                       # all training hyperparameters
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# RNG capture / restore helpers
# ---------------------------------------------------------------------------

def capture_rng_state() -> dict:
    """Capture current numpy and torch RNG states for reproducibility."""
    state: dict[str, Any] = {}
    state["numpy"] = np.random.get_state()
    state["torch_cpu"] = torch.random.get_rng_state()
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state()
        except RuntimeError:
            pass
    return state


def restore_rng_state(state: dict) -> None:
    """Restore numpy and torch RNG states from a captured snapshot."""
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state(state["torch_cuda"])
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages saving/loading full training state for pause/resume."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._last_state: TrainingState | None = None
        self._last_save_step: int = -1
        self._signal_handlers_installed: bool = False

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def install_signal_handlers(self) -> None:
        """Save state on SIGTERM/SIGINT (Ctrl+C) for graceful shutdown."""
        if self._signal_handlers_installed:
            return

        def handler(signum: int, frame: Any) -> None:
            logger.info("Signal %d received, saving checkpoint...", signum)
            if self._last_state is not None:
                self.save(self._last_state, label="interrupted")
                logger.info("Checkpoint saved. Safe to exit.")
            sys.exit(0)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
        self._signal_handlers_installed = True

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, state: TrainingState, label: str | None = None) -> Path:
        """Save complete training state.

        Saves:
        - training_state.pt  (agent weights, optimizer, RNG, metadata)
        - config.json        (all hyperparameters)
        - metrics.csv        (training history)

        If *label* is provided, saves as ``checkpoint_{label}/``.
        Otherwise saves as ``checkpoint_step_{step:07d}/``.

        Also maintains a ``latest`` symlink for easy resume.
        """
        if label is not None:
            dir_name = f"checkpoint_{label}"
        else:
            dir_name = f"checkpoint_step_{state.step:07d}"

        ckpt_dir = self.checkpoint_dir / dir_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. PyTorch state blob
        torch_blob = {
            "step": state.step,
            "iteration": state.iteration,
            "agent_state_dict": state.agent_state_dict,
            "optimizer_state_dict": state.optimizer_state_dict,
            "pool_metadata": state.pool_metadata,
            "elo_ratings": state.elo_ratings,
            "reward_shaper_state": state.reward_shaper_state,
            "self_play_ratio": state.self_play_ratio,
            "rng_state": state.rng_state,
        }
        torch.save(torch_blob, ckpt_dir / "training_state.pt")

        # 2. Config (JSON, human readable)
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(state.config, f, indent=2, default=str)

        # 3. Metrics CSV
        self._write_metrics_csv(state.metrics_history, ckpt_dir / "metrics.csv")

        # 4. Pool metadata (JSON, human readable)
        if state.pool_metadata:
            with open(ckpt_dir / "pool_metadata.json", "w") as f:
                json.dump(state.pool_metadata, f, indent=2)

        # 5. Timestamp file for listing
        with open(ckpt_dir / "checkpoint_info.json", "w") as f:
            info = {
                "step": state.step,
                "iteration": state.iteration,
                "timestamp": time.time(),
                "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
                "label": label,
            }
            # Add best ELO from elo_ratings if available
            if state.elo_ratings:
                best_name = max(state.elo_ratings, key=state.elo_ratings.get)
                info["best_elo"] = state.elo_ratings[best_name]
                info["best_elo_agent"] = best_name
            json.dump(info, f, indent=2)

        # 6. Update 'latest' symlink (relative path for portability)
        latest_link = self.checkpoint_dir / "latest"
        # Remove existing symlink/file
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        # Use relative path so it works if the whole directory is moved
        try:
            latest_link.symlink_to(dir_name)
        except OSError:
            # Symlinks may fail on some Windows configurations; fall back
            # to a plain text pointer file.
            with open(latest_link, "w") as f:
                f.write(dir_name)

        self._last_state = state
        self._last_save_step = state.step
        logger.info(
            "Checkpoint saved: %s (step=%d, iteration=%d)",
            ckpt_dir, state.step, state.iteration,
        )
        return ckpt_dir

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str | Path | None = None) -> TrainingState:
        """Load training state.

        If *checkpoint_path* is ``None``, loads from ``latest`` symlink.
        If *checkpoint_path* is ``"best"``, loads the checkpoint with
        highest ELO.  Otherwise treats it as a directory name (relative
        to *checkpoint_dir* if not absolute).

        Restores everything: agent weights, optimizer, pool, RNG, metrics.
        """
        ckpt_dir = self._resolve_checkpoint_path(checkpoint_path)

        # 1. PyTorch state blob
        pt_path = ckpt_dir / "training_state.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"No training_state.pt in {ckpt_dir}")
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)

        # 2. Config
        config: dict = {}
        config_path = ckpt_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        # 3. Metrics
        metrics_history: list[dict] = []
        metrics_path = ckpt_dir / "metrics.csv"
        if metrics_path.exists():
            metrics_history = self._read_metrics_csv(metrics_path)

        state = TrainingState(
            step=blob["step"],
            iteration=blob["iteration"],
            agent_state_dict=blob["agent_state_dict"],
            optimizer_state_dict=blob["optimizer_state_dict"],
            pool_metadata=blob.get("pool_metadata", []),
            elo_ratings=blob.get("elo_ratings", {}),
            reward_shaper_state=blob.get("reward_shaper_state", {}),
            self_play_ratio=blob.get("self_play_ratio", 0.0),
            metrics_history=metrics_history,
            rng_state=blob.get("rng_state", {}),
            config=config,
        )

        self._last_state = state
        self._last_save_step = state.step
        logger.info(
            "Checkpoint loaded: %s (step=%d, iteration=%d)",
            ckpt_dir, state.step, state.iteration,
        )
        return state

    # ------------------------------------------------------------------
    # List checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints with metadata.

        Returns a list of dicts sorted by step (ascending):
        ``[{path, step, iteration, timestamp, elo, label}, ...]``
        """
        results: list[dict] = []
        if not self.checkpoint_dir.exists():
            return results

        for entry in sorted(self.checkpoint_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name == "latest" or entry.name.startswith("."):
                continue

            info_path = entry / "checkpoint_info.json"
            if not info_path.exists():
                # Not a valid checkpoint directory
                continue

            with open(info_path) as f:
                info = json.load(f)

            results.append({
                "path": str(entry),
                "name": entry.name,
                "step": info.get("step", 0),
                "iteration": info.get("iteration", 0),
                "timestamp": info.get("timestamp_human", "unknown"),
                "elo": info.get("best_elo"),
                "label": info.get("label"),
            })

        results.sort(key=lambda r: r["step"])
        return results

    # ------------------------------------------------------------------
    # Auto save
    # ------------------------------------------------------------------

    def auto_save(
        self, state: TrainingState, interval_steps: int = 500_000
    ) -> Path | None:
        """Called every step; saves if enough steps have passed.

        Returns the checkpoint path if a save occurred, else ``None``.
        """
        self._last_state = state  # always keep for signal handler

        if state.step - self._last_save_step >= interval_steps:
            return self.save(state)
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(
        self, keep_last_n: int = 5, keep_best: bool = True
    ) -> list[str]:
        """Remove old checkpoints, keeping last N and optionally the best.

        Returns list of removed checkpoint directory paths.
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_last_n:
            return []

        # Identify the best checkpoint by ELO
        best_path: str | None = None
        if keep_best:
            with_elo = [c for c in checkpoints if c.get("elo") is not None]
            if with_elo:
                best = max(with_elo, key=lambda c: c["elo"])
                best_path = best["path"]

        # Keep the last N by step order
        keep_set: set[str] = set()
        for ckpt in checkpoints[-keep_last_n:]:
            keep_set.add(ckpt["path"])
        if best_path is not None:
            keep_set.add(best_path)

        # Also keep any checkpoint labeled "interrupted"
        for ckpt in checkpoints:
            if ckpt.get("label") == "interrupted":
                keep_set.add(ckpt["path"])

        removed: list[str] = []
        for ckpt in checkpoints:
            if ckpt["path"] not in keep_set:
                ckpt_path = Path(ckpt["path"])
                if ckpt_path.exists():
                    import shutil
                    shutil.rmtree(ckpt_path)
                    removed.append(ckpt["path"])
                    logger.info("Removed old checkpoint: %s", ckpt["path"])

        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint_path(
        self, checkpoint_path: str | Path | None
    ) -> Path:
        """Resolve a checkpoint specifier to an absolute directory path."""
        if checkpoint_path is None or str(checkpoint_path) == "":
            # Load from 'latest' symlink or pointer file
            latest_link = self.checkpoint_dir / "latest"
            if latest_link.is_symlink():
                return latest_link.resolve()
            elif latest_link.exists() and latest_link.is_file():
                # Plain text pointer file (Windows fallback)
                target = latest_link.read_text().strip()
                return self.checkpoint_dir / target
            else:
                raise FileNotFoundError(
                    f"No 'latest' symlink or pointer in {self.checkpoint_dir}. "
                    "Specify a checkpoint path explicitly."
                )

        checkpoint_path = str(checkpoint_path)

        if checkpoint_path == "best":
            checkpoints = self.list_checkpoints()
            with_elo = [c for c in checkpoints if c.get("elo") is not None]
            if not with_elo:
                raise FileNotFoundError(
                    "No checkpoints with ELO data found. "
                    "Cannot determine 'best'."
                )
            best = max(with_elo, key=lambda c: c["elo"])
            return Path(best["path"])

        path = Path(checkpoint_path)
        if path.is_absolute():
            return path
        # Treat as relative to checkpoint_dir
        candidate = self.checkpoint_dir / path
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Checkpoint not found: tried {candidate} and {path}"
        )

    @staticmethod
    def _write_metrics_csv(
        metrics: list[dict], path: Path
    ) -> None:
        """Write metrics history to CSV."""
        if not metrics:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Collect all keys across all rows
        all_keys: list[str] = []
        seen: set[str] = set()
        for row in metrics:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=all_keys, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(metrics)

    @staticmethod
    def _read_metrics_csv(path: Path) -> list[dict]:
        """Read metrics history from CSV, converting numeric strings."""
        rows: list[dict] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                converted: dict[str, Any] = {}
                for k, v in row.items():
                    # Try to convert to numeric
                    try:
                        converted[k] = int(v)
                    except (ValueError, TypeError):
                        try:
                            converted[k] = float(v)
                        except (ValueError, TypeError):
                            converted[k] = v
                rows.append(converted)
        return rows
