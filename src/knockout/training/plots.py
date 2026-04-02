"""Training visualization and plotting utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    logs: list[dict],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot training metrics (policy loss, value loss, entropy)."""
    if not logs:
        return

    rollouts = [d.get("rollout", i) for i, d in enumerate(logs)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, key, label in zip(
        axes,
        ["policy_loss", "value_loss", "entropy"],
        ["Policy Loss", "Value Loss", "Entropy"],
    ):
        values = [d.get(key, 0.0) for d in logs]
        ax.plot(rollouts, values)
        ax.set_xlabel("Rollout")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_win_rates(
    logs: list[dict],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot win rate over generations (for self-play logs)."""
    if not logs:
        return

    gens = [d.get("generation", i) for i, d in enumerate(logs)]
    win_rates = [d.get("win_rate", 0.0) for d in logs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, win_rates, marker="o", markersize=3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate vs Generation")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_elo_progression(
    logs: list[dict],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot ELO rating over generations."""
    if not logs:
        return

    gens = [d.get("generation", i) for i, d in enumerate(logs)]
    elos = [d.get("elo", 1000.0) for d in logs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, elos, marker="o", markersize=3, color="green")
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_xlabel("Generation")
    ax.set_ylabel("ELO Rating")
    ax.set_title("ELO Rating Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def save_logs_csv(logs: list[dict], path: str) -> None:
    """Save training logs to CSV."""
    if not logs:
        return

    keys = list(logs[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(logs)
