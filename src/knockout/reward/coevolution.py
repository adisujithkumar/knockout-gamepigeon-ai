"""Reward co-evolution: re-discover reward features as opponents get smarter.

Normal approach: discover reward features once (vs random), then train forever.
Co-evolution: RE-DISCOVER features periodically using SELF-PLAY games.
What matters for winning CHANGES as opponents get smarter:

  - vs random:        "enemy near edge" matters (they wander there)
  - vs intermediate:  "closing speed" matters (they don't wander, must push)
  - vs advanced:      "dodge prediction" matters (they actively evade)

This module tracks these shifts and produces a research artifact: the
evolution timeline showing what the agent discovered at each skill level.

Integration with train_self_play.py
------------------------------------
Replace the ad-hoc contrastive section (lines ~502-635) with::

    # --- At setup (after contrastive_cfg, analyzer, shaper, collector) ---
    from knockout.reward.coevolution import CoevolutionConfig, RewardCoevolution

    coev_config = CoevolutionConfig(
        discovery_interval=5,
        min_games_for_analysis=500,
        feature_momentum=0.7,
        reward_scale=0.5,
    )
    coev = RewardCoevolution(coev_config)

    # --- Replace the "Contrastive analysis (every 5 iterations)" block ---
    # Old code (lines 614-635):
    #   if (iteration + 1) % 5 == 0 and iteration > 0:
    #       ... analyzer.analyze ... shaper.update ...
    #
    # New code:
    if (iteration + 1) % coev.config.discovery_interval == 0 and iteration > 0:
        data = collector.collect(env=env, policy=current_net,
                                 num_episodes=coev.config.min_games_for_analysis,
                                 device=device_str)
        if len(data["observations"]) > 0:
            report = coev.analyze_and_update(data, current_elo, total_steps)
            features_discovered = report["num_active_features"]

    # --- In collect_selfplay_rollout, pass coev instead of shaper ---
    # Change: shaper=shaper if shaper.num_features > 0 else None
    # To:     shaper=coev if coev.num_features > 0 else None
    # (RewardCoevolution.compute_reward has the same interface)

    # --- At end of training, save the evolution timeline ---
    coev.save(output_dir / "coevolution_state.json")
    timeline = coev.get_evolution_timeline()
    with open(output_dir / "evolution_timeline.json", "w") as f:
        json.dump(timeline, f, indent=2)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from knockout.reward.contrastive import (
    ContrastiveAnalyzer,
    ContrastiveConfig,
    ContrastiveRewardShaper,
    DiscoveryResult,
    FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

# Number of observation features (89 dims).
_N_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CoevolutionConfig:
    """Configuration for reward co-evolution."""

    discovery_interval: int = 5          # re-discover every N iterations
    min_games_for_analysis: int = 500    # minimum self-play games before analysis
    feature_momentum: float = 0.7        # EMA blend: keep 70% old, 30% new
    reward_scale: float = 0.5            # total magnitude of shaped reward
    min_effect_size: float = 0.3         # Cohen's d threshold for significance
    max_reward_features: int = 10        # cap on active features
    log_dir: str = "coevolution_log"


# --------------------------------------------------------------------------
# Single snapshot of discovered features at a skill level
# --------------------------------------------------------------------------

@dataclass
class EvolutionSnapshot:
    """What the agent discovered at a particular skill level."""

    step: int
    elo: float
    timestamp: float
    feature_indices: list[int]
    feature_names: list[str]
    effect_sizes: list[float]
    blended_weights: dict[int, float]   # idx -> EMA-blended weight
    feature_shift: float                # L2 distance from previous weights
    new_features: list[str]             # features that appeared this cycle
    dropped_features: list[str]         # features that disappeared
    stable_features: list[str]          # features present in both cycles
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "elo": self.elo,
            "timestamp": self.timestamp,
            "feature_indices": self.feature_indices,
            "feature_names": self.feature_names,
            "effect_sizes": self.effect_sizes,
            "blended_weights": {str(k): v for k, v in self.blended_weights.items()},
            "feature_shift": self.feature_shift,
            "new_features": self.new_features,
            "dropped_features": self.dropped_features,
            "stable_features": self.stable_features,
            "note": self.note,
        }


# --------------------------------------------------------------------------
# Core co-evolution tracker
# --------------------------------------------------------------------------

class RewardCoevolution:
    """Tracks how reward features evolve as opponents get smarter.

    Key insight: features that matter vs random != features that matter
    vs skilled opponents. This class:
      1. Runs ContrastiveAnalyzer on self-play trajectory data
      2. EMA-blends new discoveries with previous weights
      3. Tracks feature lifetimes and shifts over the training run
      4. Produces a JSON-serializable evolution timeline for research
    """

    def __init__(self, config: CoevolutionConfig) -> None:
        self.config = config
        self.history: list[EvolutionSnapshot] = []
        self.current_weights: dict[int, float] = {}  # feature_idx -> weight
        self._feature_first_seen: dict[int, int] = {}  # idx -> step first seen
        self._feature_last_seen: dict[int, int] = {}   # idx -> step last seen

        # Internal analyzer and shaper
        self._contrastive_cfg = ContrastiveConfig(
            min_effect_size=config.min_effect_size,
            max_reward_features=config.max_reward_features,
            reward_scale=config.reward_scale,
        )
        self._analyzer = ContrastiveAnalyzer(self._contrastive_cfg)
        self._shaper = ContrastiveRewardShaper(self._contrastive_cfg)

    # ------------------------------------------------------------------
    # Main entry point: analyze trajectories and update weights
    # ------------------------------------------------------------------

    def analyze_and_update(
        self,
        trajectory_data: dict[str, np.ndarray],
        current_elo: float,
        current_step: int,
    ) -> dict[str, Any]:
        """Run contrastive analysis on self-play games and update rewards.

        Args:
            trajectory_data: Dict with 'observations' (N, 89) and
                'outcomes' (N,) arrays from TrajectoryCollector.
            current_elo: Current ELO rating of the learning agent.
            current_step: Current training step count.

        Returns:
            Discovery report dict with keys:
                new_features, dropped_features, stable_features,
                elo_at_discovery, feature_shift, num_active_features,
                discovery_result (the raw DiscoveryResult.to_dict()).
        """
        observations = trajectory_data["observations"]
        outcomes = trajectory_data["outcomes"]

        n_steps = len(observations)
        if n_steps < 4:
            logger.warning(
                "Too few trajectory steps (%d) for contrastive analysis. "
                "Skipping co-evolution update.", n_steps,
            )
            return self._empty_report(current_elo, current_step)

        # 1. Run contrastive analysis
        result = self._analyzer.analyze(observations, outcomes)

        if result.feature_indices:
            logger.info(
                "Co-evolution step %d (elo=%.0f): discovered %d features",
                current_step, current_elo, len(result.feature_indices),
            )
            logger.info(result.summary())

        # 2. Build new raw weights from discovery
        new_raw: dict[int, float] = {}
        for idx, es in zip(result.feature_indices, result.effect_sizes):
            new_raw[idx] = es

        # 3. EMA blend with previous weights
        previous_set = set(self.current_weights.keys())
        new_set = set(new_raw.keys())

        blended: dict[int, float] = {}
        momentum = self.config.feature_momentum

        # Features present in both old and new: EMA blend
        for idx in previous_set | new_set:
            old_w = self.current_weights.get(idx, 0.0)
            new_w = new_raw.get(idx, 0.0)
            blended[idx] = momentum * old_w + (1.0 - momentum) * new_w

        # Prune features with negligible weight
        blended = {
            idx: w for idx, w in blended.items() if abs(w) > 1e-4
        }

        # 4. Compute feature shift (L2 distance between old and new weight vectors)
        feature_shift = self._compute_shift(self.current_weights, blended)

        # 5. Classify features
        active_previous = set(self.current_weights.keys())
        active_blended = set(blended.keys())

        new_features = sorted(active_blended - active_previous)
        dropped_features = sorted(active_previous - active_blended)
        stable_features = sorted(active_previous & active_blended)

        new_feature_names = [FEATURE_NAMES[i] for i in new_features]
        dropped_feature_names = [FEATURE_NAMES[i] for i in dropped_features]
        stable_feature_names = [FEATURE_NAMES[i] for i in stable_features]

        # 6. Update lifetime tracking
        for idx in new_features:
            self._feature_first_seen[idx] = current_step
        for idx in active_blended:
            self._feature_last_seen[idx] = current_step

        # 7. Generate note (human-readable summary of what changed)
        note = self._generate_note(
            new_feature_names, dropped_feature_names,
            stable_feature_names, current_elo,
        )

        # 8. Record snapshot
        snapshot = EvolutionSnapshot(
            step=current_step,
            elo=current_elo,
            timestamp=time.time(),
            feature_indices=sorted(active_blended),
            feature_names=[FEATURE_NAMES[i] for i in sorted(active_blended)],
            effect_sizes=[blended[i] for i in sorted(active_blended)],
            blended_weights=dict(blended),
            feature_shift=feature_shift,
            new_features=new_feature_names,
            dropped_features=dropped_feature_names,
            stable_features=stable_feature_names,
            note=note,
        )
        self.history.append(snapshot)

        # 9. Update current weights and shaper
        self.current_weights = blended
        self._update_shaper()

        # 10. Log the shift
        if new_feature_names:
            logger.info("  New features: %s", new_feature_names)
        if dropped_feature_names:
            logger.info("  Dropped features: %s", dropped_feature_names)
        logger.info(
            "  Feature shift: %.4f, active features: %d, note: %s",
            feature_shift, len(blended), note,
        )

        return {
            "new_features": new_feature_names,
            "dropped_features": dropped_feature_names,
            "stable_features": stable_feature_names,
            "elo_at_discovery": current_elo,
            "feature_shift": feature_shift,
            "num_active_features": len(blended),
            "discovery_result": result.to_dict(),
            "note": note,
        }

    # ------------------------------------------------------------------
    # Reward computation (delegates to internal shaper)
    # ------------------------------------------------------------------

    def compute_reward(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute shaped reward using current co-evolved weights.

        Args:
            observations: (B, 3, 89) or (B, 89) tensor.

        Returns:
            (B, 3) or (B,) shaped reward tensor.
        """
        return self._shaper.compute_reward(observations)

    @property
    def num_features(self) -> int:
        """Number of currently active reward features."""
        return self._shaper.num_features

    # ------------------------------------------------------------------
    # Evolution timeline (the key research artifact)
    # ------------------------------------------------------------------

    def get_evolution_timeline(self) -> list[dict[str, Any]]:
        """Return the full history of feature evolution.

        This is the key research artifact: a timeline showing what features
        the agent discovered at each skill level and how they changed.

        Example output::

            [
                {
                    "step": 0, "elo": 1000,
                    "features": ["ego.dist_to_edge", "enemy1.alive"],
                    "note": "basic survival"
                },
                {
                    "step": 200000, "elo": 1050,
                    "features": ["enemy1.dist_to_edge", "ego.speed"],
                    "note": "aggression emerges: +enemy1.dist_to_edge, +ego.speed"
                },
            ]
        """
        timeline = []
        for snap in self.history:
            entry: dict[str, Any] = {
                "step": snap.step,
                "elo": snap.elo,
                "features": list(snap.feature_names),
                "effect_sizes": list(snap.effect_sizes),
                "feature_shift": snap.feature_shift,
                "new_features": snap.new_features,
                "dropped_features": snap.dropped_features,
                "stable_features": snap.stable_features,
                "note": snap.note,
            }
            # Include feature lifetimes for features active at this snapshot
            lifetimes = {}
            for idx in snap.feature_indices:
                first = self._feature_first_seen.get(idx, snap.step)
                lifetimes[FEATURE_NAMES[idx]] = snap.step - first
            entry["feature_lifetimes"] = lifetimes
            timeline.append(entry)
        return timeline

    def get_feature_lifetimes(self) -> dict[str, dict[str, int]]:
        """Return lifetime info for every feature ever discovered.

        Returns:
            Dict mapping feature_name -> {first_seen, last_seen, duration}.
        """
        result = {}
        for idx in self._feature_first_seen:
            name = FEATURE_NAMES[idx]
            first = self._feature_first_seen[idx]
            last = self._feature_last_seen.get(idx, first)
            result[name] = {
                "first_seen": first,
                "last_seen": last,
                "duration": last - first,
            }
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save full co-evolution state (weights + history) to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "current_weights": {str(k): v for k, v in self.current_weights.items()},
            "feature_first_seen": {str(k): v for k, v in self._feature_first_seen.items()},
            "feature_last_seen": {str(k): v for k, v in self._feature_last_seen.items()},
            "history": [snap.to_dict() for snap in self.history],
            "config": {
                "discovery_interval": self.config.discovery_interval,
                "min_games_for_analysis": self.config.min_games_for_analysis,
                "feature_momentum": self.config.feature_momentum,
                "reward_scale": self.config.reward_scale,
                "min_effect_size": self.config.min_effect_size,
                "max_reward_features": self.config.max_reward_features,
            },
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Co-evolution state saved to %s", path)

    def load(self, path: Path) -> None:
        """Load co-evolution state for resume."""
        path = Path(path)
        with open(path) as f:
            state = json.load(f)

        # Restore weights (keys are stringified ints in JSON)
        self.current_weights = {int(k): v for k, v in state["current_weights"].items()}
        self._feature_first_seen = {int(k): v for k, v in state["feature_first_seen"].items()}
        self._feature_last_seen = {int(k): v for k, v in state["feature_last_seen"].items()}

        # Restore history
        self.history = []
        for snap_dict in state["history"]:
            blended = {int(k): v for k, v in snap_dict["blended_weights"].items()}
            snap = EvolutionSnapshot(
                step=snap_dict["step"],
                elo=snap_dict["elo"],
                timestamp=snap_dict["timestamp"],
                feature_indices=snap_dict["feature_indices"],
                feature_names=snap_dict["feature_names"],
                effect_sizes=snap_dict["effect_sizes"],
                blended_weights=blended,
                feature_shift=snap_dict["feature_shift"],
                new_features=snap_dict["new_features"],
                dropped_features=snap_dict["dropped_features"],
                stable_features=snap_dict["stable_features"],
                note=snap_dict.get("note", ""),
            )
            self.history.append(snap)

        # Rebuild internal shaper from current weights
        self._update_shaper()
        logger.info(
            "Co-evolution state loaded from %s (%d snapshots, %d active features)",
            path, len(self.history), len(self.current_weights),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_shaper(self) -> None:
        """Rebuild the internal ContrastiveRewardShaper from current_weights."""
        if not self.current_weights:
            self._shaper.update(
                DiscoveryResult([], [], [], []),
            )
            return

        indices = sorted(self.current_weights.keys())
        weights = [self.current_weights[i] for i in indices]
        names = [FEATURE_NAMES[i] for i in indices]
        directions = [float(np.sign(w)) for w in weights]

        result = DiscoveryResult(
            feature_indices=indices,
            effect_sizes=weights,
            directions=directions,
            feature_names=names,
        )
        self._shaper.update(result)

    @staticmethod
    def _compute_shift(
        old_weights: dict[int, float],
        new_weights: dict[int, float],
    ) -> float:
        """Compute L2 distance between two weight dictionaries."""
        all_keys = set(old_weights.keys()) | set(new_weights.keys())
        if not all_keys:
            return 0.0
        sq_sum = 0.0
        for k in all_keys:
            diff = new_weights.get(k, 0.0) - old_weights.get(k, 0.0)
            sq_sum += diff * diff
        return float(np.sqrt(sq_sum))

    @staticmethod
    def _generate_note(
        new_features: list[str],
        dropped_features: list[str],
        stable_features: list[str],
        elo: float,
    ) -> str:
        """Generate a human-readable note for the evolution timeline."""
        parts = []

        if not new_features and not dropped_features and not stable_features:
            return "no significant features"

        if new_features:
            parts.append("+" + ", +".join(new_features))
        if dropped_features:
            parts.append("-" + ", -".join(dropped_features))

        if not parts and stable_features:
            return f"stable ({len(stable_features)} features)"

        return "; ".join(parts)

    def _empty_report(
        self, elo: float, step: int,
    ) -> dict[str, Any]:
        """Return an empty report when analysis cannot run."""
        return {
            "new_features": [],
            "dropped_features": [],
            "stable_features": [],
            "elo_at_discovery": elo,
            "feature_shift": 0.0,
            "num_active_features": len(self.current_weights),
            "discovery_result": {"features": []},
            "note": "insufficient data",
        }
