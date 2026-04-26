"""Feature Attention Discovery: zero-knowledge reward shaping via gradient attribution.

Trains a PPO agent with sparse +1/-1 win/loss, then periodically computes
gradient attribution on the value network to discover which observation
features predict winning.  The top-K features are crystallized into explicit
reward components that supplement the sparse signal, accelerating learning.

Usage:
    trainer = AttentionTrainer(ppo_trainer)
    # inside rollout loop:
    trainer.on_step(observations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Feature name table (matches observations.py / tensor_observations.py)
# ---------------------------------------------------------------------------
_PER_PENGUIN = [
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "dist_from_center",
    "dist_to_edge",
    "speed",
    "heading",
    "alive",
    "rel_position_x",
    "rel_position_y",
    "rel_velocity_x",
    "rel_velocity_y",
    "dist_to_ego",
]
_PENGUIN_LABELS = ["ego", "ally1", "ally2", "enemy1", "enemy2", "enemy3"]
_GLOBAL = [
    "team_a_alive",
    "team_b_alive",
    "ego_team_alive",
    "opp_team_alive",
    "timestep",
]

# Binary/non-actionable features to exclude from crystallization.
# "alive" is always highly attributed but provides no useful gradient.
_EXCLUDED_FEATURE_NAMES = {"alive", "team_a_alive", "team_b_alive",
                           "ego_team_alive", "opp_team_alive"}


def get_feature_names(obs_dim: int = 89) -> list[str]:
    """Return human-readable names for each of the 89 observation dims."""
    names: list[str] = []
    for label in _PENGUIN_LABELS:
        for feat in _PER_PENGUIN:
            names.append(f"{label}.{feat}")
    for g in _GLOBAL:
        names.append(g)
    assert len(names) == obs_dim, f"Expected {obs_dim} names, got {len(names)}"
    return names


def _is_excluded(name: str) -> bool:
    """True if feature should be excluded from crystallization."""
    # Check against the base feature name (after the dot)
    base = name.split(".")[-1] if "." in name else name
    return base in _EXCLUDED_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttentionConfig:
    """Hyper-parameters for the attention discovery system."""

    attribution_interval: int = 50_000
    """Run attribution every this many env steps."""

    top_k: int = 8
    """Number of features to crystallize."""

    initial_shaping_weight: float = 0.5
    """Total weight budget for shaping reward (sum of |w_i|)."""

    decay_rate: float = 0.95
    """Multiply shaping weights by this each attribution cycle."""

    min_weight: float = 0.01
    """Drop crystallized features whose weight falls below this."""

    buffer_size: int = 10_000
    """Max recent observations to keep for attribution."""


# ---------------------------------------------------------------------------
# Gradient Attributor
# ---------------------------------------------------------------------------

class GradientAttributor:
    """Gradient-based feature attribution on the value (critic) network.

    For each observation feature, computes how much the value estimate
    changes with respect to that feature, averaged over a buffer of states.
    """

    def compute_attribution(
        self,
        value_network: nn.Module,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-feature absolute attribution scores.

        Args:
            value_network: An ``ActorCritic`` (or any module whose forward
                returns ``(_, _, value)`` with ``value`` shape ``(B, 1)``).
            observations: Tensor of shape ``(N, obs_dim)`` with recent states.

        Returns:
            Tensor of shape ``(obs_dim,)`` with mean |dV/d(obs_i)|.
        """
        obs = observations.detach().clone().requires_grad_(True)

        # Forward through the full ActorCritic; value is the 3rd output.
        out = value_network(obs)
        values = out[2]  # (N, 1)

        values.sum().backward()

        assert obs.grad is not None
        # Mean absolute gradient per feature.
        attribution = obs.grad.abs().mean(dim=0)  # (obs_dim,)
        return attribution.detach()

    def compute_signed_attribution(
        self,
        value_network: nn.Module,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attribution scores AND per-feature sign.

        The sign indicates whether increasing the feature increases (+1)
        or decreases (-1) the expected value.

        Returns:
            (attribution, signs) each of shape ``(obs_dim,)``.
        """
        obs = observations.detach().clone().requires_grad_(True)

        out = value_network(obs)
        values = out[2]  # (N, 1)

        values.sum().backward()

        assert obs.grad is not None
        grad = obs.grad  # (N, obs_dim)
        attribution = grad.abs().mean(dim=0)
        signs = grad.mean(dim=0).sign()
        return attribution.detach(), signs.detach()


# ---------------------------------------------------------------------------
# Crystallized Feature
# ---------------------------------------------------------------------------

@dataclass
class CrystallizedFeature:
    """A single observation feature turned into an explicit reward component."""

    feature_index: int
    feature_name: str
    weight: float
    sign: float
    attribution_score: float
    discovered_at_step: int


# ---------------------------------------------------------------------------
# Attention Reward Shaper
# ---------------------------------------------------------------------------

class AttentionRewardShaper:
    """Manages crystallized features and computes shaped rewards.

    Call ``update()`` after each attribution cycle to refresh the feature
    list, then ``compute_reward()`` every step to augment sparse rewards.
    """

    def __init__(self, config: AttentionConfig | None = None) -> None:
        self.config = config or AttentionConfig()
        self.features: list[CrystallizedFeature] = []
        self._discovery_log: list[dict] = []

    # ---- update -----------------------------------------------------------

    def update(
        self,
        attributions: torch.Tensor,
        signs: torch.Tensor,
        step: int,
        feature_names: list[str],
    ) -> None:
        """Re-crystallize features from fresh attribution scores.

        1. Decay existing weights.
        2. Select top-K non-excluded features by attribution.
        3. Normalise weights to sum to ``initial_shaping_weight``.
        4. Log the discovery event.
        """
        # Decay existing
        for f in self.features:
            f.weight *= self.config.decay_rate

        # Prune tiny weights
        self.features = [f for f in self.features
                         if abs(f.weight) >= self.config.min_weight]

        # Build candidate list (exclude binary/non-actionable features)
        attr_np = attributions.cpu()
        signs_np = signs.cpu()

        candidates: list[tuple[int, float, float]] = []  # (idx, score, sign)
        for i in range(len(attr_np)):
            if _is_excluded(feature_names[i]):
                continue
            candidates.append((i, float(attr_np[i]), float(signs_np[i])))

        # Sort by attribution score descending
        candidates.sort(key=lambda c: c[1], reverse=True)
        top = candidates[: self.config.top_k]

        if not top:
            return

        # Normalise weights so sum(|w|) = initial_shaping_weight
        total_attr = sum(c[1] for c in top)
        if total_attr < 1e-12:
            return

        new_features: list[CrystallizedFeature] = []
        for idx, score, sign in top:
            w = (score / total_attr) * self.config.initial_shaping_weight
            # Preserve any existing discovered_at_step for features we
            # already know about.
            existing_step = step
            for f in self.features:
                if f.feature_index == idx:
                    existing_step = f.discovered_at_step
                    break
            new_features.append(CrystallizedFeature(
                feature_index=idx,
                feature_name=feature_names[idx],
                weight=w,
                sign=sign,
                attribution_score=score,
                discovered_at_step=existing_step,
            ))

        self.features = new_features

        # Log
        self._discovery_log.append({
            "step": step,
            "features": [
                {
                    "index": f.feature_index,
                    "name": f.feature_name,
                    "weight": f.weight,
                    "sign": f.sign,
                    "attribution": f.attribution_score,
                }
                for f in self.features
            ],
        })

    # ---- reward computation -----------------------------------------------

    def compute_reward(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute shaped reward for a batch of observations.

        Args:
            observations: ``(B, obs_dim)`` or ``(B, N, obs_dim)``.

        Returns:
            Shaped reward tensor.  Same leading dims as *observations*
            minus the last (obs) dimension.  E.g. ``(B,)`` or ``(B, N)``.
        """
        if not self.features:
            return torch.zeros(
                observations.shape[:-1],
                device=observations.device,
                dtype=observations.dtype,
            )

        # Gather indices, weights, signs
        indices = torch.tensor(
            [f.feature_index for f in self.features],
            device=observations.device,
            dtype=torch.long,
        )
        weights = torch.tensor(
            [f.weight * f.sign for f in self.features],
            device=observations.device,
            dtype=observations.dtype,
        )

        # Select features: (..., K)
        selected = observations[..., indices]
        # Weighted sum over K -> (...)
        reward = (selected * weights).sum(dim=-1)
        return reward

    # ---- logging ----------------------------------------------------------

    def get_discovery_log(self) -> list[dict]:
        """Return the full discovery log (one entry per attribution cycle)."""
        return list(self._discovery_log)


# ---------------------------------------------------------------------------
# Attention Trainer (hooks into PPO)
# ---------------------------------------------------------------------------

class AttentionTrainer:
    """Orchestrates periodic attribution and reward shaping.

    Sits alongside a ``PPOTrainer``: call ``on_step`` after each env step
    with the current observations.  The trainer maintains an observation
    buffer, runs attribution at the configured interval, updates the
    reward shaper, and exposes ``compute_shaped_reward`` for augmenting
    the sparse signal.
    """

    def __init__(
        self,
        value_network: nn.Module,
        config: AttentionConfig | None = None,
        obs_dim: int = 89,
        device: str = "cpu",
    ) -> None:
        self.config = config or AttentionConfig()
        self.value_network = value_network
        self.device = torch.device(device)
        self.obs_dim = obs_dim

        self.attributor = GradientAttributor()
        self.shaper = AttentionRewardShaper(self.config)
        self.feature_names = get_feature_names(obs_dim)

        # Rolling observation buffer for attribution
        self._obs_buffer = torch.zeros(
            self.config.buffer_size, obs_dim,
            device=self.device, dtype=torch.float32,
        )
        self._buf_pos: int = 0
        self._buf_full: bool = False

        self._total_steps: int = 0

    # ---- public API -------------------------------------------------------

    def on_step(self, observations: torch.Tensor) -> None:
        """Record observations and trigger attribution if due.

        Args:
            observations: ``(B, obs_dim)`` — flat batch of observations
                from recent env steps.
        """
        obs = observations.detach()
        if obs.device != self.device:
            obs = obs.to(self.device)

        n = obs.shape[0]
        buf_size = self.config.buffer_size

        if n >= buf_size:
            # More data than buffer — just keep the last buf_size rows.
            self._obs_buffer[:] = obs[-buf_size:]
            self._buf_pos = 0
            self._buf_full = True
        else:
            end = self._buf_pos + n
            if end <= buf_size:
                self._obs_buffer[self._buf_pos:end] = obs
            else:
                # Wrap around
                first = buf_size - self._buf_pos
                self._obs_buffer[self._buf_pos:] = obs[:first]
                self._obs_buffer[:n - first] = obs[first:]
            self._buf_pos = end % buf_size
            if end >= buf_size:
                self._buf_full = True

        self._total_steps += n

        if (self._total_steps >= self.config.attribution_interval
                and self._total_steps % self.config.attribution_interval
                < n + self.config.attribution_interval // 10):
            # Close enough to the interval boundary — run attribution.
            self._run_attribution()

    def force_attribution(self) -> None:
        """Force an attribution cycle regardless of step count."""
        self._run_attribution()

    def compute_shaped_reward(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute the current shaping reward for a batch of observations."""
        return self.shaper.compute_reward(observations)

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def discovery_log(self) -> list[dict]:
        return self.shaper.get_discovery_log()

    @property
    def crystallized_features(self) -> list[CrystallizedFeature]:
        return list(self.shaper.features)

    # ---- internals --------------------------------------------------------

    def _run_attribution(self) -> None:
        """Run gradient attribution and update the reward shaper."""
        valid = self.config.buffer_size if self._buf_full else self._buf_pos
        if valid < 64:
            # Not enough data yet.
            return

        obs_sample = self._obs_buffer[:valid]

        attr, signs = self.attributor.compute_signed_attribution(
            self.value_network, obs_sample,
        )

        self.shaper.update(attr, signs, self._total_steps, self.feature_names)
