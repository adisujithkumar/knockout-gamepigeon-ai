"""Fully vectorized knockout environment using TensorPhysicsEngine.

Replaces SingleTeamVecEnv for high-throughput training.  All physics
and observation construction run as batched tensor operations — no
Python loops over environments.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import numpy as np

from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.tensor_physics import TensorPhysicsEngine
from knockout.env.tensor_observations import build_observations, OBS_DIM


class TensorVecEnv:
    """Fully vectorized knockout environment using TensorPhysicsEngine.

    Exposes the same Team-A-centric API as SingleTeamVecEnv:

        obs     : (num_envs, 3, 89)   float32
        masks   : (num_envs, 3)        bool
        rewards : (num_envs, 3)        float32
        dones   : (num_envs,)          bool

    Opponent (Team B) actions are sampled uniformly at random using tensor
    operations (no CPU round-trip).
    """

    TEAM_A = [0, 1, 2]
    TEAM_B = [3, 4, 5]
    OBS_DIM = OBS_DIM
    ACTION_DIM = 2

    def __init__(
        self,
        num_envs: int,
        config: GameConfig = DEFAULTS,
        device: str = "cpu",
        max_rounds: int = 1000,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        self.num_envs = num_envs
        self.config = config
        self.device = torch.device(device)
        self.max_rounds = max_rounds

        self.physics = TensorPhysicsEngine(config, device)

        # Mutable state (populated by reset)
        self.positions: torch.Tensor = torch.empty(0)
        self.velocities: torch.Tensor = torch.empty(0)
        self.alive: torch.Tensor = torch.empty(0)
        self.arena_hw: torch.Tensor = torch.empty(0)
        self.round_number: torch.Tensor = torch.empty(0)
        self.step_count: torch.Tensor = torch.empty(0)

        # RNG for opponent actions
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(42)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset all environments.

        Returns:
            obs   : ndarray (num_envs, 3, 89) float32
            masks : ndarray (num_envs, 3) bool
        """
        self.positions, self.velocities, self.alive = self.physics.reset(
            self.num_envs
        )
        self.arena_hw = torch.full(
            (self.num_envs,), self.config.ARENA_HALF_WIDTH,
            device=self.device, dtype=torch.float32,
        )
        self.round_number = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long,
        )
        self.step_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long,
        )

        return self._build_team_a_output()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, team_a_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step all environments.

        Args:
            team_a_actions: ndarray (num_envs, 3, 2) — [angle_deg, power]

        Returns:
            obs     : ndarray (num_envs, 3, 89) float32
            rewards : ndarray (num_envs, 3) float32
            dones   : ndarray (num_envs,) bool
            masks   : ndarray (num_envs, 3) bool
            infos   : list[dict]
        """
        # 1. Build combined actions tensor (B, 6, 2)
        ta = torch.as_tensor(team_a_actions, dtype=torch.float32, device=self.device)

        # Random opponent actions: angle ~ U[0, 360], power ~ U[0, MAX]
        opp_angle = torch.rand(
            self.num_envs, 3, device=self.device, generator=self._rng
        ) * 360.0
        opp_power = torch.rand(
            self.num_envs, 3, device=self.device, generator=self._rng
        ) * self.config.MAX_LAUNCH_FORCE
        tb = torch.stack([opp_angle, opp_power], dim=-1)  # (B, 3, 2)

        actions = torch.cat([ta, tb], dim=1)  # (B, 6, 2)

        # 2. Apply impulses
        self.velocities = self.physics.apply_impulses(
            self.velocities, actions, self.alive
        )

        # 3. Step until settled
        self.positions, self.velocities, self.alive, steps = (
            self.physics.step_until_settled(
                self.positions, self.velocities, self.alive, self.arena_hw
            )
        )
        self.step_count += steps
        self.round_number += 1

        # 4. Shrink check (every SHRINK_INTERVAL rounds)
        shrink_mask = (
            (self.round_number > 0)
            & (self.round_number % self.config.SHRINK_INTERVAL == 0)
        )
        if shrink_mask.any():
            self._apply_shrink(shrink_mask)

        # 5. Check game over
        team_a_alive = self.alive[:, :3].any(dim=-1)  # (B,)
        team_b_alive = self.alive[:, 3:].any(dim=-1)  # (B,)
        game_over = ~team_a_alive | ~team_b_alive  # (B,)

        truncated = self.round_number >= self.max_rounds  # (B,)
        done = game_over | truncated  # (B,)

        # 6. Rewards (sparse: +1 win, -1 loss, 0 draw/ongoing)
        rewards_t = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )
        # Team A wins: B eliminated and A alive
        a_wins = (~team_b_alive) & team_a_alive  # (B,)
        b_wins = (~team_a_alive) & team_b_alive  # (B,)
        rewards_t[a_wins] = 1.0
        rewards_t[b_wins] = -1.0

        # 7. Build infos (vectorised -- avoid per-env .item() calls)
        done_np = done.cpu().numpy()
        a_wins_np = a_wins.cpu().numpy()
        b_wins_np = b_wins.cpu().numpy()
        n_done_approx = int(done_np.sum())

        if n_done_approx == 0:
            infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        else:
            # Batch the .sum() and transfer once
            ta_alive_np = self.alive[:, :3].sum(dim=-1).cpu().numpy()
            tb_alive_np = self.alive[:, 3:].sum(dim=-1).cpu().numpy()
            infos = []
            for i in range(self.num_envs):
                info: dict[str, Any] = {}
                if done_np[i]:
                    if a_wins_np[i]:
                        info["winner"] = 0
                    elif b_wins_np[i]:
                        info["winner"] = 1
                    else:
                        info["winner"] = -1
                    info["team_a_alive"] = int(ta_alive_np[i])
                    info["team_b_alive"] = int(tb_alive_np[i])
                    info["terminal_observation"] = True
                infos.append(info)

        # 8. Build observations (before auto-reset so terminal obs come through)
        obs_tensor = build_observations(
            self.positions, self.velocities, self.alive,
            self.arena_hw, self.round_number, self.step_count, self.config,
        )  # (B, 6, 89)

        # Extract Team A perspective
        obs_a = obs_tensor[:, :3].cpu().numpy()  # (B, 3, 89)
        masks = self.alive[:, :3].cpu().numpy()  # (B, 3)

        # 9. Auto-reset finished environments
        if done.any():
            self._auto_reset(done)

        return (
            obs_a,
            rewards_t.cpu().numpy(),
            done_np,
            masks,
            infos,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_team_a_output(self) -> tuple[np.ndarray, np.ndarray]:
        """Build (obs, masks) from current state for Team A."""
        obs_tensor = build_observations(
            self.positions, self.velocities, self.alive,
            self.arena_hw, self.round_number, self.step_count, self.config,
        )  # (B, 6, 89)
        obs_a = obs_tensor[:, :3].cpu().numpy()
        masks = self.alive[:, :3].cpu().numpy()
        return obs_a, masks

    def _apply_shrink(self, mask: torch.Tensor) -> None:
        """Apply arena shrink to environments where *mask* is True."""
        old_hw = self.arena_hw.clone()
        new_hw = (
            self.arena_hw * self.config.SHRINK_FACTOR
        ).clamp(min=self.config.MIN_ARENA_HALF_WIDTH)

        # Only update masked envs
        self.arena_hw = torch.where(mask, new_hw, self.arena_hw)
        scale = self.arena_hw / old_hw  # (B,)

        # Rescale only envs that shrank
        if mask.any():
            # For simplicity, rescale all but the effect is gated by scale=1
            # for non-shrink envs.
            self.positions, self.velocities = self.physics.rescale_penguins(
                self.positions, self.velocities, self.alive,
                scale, self.arena_hw,
            )

    def _auto_reset(self, done: torch.Tensor) -> None:
        """Reset environments that are done, in-place."""
        n_done = done.sum().item()
        if n_done == 0:
            return

        new_pos, new_vel, new_alive = self.physics.reset(int(n_done))

        self.positions[done] = new_pos
        self.velocities[done] = new_vel
        self.alive[done] = new_alive
        self.arena_hw[done] = self.config.ARENA_HALF_WIDTH
        self.round_number[done] = 0
        self.step_count[done] = 0

    def close(self) -> None:
        """No resources to release, but API-compatible with SingleTeamVecEnv."""
        pass
