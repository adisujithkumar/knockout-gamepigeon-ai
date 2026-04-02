"""Tensor-based observation builder for batched environments.

Mirrors the 89-dimensional observation layout from observations.py
but uses pure tensor operations with no Python loops over the batch.

Layout per agent (89 dims):
    ego(14) + ally1(14) + ally2(14) + enemy1(14) + enemy2(14) + enemy3(14) + global(5)
"""

from __future__ import annotations

import math

import torch

from knockout.core.config import GameConfig, DEFAULTS


FEATURES_PER_PENGUIN = 14
GLOBAL_FEATURES = 5
OBS_DIM = 89  # 14 * 6 + 5

# Max speed for normalisation (matches scalar ObservationBuilder)
_MAX_SPEED = 200.0


def build_observations(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    alive: torch.Tensor,
    arena_hw: torch.Tensor | float,
    round_number: torch.Tensor | int,
    step_count: torch.Tensor | int,
    config: GameConfig = DEFAULTS,
) -> torch.Tensor:
    """Build 89-dim observation for every agent in every env.

    Args:
        positions:     (B, 6, 2)
        velocities:    (B, 6, 2)
        alive:         (B, 6) bool
        arena_hw:      (B,) or scalar — current arena half-width
        round_number:  (B,) or scalar — current round (unused directly but
                       kept for API symmetry; timestep used for feature)
        step_count:    (B,) or scalar — physics step count (for global feature)
        config:        GameConfig

    Returns:
        observations: (B, 6, 89)
    """
    B = positions.shape[0]
    device = positions.device
    initial_hw = config.ARENA_HALF_WIDTH

    # ---- Per-penguin raw features (absolute) ----------------------------
    # All are (B, 6) or (B, 6, 2)

    pos_norm = positions / initial_hw  # normalise by *initial* hw
    vel_norm = velocities / _MAX_SPEED

    pos_x = pos_norm[..., 0].clamp(-1, 1)  # (B, 6)
    pos_y = pos_norm[..., 1].clamp(-1, 1)
    vel_x = vel_norm[..., 0].clamp(-1, 1)
    vel_y = vel_norm[..., 1].clamp(-1, 1)

    # Chebyshev distance from center
    cheb = torch.max(positions[..., 0].abs(), positions[..., 1].abs())  # (B, 6)
    dist_from_center = (cheb / initial_hw).clamp(0, 1)

    # Distance to edge (using current hw for the arena, but normalised
    # by *initial* hw to match the scalar builder which uses config.ARENA_HALF_WIDTH)
    if isinstance(arena_hw, (int, float)):
        _hw = arena_hw
    else:
        _hw = arena_hw.unsqueeze(-1)  # (B, 1)
    dist_to_edge = ((_hw - cheb) / initial_hw).clamp(-1, 1)  # (B, 6)

    speed = torch.sqrt((velocities * velocities).sum(dim=-1) + 1e-12)  # (B, 6)
    speed_norm = (speed / _MAX_SPEED).clamp(0, 1)

    heading = torch.atan2(velocities[..., 1], velocities[..., 0])  # (B, 6)
    heading = torch.where(speed > 1e-6, heading, torch.zeros_like(heading))
    heading_norm = (heading / math.pi).clamp(-1, 1)

    alive_f = alive.float()  # (B, 6)

    # Stack absolute features: (B, 6, 9)  — first 9 of the 14
    abs_feats = torch.stack(
        [pos_x, pos_y, vel_x, vel_y, dist_from_center, dist_to_edge,
         speed_norm, heading_norm, alive_f],
        dim=-1,
    )  # (B, 6, 9)

    # ---- Global features (same for all agents in an env) ----------------
    team_a_alive = alive[:, :3].float().sum(dim=-1)  # (B,)
    team_b_alive = alive[:, 3:].float().sum(dim=-1)  # (B,)

    if isinstance(step_count, int):
        ts = torch.full((B,), min(step_count / 1000.0, 1.0),
                        device=device, dtype=torch.float32)
    else:
        ts = (step_count.float() / 1000.0).clamp(max=1.0)

    # global_feats: (B, 5)  — [team_a/3, team_b/3, ego_team/3, opp_team/3, timestep]
    # ego_team and opp_team depend on the agent, so we build two variants.
    global_base = torch.stack([team_a_alive / 3.0, team_b_alive / 3.0, ts], dim=-1)  # (B, 3)
    # For team A agents (0-2): ego=A, opp=B
    # For team B agents (3-5): ego=B, opp=A
    global_a = torch.cat([
        global_base[:, :2],
        (team_a_alive / 3.0).unsqueeze(-1),
        (team_b_alive / 3.0).unsqueeze(-1),
        global_base[:, 2:3],
    ], dim=-1)  # (B, 5)
    global_b = torch.cat([
        global_base[:, :2],
        (team_b_alive / 3.0).unsqueeze(-1),
        (team_a_alive / 3.0).unsqueeze(-1),
        global_base[:, 2:3],
    ], dim=-1)  # (B, 5)

    # ---- Build per-agent observations -----------------------------------
    # For each ego agent we need:
    #   ego(14) + sorted-allies(2*14) + sorted-enemies(3*14) + global(5)
    #
    # Relative features (5 per other penguin):
    #   rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, dist_to_ego
    #
    # For ego itself, relative features are all zero.

    obs = torch.zeros(B, 6, OBS_DIM, device=device, dtype=torch.float32)

    for ego_idx in range(6):
        ego_alive = alive[:, ego_idx]  # (B,) bool

        # --- ego penguin features (14) ---
        ego_abs = abs_feats[:, ego_idx]  # (B, 9)
        ego_rel = torch.zeros(B, 5, device=device, dtype=torch.float32)
        ego_14 = torch.cat([ego_abs, ego_rel], dim=-1)  # (B, 14)
        # Zero out if ego is dead
        ego_14 = ego_14 * ego_alive.unsqueeze(-1).float()

        # --- Relative features of all other penguins to this ego ---
        ego_pos = positions[:, ego_idx].unsqueeze(1)  # (B, 1, 2)
        ego_vel = velocities[:, ego_idx].unsqueeze(1)  # (B, 1, 2)

        rel_pos = positions - ego_pos  # (B, 6, 2)
        rel_vel = velocities - ego_vel  # (B, 6, 2)

        rel_pos_norm_x = (rel_pos[..., 0] / (2 * initial_hw)).clamp(-1, 1)  # (B, 6)
        rel_pos_norm_y = (rel_pos[..., 1] / (2 * initial_hw)).clamp(-1, 1)
        rel_vel_norm_x = (rel_vel[..., 0] / _MAX_SPEED).clamp(-1, 1)
        rel_vel_norm_y = (rel_vel[..., 1] / _MAX_SPEED).clamp(-1, 1)
        dist_to_ego = torch.sqrt((rel_pos * rel_pos).sum(dim=-1) + 1e-12)  # (B, 6)
        dist_to_ego_norm = (dist_to_ego / (2 * initial_hw)).clamp(0, 1)

        rel_feats = torch.stack(
            [rel_pos_norm_x, rel_pos_norm_y, rel_vel_norm_x, rel_vel_norm_y,
             dist_to_ego_norm],
            dim=-1,
        )  # (B, 6, 5)

        # Full 14-dim features for every penguin (from this ego's POV)
        full_feats = torch.cat([abs_feats, rel_feats], dim=-1)  # (B, 6, 14)

        # --- Determine allies and enemies ---
        ego_team = 0 if ego_idx < 3 else 1

        if ego_team == 0:
            ally_indices = [i for i in range(3) if i != ego_idx]
            enemy_indices = [3, 4, 5]
        else:
            ally_indices = [i for i in range(3, 6) if i != ego_idx]
            enemy_indices = [0, 1, 2]

        # --- Sort allies by distance to ego (closest first) ---
        ally_feats_raw = full_feats[:, ally_indices]  # (B, 2, 14)
        ally_alive_raw = alive[:, ally_indices]  # (B, 2)
        ally_dist = dist_to_ego[:, ally_indices]  # (B, 2)

        # Dead allies get infinite distance so they sort last
        ally_sort_key = torch.where(
            ally_alive_raw, ally_dist, torch.full_like(ally_dist, 1e6)
        )
        ally_order = ally_sort_key.argsort(dim=-1)  # (B, 2)
        ally_order_exp = ally_order.unsqueeze(-1).expand(-1, -1, 14)
        sorted_ally_feats = ally_feats_raw.gather(1, ally_order_exp)  # (B, 2, 14)

        # --- Sort enemies by distance to ego (closest first) ---
        enemy_feats_raw = full_feats[:, enemy_indices]  # (B, 3, 14)
        enemy_alive_raw = alive[:, enemy_indices]  # (B, 3)
        enemy_dist = dist_to_ego[:, enemy_indices]  # (B, 3)

        enemy_sort_key = torch.where(
            enemy_alive_raw, enemy_dist, torch.full_like(enemy_dist, 1e6)
        )
        enemy_order = enemy_sort_key.argsort(dim=-1)  # (B, 3)
        enemy_order_exp = enemy_order.unsqueeze(-1).expand(-1, -1, 14)
        sorted_enemy_feats = enemy_feats_raw.gather(1, enemy_order_exp)  # (B, 3, 14)

        # --- Assemble final obs ---
        ally_flat = sorted_ally_feats.reshape(B, 28)  # (B, 28)
        enemy_flat = sorted_enemy_feats.reshape(B, 42)  # (B, 42)

        glob = global_a if ego_team == 0 else global_b  # (B, 5)

        agent_obs = torch.cat([ego_14, ally_flat, enemy_flat, glob], dim=-1)  # (B, 89)

        # Zero out entire observation if ego is dead
        agent_obs = agent_obs * ego_alive.unsqueeze(-1).float()

        obs[:, ego_idx] = agent_obs

    return obs
