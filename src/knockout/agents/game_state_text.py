"""Convert 89-dim observation vectors to human-readable game state text.

Standalone module so it can be reused by different agent types (LLM, logging, etc.).
"""

import math

import numpy as np

from knockout.core.config import GameConfig, DEFAULTS


# Observation layout constants (must match observations.py)
FEATURES_PER_PENGUIN = 14
NUM_ALLIES = 2
NUM_ENEMIES = 3
GLOBAL_FEATURES = 5
MAX_SPEED = 200.0  # Velocity normalization constant used in observations.py

# Feature indices within each 14-feature penguin block
_POS_X = 0
_POS_Y = 1
_VEL_X = 2
_VEL_Y = 3
_DIST_CENTER = 4
_DIST_EDGE = 5
_SPEED = 6
_HEADING = 7
_ALIVE = 8
_REL_POS_X = 9
_REL_POS_Y = 10
_REL_VEL_X = 11
_REL_VEL_Y = 12
_DIST_EGO = 13

# Block start indices
_EGO_START = 0
_ALLY_START = FEATURES_PER_PENGUIN  # 14
_ENEMY_START = FEATURES_PER_PENGUIN * (1 + NUM_ALLIES)  # 42
_GLOBAL_START = FEATURES_PER_PENGUIN * (1 + NUM_ALLIES + NUM_ENEMIES)  # 84


def _agent_team(agent_id: str) -> str:
    """Return 'A' or 'B' based on agent index."""
    idx = int(agent_id.split("_")[1])
    return "A" if idx < 3 else "B"


def _ally_ids(agent_id: str) -> list[str]:
    """Return the two ally agent IDs (sorted by obs order = distance)."""
    idx = int(agent_id.split("_")[1])
    if idx < 3:
        teammates = [i for i in range(3) if i != idx]
    else:
        teammates = [i for i in range(3, 6) if i != idx]
    return [f"penguin_{i}" for i in teammates]


def _enemy_ids(agent_id: str) -> list[str]:
    """Return the three enemy agent IDs."""
    idx = int(agent_id.split("_")[1])
    if idx < 3:
        return [f"penguin_{i}" for i in range(3, 6)]
    else:
        return [f"penguin_{i}" for i in range(3)]


def _parse_block(obs: np.ndarray, block_start: int,
                 half_width: float) -> dict:
    """Parse a 14-feature penguin block into denormalized values."""
    b = obs[block_start:block_start + FEATURES_PER_PENGUIN]
    alive = bool(b[_ALIVE] > 0.5)
    return {
        "pos_x": float(b[_POS_X] * half_width),
        "pos_y": float(b[_POS_Y] * half_width),
        "vel_x": float(b[_VEL_X] * MAX_SPEED),
        "vel_y": float(b[_VEL_Y] * MAX_SPEED),
        "dist_center": float(b[_DIST_CENTER] * half_width),
        "dist_edge": float(b[_DIST_EDGE] * half_width),
        "speed": float(b[_SPEED] * MAX_SPEED),
        "heading_norm": float(b[_HEADING]),
        "alive": alive,
        "rel_pos_x": float(b[_REL_POS_X] * 2 * half_width),
        "rel_pos_y": float(b[_REL_POS_Y] * 2 * half_width),
        "dist_ego": float(b[_DIST_EGO] * 2 * half_width),
    }


def _format_penguin_line(label: str, data: dict, *,
                         tag: str = "") -> str:
    """Format a single penguin's state as one readable line."""
    if not data["alive"]:
        return f"  {label}: ELIMINATED"
    pos = f"({data['pos_x']:.1f}, {data['pos_y']:.1f})"
    spd = f"speed: {data['speed']:.1f}"
    edge = f"{data['dist_edge']:.1f} units from edge"
    suffix = f" {tag}" if tag else ""
    return f"  {label}: {pos}, {spd}, {edge}{suffix}"


def obs_to_text(obs: np.ndarray, agent_id: str,
                config: GameConfig = DEFAULTS) -> str:
    """Convert an 89-dim observation to human-readable game state text.

    All values are denormalized to actual game coordinates.

    Args:
        obs: 89-dimensional normalized observation vector
        agent_id: e.g. "penguin_0"
        config: Game configuration for arena size

    Returns:
        Multi-line text description of the game state.
    """
    hw = config.ARENA_HALF_WIDTH
    team = _agent_team(agent_id)

    # Check if observation is all zeros (dead agent)
    if np.allclose(obs, 0.0):
        return (
            f"You are {agent_id} (Team {team}).\n"
            f"STATUS: You have been ELIMINATED.\n"
            f"No action required."
        )

    # --- Ego ---
    ego = _parse_block(obs, _EGO_START, hw)

    # --- Allies (sorted by distance in obs) ---
    ally_labels = _ally_ids(agent_id)
    allies = []
    for i in range(NUM_ALLIES):
        start = _ALLY_START + i * FEATURES_PER_PENGUIN
        data = _parse_block(obs, start, hw)
        allies.append((ally_labels[i], data))

    # --- Enemies (sorted by distance in obs) ---
    enemy_labels = _enemy_ids(agent_id)
    enemies = []
    for i in range(NUM_ENEMIES):
        start = _ENEMY_START + i * FEATURES_PER_PENGUIN
        data = _parse_block(obs, start, hw)
        enemies.append((enemy_labels[i], data))

    # Find closest alive enemy for tagging
    closest_enemy = None
    closest_dist = float("inf")
    for label, data in enemies:
        if data["alive"] and data["dist_ego"] < closest_dist:
            closest_dist = data["dist_ego"]
            closest_enemy = label

    # --- Global features ---
    g = obs[_GLOBAL_START:]
    team_a_alive = int(round(g[0] * 3))
    team_b_alive = int(round(g[1] * 3))
    ego_team_alive = int(round(g[2] * 3))
    opp_team_alive = int(round(g[3] * 3))
    timestep_norm = float(g[4])

    # Infer round number (timestep_norm = min(timestep / 1000, 1))
    # This is approximate since timestep includes sub-steps
    round_est = int(timestep_norm * 1000)

    # --- Build text ---
    lines = []
    lines.append(f"You are {agent_id} (Team {team}).")
    lines.append("")
    lines.append(
        f"YOUR POSITION: ({ego['pos_x']:.1f}, {ego['pos_y']:.1f}), "
        f"speed: {ego['speed']:.1f}, "
        f"{ego['dist_edge']:.1f} units from nearest edge"
    )
    lines.append("")

    # Allies
    lines.append("ALLIES:")
    for label, data in allies:
        lines.append(_format_penguin_line(label, data))
    lines.append("")

    # Enemies
    lines.append("ENEMIES:")
    for label, data in enemies:
        tag = "[CLOSEST]" if label == closest_enemy else ""
        lines.append(_format_penguin_line(label, data, tag=tag))
    lines.append("")

    # Arena and team status
    arena_size = hw * 2  # full width
    lines.append(f"ARENA: {arena_size:.0f}x{arena_size:.0f}, timestep ~{round_est}")
    lines.append(f"TEAM STATUS: {ego_team_alive} alive vs {opp_team_alive} alive")
    lines.append("")
    lines.append("Choose your action: angle (0-360) and power (0-400).")
    lines.append("Angles: 0=right, 90=up, 180=left, 270=down")

    return "\n".join(lines)
