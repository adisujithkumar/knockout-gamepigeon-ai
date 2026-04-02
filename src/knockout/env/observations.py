"""Observation builder for penguin knockout game.

Constructs 89-dimensional observation vectors for each agent:
- Ego penguin: 14 features
- 2 Allies: 14 features each (sorted by distance, 28 total)
- 3 Enemies: 14 features each (sorted by distance, 42 total)
- Global state: 5 features

Total: 14 + 28 + 42 + 5 = 89 features
"""

import numpy as np
import numpy.typing as npt

from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.penguin import Penguin


class ObservationBuilder:
    """Builds observations for multi-agent RL."""

    # Feature dimensions
    FEATURES_PER_PENGUIN = 14
    NUM_ALLIES = 2
    NUM_ENEMIES = 3
    GLOBAL_FEATURES = 5
    TOTAL_DIM = FEATURES_PER_PENGUIN * (1 + NUM_ALLIES + NUM_ENEMIES) + GLOBAL_FEATURES

    def __init__(self, config: GameConfig = DEFAULTS) -> None:
        """Initialize observation builder.

        Args:
            config: Game configuration for normalization constants
        """
        self.config = config
        assert self.TOTAL_DIM == 89, f"Expected 89 features, got {self.TOTAL_DIM}"

    def build_observation(
        self, ego_id: str, penguins: dict[str, Penguin], timestep: int
    ) -> npt.NDArray[np.float32]:
        """Build observation for a single agent.

        Args:
            ego_id: Agent ID of the observer (e.g., "penguin_0")
            penguins: Dictionary of all penguins in the game
            timestep: Current simulation timestep

        Returns:
            89-dimensional numpy array with normalized features
        """
        ego = penguins[ego_id]
        ego_state = ego.get_state()

        # If ego is dead, return zero observation
        if not ego_state.alive:
            return np.zeros(self.TOTAL_DIM, dtype=np.float32)

        # BUG FIX #5: capture ego velocity for true relative velocity calculation
        ego_vel = ego_state.velocity

        # Separate penguins by team
        allies = []
        enemies = []
        for p_id, penguin in penguins.items():
            if p_id == ego_id:
                continue
            p_state = penguin.get_state()
            if p_state.team_id == ego_state.team_id:
                allies.append(penguin)
            else:
                enemies.append(penguin)

        # Sort by distance to ego
        allies = self._sort_by_distance(allies, ego_state.position)
        enemies = self._sort_by_distance(enemies, ego_state.position)

        # Build observation vector
        obs: list[float] = []

        # 1. Ego features (14)
        obs.extend(
            self._extract_penguin_features(ego, ego_state.position, ego_vel, is_ego=True)
        )

        # 2. Ally features (14 * 2 = 28)
        for i in range(self.NUM_ALLIES):
            if i < len(allies):
                obs.extend(
                    self._extract_penguin_features(
                        allies[i], ego_state.position, ego_vel, is_ego=False
                    )
                )
            else:
                # Pad with zeros if ally is eliminated
                obs.extend([0.0] * self.FEATURES_PER_PENGUIN)

        # 3. Enemy features (14 * 3 = 42)
        for i in range(self.NUM_ENEMIES):
            if i < len(enemies):
                obs.extend(
                    self._extract_penguin_features(
                        enemies[i], ego_state.position, ego_vel, is_ego=False
                    )
                )
            else:
                # Pad with zeros if enemy is eliminated
                obs.extend([0.0] * self.FEATURES_PER_PENGUIN)

        # 4. Global features (5)
        obs.extend(self._extract_global_features(penguins, ego_state.team_id, timestep))

        return np.array(obs, dtype=np.float32)

    def _extract_penguin_features(
        self,
        penguin: Penguin,
        ego_pos: tuple[float, float],
        ego_vel: tuple[float, float],
        is_ego: bool = False,
    ) -> list[float]:
        """Extract 14 features for a single penguin. All clipped to [-1, 1].

        Features:
        1. position_x (normalized by half_width)
        2. position_y (normalized by half_width)
        3. velocity_x (normalized)
        4. velocity_y (normalized)
        5. distance_from_center (Chebyshev, normalized)
        6. distance_to_edge (min dist to nearest wall, normalized)
        7. speed (normalized)
        8. heading (velocity angle, normalized to [-1, 1])
        9. alive (binary: 1 or 0)
        10. relative_position_x (to ego, normalized)
        11. relative_position_y (to ego, normalized)
        12. relative_velocity_x (to ego, normalized)
        13. relative_velocity_y (to ego, normalized)
        14. distance_to_ego (normalized)

        Args:
            penguin: Penguin to extract features from
            ego_pos: Position of the observing agent
            ego_vel: Velocity of the observing agent (for BUG FIX #5)
            is_ego: True if this is the ego penguin

        Returns:
            List of 14 normalized features
        """
        state = penguin.get_state()
        pos = state.position
        vel = state.velocity

        # Normalization constants
        half_width = self.config.ARENA_HALF_WIDTH
        max_speed = 200.0  # Reasonable max speed for normalization

        # BUG FIX #4: clip ALL values to [-1, 1]
        pos_x_norm = np.clip(pos[0] / half_width, -1.0, 1.0)
        pos_y_norm = np.clip(pos[1] / half_width, -1.0, 1.0)
        vel_x_norm = np.clip(vel[0] / max_speed, -1.0, 1.0)
        vel_y_norm = np.clip(vel[1] / max_speed, -1.0, 1.0)

        # Distance features (Chebyshev distance for square arena)
        dist_from_center = max(abs(pos[0]), abs(pos[1]))
        dist_from_center_norm = np.clip(dist_from_center / half_width, 0.0, 1.0)
        dist_to_edge = half_width - dist_from_center
        dist_to_edge_norm = np.clip(dist_to_edge / half_width, -1.0, 1.0)

        # Speed and heading
        speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
        speed_norm = np.clip(speed / max_speed, 0.0, 1.0)
        heading = np.arctan2(vel[1], vel[0]) if speed > 1e-6 else 0.0
        heading_norm = np.clip(heading / np.pi, -1.0, 1.0)

        # Alive status
        alive = 1.0 if state.alive else 0.0

        # Relative features (to ego)
        if is_ego:
            # For ego, relative features are zero
            rel_pos_x_norm = 0.0
            rel_pos_y_norm = 0.0
            rel_vel_x_norm = 0.0
            rel_vel_y_norm = 0.0
            dist_to_ego_norm = 0.0
        else:
            rel_pos_x = pos[0] - ego_pos[0]
            rel_pos_y = pos[1] - ego_pos[1]
            rel_pos_x_norm = np.clip(rel_pos_x / (2 * half_width), -1.0, 1.0)
            rel_pos_y_norm = np.clip(rel_pos_y / (2 * half_width), -1.0, 1.0)

            # BUG FIX #5: true relative velocity (other - ego)
            rel_vel_x = vel[0] - ego_vel[0]
            rel_vel_y = vel[1] - ego_vel[1]
            rel_vel_x_norm = np.clip(rel_vel_x / max_speed, -1.0, 1.0)
            rel_vel_y_norm = np.clip(rel_vel_y / max_speed, -1.0, 1.0)

            dist_to_ego = np.sqrt(rel_pos_x ** 2 + rel_pos_y ** 2)
            dist_to_ego_norm = np.clip(dist_to_ego / (2 * half_width), 0.0, 1.0)

        return [
            pos_x_norm,
            pos_y_norm,
            vel_x_norm,
            vel_y_norm,
            dist_from_center_norm,
            dist_to_edge_norm,
            speed_norm,
            heading_norm,
            alive,
            rel_pos_x_norm,
            rel_pos_y_norm,
            rel_vel_x_norm,
            rel_vel_y_norm,
            dist_to_ego_norm,
        ]

    def _extract_global_features(
        self, penguins: dict[str, Penguin], ego_team_id: int, timestep: int
    ) -> list[float]:
        """Extract 5 global game state features.

        Features:
        1. team_a_alive_count (normalized by 3)
        2. team_b_alive_count (normalized by 3)
        3. ego_team_alive_count (normalized by 3)
        4. opponent_team_alive_count (normalized by 3)
        5. timestep (normalized by 1000)

        Args:
            penguins: Dictionary of all penguins
            ego_team_id: Team ID of the observing agent
            timestep: Current simulation timestep

        Returns:
            List of 5 normalized global features
        """
        team_a_alive = sum(1 for p in penguins.values() if p.team_id == 0 and p.alive)
        team_b_alive = sum(1 for p in penguins.values() if p.team_id == 1 and p.alive)

        ego_team_alive = team_a_alive if ego_team_id == 0 else team_b_alive
        opponent_team_alive = team_b_alive if ego_team_id == 0 else team_a_alive

        return [
            team_a_alive / 3.0,
            team_b_alive / 3.0,
            ego_team_alive / 3.0,
            opponent_team_alive / 3.0,
            min(timestep / 1000.0, 1.0),
        ]

    def _sort_by_distance(
        self, penguins: list[Penguin], ego_pos: tuple[float, float]
    ) -> list[Penguin]:
        """Sort penguins by distance to ego (closest first).

        Args:
            penguins: List of penguins to sort
            ego_pos: Position of the observing agent

        Returns:
            Sorted list of penguins (closest first)
        """

        def distance_to_ego(penguin: Penguin) -> float:
            state = penguin.get_state()
            if not state.alive:
                return float("inf")  # Dead penguins go to the end
            pos = state.position
            return float(np.sqrt((pos[0] - ego_pos[0]) ** 2 + (pos[1] - ego_pos[1]) ** 2))

        return sorted(penguins, key=distance_to_ego)
