"""Heuristic agent with predictive aiming (Tier 1).

Bug fixes applied vs knockout-claude:
  #6 - Geometric-sum prediction damping (was applying final damping to whole displacement)
  #7 - Utility weights sum to 1.0 with velocity-toward-edge factor
"""

import numpy as np

from knockout.agents.base import Agent
from knockout.core.config import GameConfig, DEFAULTS


class HeuristicAgent(Agent):
    """Rule-based agent with predictive aiming and edge awareness.

    Strategy:
    1. Target Selection: Attack closest enemy
    2. Predictive Aiming: Aim where enemy will be (not where they are)
    3. Edge Awareness: Avoid launching self off edge
    4. Power Calculation: Scale power based on distance
    """

    def __init__(
        self,
        agent_id: str,
        seed: int | None = None,
        config: GameConfig = DEFAULTS,
    ):
        """Initialize heuristic agent.

        Args:
            agent_id: Unique identifier for this agent
            seed: Random seed for reproducibility (optional)
            config: Game configuration constants
        """
        super().__init__(agent_id)
        self.config = config
        self.arena_half_width = config.ARENA_HALF_WIDTH
        self.rng = np.random.default_rng(seed)
        self.target_selection_temperature = 0.4  # Softmax temperature

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Select action using heuristic rules.

        Args:
            observation: 89-dimensional observation vector

        Returns:
            Action [angle_degrees, power_newtons]
        """
        # Parse observation
        ego_state = self._parse_ego(observation)
        enemies = self._parse_enemies(observation)

        # Safety check: if too close to edge, launch toward center
        if ego_state["distance_to_edge"] < self.config.PENGUIN_RADIUS * 1.5:
            return self._launch_toward_center(ego_state)

        # Find alive enemies
        alive_enemies = [e for e in enemies if e["alive"]]
        if not alive_enemies:
            # No enemies, do nothing
            return np.array([0.0, 0.0], dtype=np.float32)

        # Target selection: probabilistic based on utility scores
        target = self._select_target_probabilistic(alive_enemies, ego_state)

        # Predictive aiming: estimate where target will be
        # Use variable prediction steps based on distance
        target_distance = target["distance_to_ego"]
        prediction_steps = self._get_prediction_steps(target_distance)
        predicted_pos = self._predict_position(target, prediction_steps)

        # Calculate launch angle and power
        angle = self._calculate_angle(ego_state["position"], predicted_pos)
        base_power = self._calculate_power(ego_state["position"], predicted_pos)
        power = self._add_power_variation(base_power)

        return np.array([angle, power], dtype=np.float32)

    def _parse_ego(self, obs: np.ndarray) -> dict:
        """Parse ego penguin features from observation.

        Ego features are indices 0-13:
        0-1: position_x, position_y (normalized)
        2-3: velocity_x, velocity_y (normalized)
        4: distance_from_center (normalized)
        5: distance_to_edge (normalized)
        6: speed (normalized)
        7: heading (normalized)
        8: alive
        9-13: relative features (all zero for ego)
        """
        radius = self.arena_half_width
        return {
            "position": (obs[0] * radius, obs[1] * radius),
            "velocity": (obs[2] * 200.0, obs[3] * 200.0),  # Denormalize velocity
            "distance_from_center": obs[4] * radius,
            "distance_to_edge": obs[5] * radius,
            "speed": obs[6] * 200.0,
            "alive": bool(obs[8] > 0.5),
        }

    def _parse_enemies(self, obs: np.ndarray) -> list[dict]:
        """Parse enemy features from observation.

        Enemies are indices 42-83 (3 enemies * 14 features each).
        Extracts position, velocity, distance info, and alive status.
        """
        enemies = []
        radius = self.arena_half_width

        for i in range(3):
            start_idx = 42 + i * 14
            enemy_obs = obs[start_idx : start_idx + 14]

            enemies.append(
                {
                    "position": (enemy_obs[0] * radius, enemy_obs[1] * radius),
                    "velocity": (enemy_obs[2] * 200.0, enemy_obs[3] * 200.0),
                    "distance_to_ego": enemy_obs[13] * (2 * radius),  # Denormalize
                    "distance_to_edge": enemy_obs[5] * radius,  # Denormalize edge distance
                    "alive": bool(enemy_obs[8] > 0.5),
                }
            )

        return enemies

    def _calculate_target_utility(self, enemy: dict, ego_state: dict) -> float:
        """Calculate utility score for targeting this enemy.

        BUG FIX #7: Weights now sum to 1.0 (was 0.65).
        Factors:
        - Edge proximity (40%): Enemies near edge are easier to eliminate
        - Distance to ego (25%): Closer = easier to hit accurately
        - Velocity toward edge (35%): Enemies moving outward are better targets

        Args:
            enemy: Enemy state dict with position, velocity, etc.
            ego_state: Ego penguin state dict

        Returns:
            Utility score in [0, 1] (higher = better target)
        """
        # Edge proximity score (higher = closer to edge = higher priority)
        edge_score = 1.0 - (enemy["distance_to_edge"] / self.arena_half_width)

        # Distance score (closer = higher priority)
        distance_score = 1.0 - min(enemy["distance_to_ego"] / (2 * self.arena_half_width), 1.0)

        # BUG FIX #7: velocity-toward-edge factor (35%)
        enemy_pos = np.array(enemy["position"])
        enemy_vel = np.array(enemy["velocity"])
        dist_from_center = float(np.linalg.norm(enemy_pos))
        if dist_from_center > 1e-6:
            outward_dir = enemy_pos / dist_from_center
            vel_toward_edge = float(np.dot(enemy_vel, outward_dir))
            # Normalize: positive = moving toward edge (good target), scale by max_speed
            vel_toward_edge_score = float(np.clip(vel_toward_edge / 200.0, 0.0, 1.0))
        else:
            vel_toward_edge_score = 0.0

        # Weighted combination: 0.40 + 0.25 + 0.35 = 1.0
        utility = 0.4 * edge_score + 0.25 * distance_score + 0.35 * vel_toward_edge_score

        return utility

    def _select_target_probabilistic(
        self, alive_enemies: list[dict], ego_state: dict
    ) -> dict:
        """Select target using probabilistic sampling based on utility scores.

        Uses softmax with temperature to balance exploitation vs exploration.

        Args:
            alive_enemies: List of alive enemy state dicts
            ego_state: Ego penguin state dict

        Returns:
            Selected target enemy dict
        """
        if len(alive_enemies) == 1:
            return alive_enemies[0]

        # Calculate utilities for all enemies
        utilities = np.array([
            self._calculate_target_utility(enemy, ego_state)
            for enemy in alive_enemies
        ])

        # Softmax with temperature
        exp_utilities = np.exp(utilities / self.target_selection_temperature)
        probabilities = exp_utilities / exp_utilities.sum()

        # Sample target based on probabilities
        target_idx = self.rng.choice(len(alive_enemies), p=probabilities)

        return alive_enemies[target_idx]

    def _get_prediction_steps(self, distance_to_target: float) -> int:
        """Variable prediction horizon based on target distance.

        Closer targets = shorter prediction (faster reaction)
        Farther targets = longer prediction (more lead time)

        Args:
            distance_to_target: Distance from ego to target in units

        Returns:
            Number of prediction steps
        """
        if distance_to_target < 50.0:
            return int(self.rng.integers(6, 9))  # Quick reactions for close targets
        elif distance_to_target < 100.0:
            return int(self.rng.integers(8, 11))  # Medium prediction
        else:
            return int(self.rng.integers(10, 14))  # Long prediction for distant targets

    def _predict_position(
        self, target: dict, prediction_steps: int
    ) -> tuple[float, float]:
        """Predict where target will be after N steps.

        BUG FIX #6: Uses geometric sum for position with per-step damping.
        At each step k, velocity is v * d^k, displacement is v * dt * d^k.
        Total displacement = v * dt * sum(d^k for k=0..N-1)
                           = v * dt * (1 - d^N) / (1 - d)

        Note: Pymunk's space.damping is the fraction of velocity retained per
        SECOND, so per-step retention = damping^dt.

        Args:
            target: Target enemy state dict
            prediction_steps: Number of physics steps to predict ahead

        Returns:
            Predicted (x, y) position
        """
        dt = self.config.FIXED_DT
        # Pymunk damping is per-second; convert to per-step
        d = self.config.DAMPING ** dt

        if abs(d - 1.0) < 1e-9:
            displacement_factor = dt * prediction_steps
        else:
            displacement_factor = dt * (1.0 - d**prediction_steps) / (1.0 - d)

        pred_x = target["position"][0] + target["velocity"][0] * displacement_factor
        pred_y = target["position"][1] + target["velocity"][1] * displacement_factor

        return (pred_x, pred_y)

    def _calculate_angle(
        self, from_pos: tuple[float, float], to_pos: tuple[float, float]
    ) -> float:
        """Calculate launch angle to hit target.

        Args:
            from_pos: Current position (x, y)
            to_pos: Target position (x, y)

        Returns:
            Angle in degrees (0-360)
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.rad2deg(angle_rad)

        # Convert to 0-360 range
        if angle_deg < 0:
            angle_deg += 360.0

        return float(angle_deg)

    def _calculate_power(
        self, from_pos: tuple[float, float], to_pos: tuple[float, float]
    ) -> float:
        """Calculate launch power based on distance to target.

        Note: Pymunk's space.damping is the fraction of velocity retained per
        SECOND, so the per-step retention factor is damping^dt.  The impulse
        needed to travel a given distance d is:
            power = d * mass * (1 - damping^dt) / dt

        Args:
            from_pos: Current position (x, y)
            to_pos: Target position (x, y)

        Returns:
            Power in Newtons
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Pymunk damping is per-second; convert to per-step retention
        per_step = self.config.DAMPING ** self.config.FIXED_DT
        k = self.config.PENGUIN_MASS * (1 - per_step) / self.config.FIXED_DT
        power = np.clip(distance * k, 100.0, self.config.MAX_LAUNCH_FORCE)

        return float(power)

    def _add_power_variation(self, base_power: float) -> float:
        """Add small random variation (+/-5%) to power to break symmetry.

        Args:
            base_power: Base power in Newtons

        Returns:
            Power with random variation applied
        """
        noise = self.rng.uniform(-0.05, 0.05)
        return float(np.clip(base_power * (1.0 + noise), 100.0, self.config.MAX_LAUNCH_FORCE))

    def _launch_toward_center(self, ego_state: dict) -> np.ndarray:
        """Emergency action: launch toward center to avoid falling off.

        Args:
            ego_state: Ego penguin state dict

        Returns:
            Action [angle_degrees, power_newtons] toward center
        """
        ego_pos = ego_state["position"]
        center_pos = (0.0, 0.0)

        angle = self._calculate_angle(ego_pos, center_pos)
        power = self.config.MAX_LAUNCH_FORCE * 0.6  # Medium power to move toward center

        return np.array([angle, power], dtype=np.float32)
