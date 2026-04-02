"""Deterministic physics simulation for penguin knockout game."""

import numpy as np
import pymunk

from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.ice_sheet import IceSheet
from knockout.core.penguin import Penguin


class PhysicsEngine:
    """Deterministic physics simulation for penguin knockout game."""

    def __init__(self, seed: int | None = None, config: GameConfig = DEFAULTS):
        """Initialize physics engine with optional seed for determinism.

        Args:
            seed: Random seed for deterministic physics (None for non-deterministic)
            config: Physics configuration to use
        """
        self.config = config

        # BUG FIX #2: Use instance-level RNG instead of global np.random.seed()
        self.rng = np.random.default_rng(seed)

        # Create Pymunk space
        self.space = pymunk.Space()
        self.space.gravity = config.GRAVITY
        self.space.damping = config.DAMPING
        self.space.iterations = config.COLLISION_ITERATIONS
        self.space.collision_bias = config.COLLISION_BIAS

        # Disable sleeping bodies for determinism
        self.space.sleep_time_threshold = float("inf")

        # Game state
        self.ice_sheet = IceSheet(half_width=config.ARENA_HALF_WIDTH)
        self.penguins: dict[str, Penguin] = {}
        self.step_count = 0

    def initialize_game(self) -> None:
        """Create 6 penguins at spawn positions."""
        # Clear any existing penguins
        for penguin in list(self.penguins.values()):
            if penguin.alive:
                penguin.eliminate(self.space)

        self.penguins.clear()
        self.step_count = 0

        # Team A (penguins 0-2)
        spawn_positions_a = self.ice_sheet.get_spawn_positions(team_id=0)
        for i, pos in enumerate(spawn_positions_a):
            agent_id = f"penguin_{i}"
            self.penguins[agent_id] = Penguin(
                agent_id=agent_id,
                team_id=0,
                position=pos,
                space=self.space,
                config=self.config,
            )

        # Team B (penguins 3-5)
        spawn_positions_b = self.ice_sheet.get_spawn_positions(team_id=1)
        for i, pos in enumerate(spawn_positions_b):
            agent_id = f"penguin_{i + 3}"
            self.penguins[agent_id] = Penguin(
                agent_id=agent_id,
                team_id=1,
                position=pos,
                space=self.space,
                config=self.config,
            )

    def apply_actions(self, actions: dict[str, tuple[float, float]]) -> None:
        """Apply all actions simultaneously BEFORE physics step.

        Args:
            actions: Dict mapping agent_id -> (angle_degrees, power_newtons)
        """
        for agent_id, (angle, power) in actions.items():
            if agent_id in self.penguins:
                # Clip actions to valid range
                angle_clipped = np.clip(angle, 0.0, 360.0)
                power_clipped = np.clip(power, 0.0, self.config.MAX_LAUNCH_FORCE)

                self.penguins[agent_id].apply_launch(angle_clipped, power_clipped)

    def step(self, num_steps: int = 1) -> None:
        """Step physics simulation with fixed timestep.

        Args:
            num_steps: Number of physics steps to execute
        """
        for _ in range(num_steps):
            # Step with FIXED timestep (critical for determinism)
            self.space.step(self.config.FIXED_DT)
            self.step_count += 1

            # Check for eliminations after each step
            self._check_eliminations()

    def are_all_settled(self, speed_threshold: float | None = None) -> bool:
        """Check if all alive penguins have near-zero velocity.

        Args:
            speed_threshold: Speed below which a penguin is considered at rest.
                             Defaults to config.SETTLE_SPEED_THRESHOLD.

        Returns:
            True if every alive penguin's speed is below the threshold.
        """
        if speed_threshold is None:
            speed_threshold = self.config.SETTLE_SPEED_THRESHOLD
        for penguin in self.penguins.values():
            if penguin.alive:
                vx, vy = penguin.velocity
                speed = (vx**2 + vy**2) ** 0.5
                if speed > speed_threshold:
                    return False
        return True

    def step_until_settled(
        self,
        speed_threshold: float | None = None,
        max_steps: int = 10000,
    ) -> int:
        """Step physics until all penguins have settled or max_steps reached.

        The simulation continues until every alive penguin is at rest (speed
        below *speed_threshold*) **or** no alive penguins remain.  Crucially,
        ``is_game_over()`` is **not** used as a stopping condition because a
        mutual knockout (both teams' last penguins collide and fly off) must
        be fully resolved before we declare the result.

        Args:
            speed_threshold: Speed below which a penguin is considered at rest.
                             Defaults to config.SETTLE_SPEED_THRESHOLD.
            max_steps: Safety cap to avoid infinite loops.

        Returns:
            The number of physics steps taken.
        """
        if speed_threshold is None:
            speed_threshold = self.config.SETTLE_SPEED_THRESHOLD
        steps_taken = 0
        for _ in range(max_steps):
            self.space.step(self.config.FIXED_DT)
            self.step_count += 1
            steps_taken += 1
            self._check_eliminations()

            # Stop only when every alive penguin is at rest.
            # are_all_settled() returns True when no alive penguins remain
            # (vacuously true), which correctly handles the mutual-KO case.
            if self.are_all_settled(speed_threshold):
                break
        return steps_taken

    def rescale_penguins(self, scale_factor: float) -> None:
        """Rescale all alive penguin positions and velocities.

        Used after arena shrink so penguins maintain their relative
        positions within the arena (normalized coordinates stay the same).
        After rescaling, positions are clamped to stay safely inside
        the new boundary so that shrinking NEVER eliminates penguins.

        Args:
            scale_factor: Multiply positions/velocities by this
                (new_half_width / old_half_width)
        """
        hw = self.ice_sheet.half_width
        # Leave a small margin so penguins are not exactly on the edge
        # (prevents floating-point rounding from pushing them outside).
        margin = max(0.01, hw * 0.005)
        limit = hw - margin

        for penguin in self.penguins.values():
            if penguin.alive:
                px, py = penguin.body.position
                new_px = px * scale_factor
                new_py = py * scale_factor
                # Clamp to stay safely inside the new arena boundary
                new_px = max(-limit, min(limit, new_px))
                new_py = max(-limit, min(limit, new_py))
                penguin.body.position = (new_px, new_py)
                vx, vy = penguin.body.velocity
                penguin.body.velocity = (vx * scale_factor, vy * scale_factor)

    def _check_eliminations(self) -> None:
        """Remove penguins that left the ice sheet."""
        for penguin in self.penguins.values():
            if penguin.alive and not self.ice_sheet.is_inside(penguin.position):
                penguin.eliminate(self.space)

    def get_alive_count(self, team_id: int) -> int:
        """Count alive penguins for a team.

        Args:
            team_id: Team number (0 or 1)

        Returns:
            Number of alive penguins on that team
        """
        return sum(1 for p in self.penguins.values() if p.team_id == team_id and p.alive)

    def is_game_over(self) -> bool:
        """Check if game is over (one team eliminated).

        Returns:
            True if at least one team has zero penguins alive
        """
        team_a_alive = self.get_alive_count(team_id=0)
        team_b_alive = self.get_alive_count(team_id=1)
        return team_a_alive == 0 or team_b_alive == 0

    def get_winner(self) -> int | None:
        """Get winning team ID (0 or 1) or None if game ongoing.

        Returns:
            0 if Team A wins, 1 if Team B wins, -1 if draw, None if game ongoing
        """
        team_a_alive = self.get_alive_count(team_id=0)
        team_b_alive = self.get_alive_count(team_id=1)

        if team_a_alive == 0 and team_b_alive > 0:
            return 1  # Team B wins
        elif team_b_alive == 0 and team_a_alive > 0:
            return 0  # Team A wins
        elif team_a_alive == 0 and team_b_alive == 0:
            return -1  # Draw
        else:
            return None  # Game ongoing
