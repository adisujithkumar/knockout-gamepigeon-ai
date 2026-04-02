"""Penguin entity with Pymunk physics."""

from dataclasses import dataclass

import numpy as np
import pymunk

from knockout.core.config import GameConfig, DEFAULTS


@dataclass
class PenguinState:
    """Observation snapshot of a penguin."""

    position: tuple[float, float]
    velocity: tuple[float, float]
    alive: bool
    agent_id: str
    team_id: int


class Penguin:
    """Penguin entity with Pymunk physics body."""

    def __init__(
        self,
        agent_id: str,
        team_id: int,
        position: tuple[float, float],
        space: pymunk.Space,
        config: GameConfig = DEFAULTS,
    ):
        """Create a penguin with physics body.

        Args:
            agent_id: Unique identifier (e.g., "penguin_0")
            team_id: Team number (0 or 1)
            position: Initial (x, y) position
            space: Pymunk space to add body to
            config: Physics configuration to use
        """
        self.agent_id = agent_id
        self.team_id = team_id
        self.alive = True

        # Create Pymunk body (dynamic, not kinematic)
        mass = config.PENGUIN_MASS
        radius = config.PENGUIN_RADIUS
        moment = pymunk.moment_for_circle(mass, 0, radius)

        self.body = pymunk.Body(mass, moment)
        self.body.position = position

        # Create collision shape
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = config.ELASTICITY
        self.shape.friction = config.ICE_FRICTION

        # Store metadata on shape for collision detection
        self.shape.penguin_id = agent_id
        self.shape.team_id = team_id

        # Add to space
        space.add(self.body, self.shape)

    def apply_launch(self, angle_degrees: float, power_newtons: float) -> None:
        """Apply launch force as impulse (action execution).

        Args:
            angle_degrees: Launch angle in degrees (0-360)
            power_newtons: Launch power in Newtons (0-MAX_LAUNCH_FORCE)
        """
        if not self.alive:
            return

        # Convert angle to radians
        angle_rad = np.deg2rad(angle_degrees)

        # Calculate impulse components
        impulse_x = power_newtons * np.cos(angle_rad)
        impulse_y = power_newtons * np.sin(angle_rad)

        # BUG FIX #1: Use world point to avoid rotation-dependent direction
        # OLD (BUG): self.body.apply_impulse_at_local_point((impulse_x, impulse_y))
        self.body.apply_impulse_at_world_point((impulse_x, impulse_y), self.body.position)

    def eliminate(self, space: pymunk.Space) -> None:
        """Mark penguin as eliminated and remove from physics.

        Args:
            space: Pymunk space to remove body from
        """
        if not self.alive:
            return

        self.alive = False
        space.remove(self.body, self.shape)

    def get_state(self) -> PenguinState:
        """Get current state snapshot for observations.

        Returns:
            PenguinState with current position, velocity, etc.
        """
        return PenguinState(
            position=(self.body.position.x, self.body.position.y),
            velocity=(self.body.velocity.x, self.body.velocity.y),
            alive=self.alive,
            agent_id=self.agent_id,
            team_id=self.team_id,
        )

    @property
    def position(self) -> tuple[float, float]:
        """Convenience property for position."""
        return (self.body.position.x, self.body.position.y)

    @property
    def velocity(self) -> tuple[float, float]:
        """Convenience property for velocity."""
        return (self.body.velocity.x, self.body.velocity.y)
