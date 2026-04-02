"""Tests for Penguin class."""

import math

import numpy as np
import pymunk

from knockout.core.config import DEFAULTS
from knockout.core.penguin import Penguin, PenguinState


class TestPenguin:
    """Tests for Penguin entity with Pymunk physics."""

    def test_creation(self) -> None:
        """Test creating a penguin."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(10.0, 20.0),
            space=space,
        )

        assert penguin.agent_id == "penguin_0"
        assert penguin.team_id == 0
        assert penguin.alive is True

        # Check body exists
        assert penguin.body is not None
        assert penguin.body.mass == DEFAULTS.PENGUIN_MASS

        # Check shape exists
        assert penguin.shape is not None
        assert penguin.shape.radius == DEFAULTS.PENGUIN_RADIUS

        # Check metadata stored on shape
        assert penguin.shape.penguin_id == "penguin_0"  # type: ignore[attr-defined]
        assert penguin.shape.team_id == 0  # type: ignore[attr-defined]

    def test_initial_position(self) -> None:
        """Test penguin starts at correct position."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(50.0, -30.0),
            space=space,
        )

        x, y = penguin.position
        assert abs(x - 50.0) < 1e-6
        assert abs(y + 30.0) < 1e-6

    def test_initial_velocity_zero(self) -> None:
        """Test penguin starts with zero velocity."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        vx, vy = penguin.velocity
        assert abs(vx) < 1e-6
        assert abs(vy) < 1e-6

    def test_apply_launch_changes_velocity(self) -> None:
        """Test that applying launch changes velocity."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        # Apply launch at 0 degrees (east) with 300N power
        penguin.apply_launch(angle_degrees=0.0, power_newtons=300.0)

        vx, vy = penguin.velocity
        # Should have positive x velocity
        assert vx > 0
        # Y velocity should be close to zero
        assert abs(vy) < 1.0

    def test_apply_launch_90_degrees(self) -> None:
        """Test launching at 90 degrees (north)."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        # Apply launch at 90 degrees (north) with 300N power
        penguin.apply_launch(angle_degrees=90.0, power_newtons=300.0)

        vx, vy = penguin.velocity
        # X velocity should be close to zero
        assert abs(vx) < 1.0
        # Should have positive y velocity
        assert vy > 0

    def test_apply_launch_180_degrees(self) -> None:
        """Test launching at 180 degrees (west)."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        # Apply launch at 180 degrees (west) with 300N power
        penguin.apply_launch(angle_degrees=180.0, power_newtons=300.0)

        vx, vy = penguin.velocity
        # Should have negative x velocity
        assert vx < 0
        # Y velocity should be close to zero
        assert abs(vy) < 1.0

    def test_apply_launch_dead_penguin_ignored(self) -> None:
        """Test that dead penguins ignore launch commands."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        # Eliminate penguin
        penguin.eliminate(space)

        # Try to launch (should be ignored)
        penguin.apply_launch(angle_degrees=0.0, power_newtons=300.0)

        vx, vy = penguin.velocity
        # Velocity should still be zero
        assert abs(vx) < 1e-6
        assert abs(vy) < 1e-6

    def test_eliminate_marks_dead(self) -> None:
        """Test that eliminate marks penguin as dead."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        assert penguin.alive is True

        penguin.eliminate(space)

        assert penguin.alive is False

    def test_eliminate_removes_from_space(self) -> None:
        """Test that eliminate removes body and shape from space."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        # Verify added to space
        assert len(space.bodies) == 1
        assert len(space.shapes) == 1

        penguin.eliminate(space)

        # Verify removed from space
        assert len(space.bodies) == 0
        assert len(space.shapes) == 0

    def test_eliminate_idempotent(self) -> None:
        """Test that calling eliminate multiple times is safe."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        # Eliminate twice
        penguin.eliminate(space)
        penguin.eliminate(space)  # Should not raise error

        assert penguin.alive is False

    def test_get_state_snapshot(self) -> None:
        """Test getting state snapshot."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(10.0, 20.0),
            space=space,
        )

        state = penguin.get_state()

        assert isinstance(state, PenguinState)
        assert state.agent_id == "penguin_0"
        assert state.team_id == 0
        assert state.alive is True
        assert abs(state.position[0] - 10.0) < 1e-6
        assert abs(state.position[1] - 20.0) < 1e-6

    def test_get_state_after_launch(self) -> None:
        """Test state snapshot captures velocity after launch."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        penguin.apply_launch(angle_degrees=0.0, power_newtons=300.0)
        state = penguin.get_state()

        # Should have non-zero velocity
        assert abs(state.velocity[0]) > 0 or abs(state.velocity[1]) > 0

    def test_get_state_dead_penguin(self) -> None:
        """Test state snapshot for eliminated penguin."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        penguin.eliminate(space)
        state = penguin.get_state()

        assert state.alive is False

    def test_position_property(self) -> None:
        """Test position property returns correct values."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(15.0, 25.0),
            space=space,
        )

        x, y = penguin.position
        assert abs(x - 15.0) < 1e-6
        assert abs(y - 25.0) < 1e-6

    def test_velocity_property(self) -> None:
        """Test velocity property returns correct values."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        penguin.apply_launch(angle_degrees=45.0, power_newtons=200.0)

        vx, vy = penguin.velocity
        # Both components should be non-zero for 45 degree angle
        assert vx > 0
        assert vy > 0

    def test_physics_properties(self) -> None:
        """Test that physics properties are set correctly."""
        space = pymunk.Space()
        penguin = Penguin(
            agent_id="penguin_0",
            team_id=0,
            position=(0.0, 0.0),
            space=space,
        )

        # Check elasticity
        assert penguin.shape.elasticity == DEFAULTS.ELASTICITY

        # Check friction
        assert penguin.shape.friction == DEFAULTS.ICE_FRICTION

    def test_impulse_direction_independent_of_rotation(self) -> None:
        """Test that launch direction is independent of body rotation.

        BUG FIX #1 verification: apply_impulse_at_world_point ensures
        the impulse direction is always in world coordinates, regardless
        of the penguin body's current rotation angle.
        """
        angles_to_test = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        velocities = []

        for body_angle in angles_to_test:
            space = pymunk.Space()
            penguin = Penguin(
                agent_id="penguin_0",
                team_id=0,
                position=(0.0, 0.0),
                space=space,
            )

            # Set body rotation to various angles
            penguin.body.angle = body_angle

            # Apply the same launch (0 degrees, east, 300N)
            penguin.apply_launch(angle_degrees=0.0, power_newtons=300.0)

            vx, vy = penguin.velocity
            velocities.append((vx, vy))

        # All velocities should be the same regardless of body rotation
        for i in range(1, len(velocities)):
            assert abs(velocities[i][0] - velocities[0][0]) < 1e-6, (
                f"vx differs at body_angle={angles_to_test[i]}: "
                f"{velocities[i][0]} vs {velocities[0][0]}"
            )
            assert abs(velocities[i][1] - velocities[0][1]) < 1e-6, (
                f"vy differs at body_angle={angles_to_test[i]}: "
                f"{velocities[i][1]} vs {velocities[0][1]}"
            )
