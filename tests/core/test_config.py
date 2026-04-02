"""Tests for physics constants."""

from dataclasses import FrozenInstanceError

import pytest

from knockout.core.config import DEFAULTS, GameConfig


class TestGameConfig:
    """Tests for GameConfig dataclass."""

    def test_constants_are_immutable(self) -> None:
        """Test that constants cannot be modified (frozen=True)."""
        with pytest.raises(FrozenInstanceError):
            DEFAULTS.FIXED_DT = 0.5  # type: ignore[misc]

    def test_singleton_instance_exists(self) -> None:
        """Test that DEFAULTS singleton is properly initialized."""
        assert isinstance(DEFAULTS, GameConfig)

    def test_fixed_timestep_correct(self) -> None:
        """Test that fixed timestep is 60 Hz."""
        assert DEFAULTS.FIXED_DT == 1.0 / 60.0
        assert abs(DEFAULTS.FIXED_DT - 0.01666666666) < 1e-9

    def test_team_indices_correct(self) -> None:
        """Test that team indices are correctly defined."""
        assert DEFAULTS.TEAM_A_INDICES == (0, 1, 2)
        assert DEFAULTS.TEAM_B_INDICES == (3, 4, 5)

    def test_possible_agents_correct(self) -> None:
        """Test that agent IDs are correctly defined."""
        assert len(DEFAULTS.POSSIBLE_AGENTS) == 6
        assert DEFAULTS.POSSIBLE_AGENTS[0] == "penguin_0"
        assert DEFAULTS.POSSIBLE_AGENTS[5] == "penguin_5"

        # Test all agent IDs are unique
        assert len(set(DEFAULTS.POSSIBLE_AGENTS)) == 6

    def test_physics_values_in_valid_range(self) -> None:
        """Test that physics constants are in reasonable ranges."""
        # Ice friction should be low (ice is slippery)
        assert 0.01 <= DEFAULTS.ICE_FRICTION <= 0.05

        # Damping is velocity fraction retained per SECOND (Pymunk convention)
        # Range: 0.5 (heavy friction) to 1.0 (frictionless)
        assert 0.5 <= DEFAULTS.DAMPING <= 1.0

        # Penguin properties
        assert DEFAULTS.PENGUIN_MASS > 0
        assert DEFAULTS.PENGUIN_RADIUS > 0
        assert 0 <= DEFAULTS.ELASTICITY <= 1

        # Arena must be larger than penguin
        assert DEFAULTS.ARENA_HALF_WIDTH > DEFAULTS.PENGUIN_RADIUS

        # Shrink settings
        assert 0 < DEFAULTS.SHRINK_FACTOR < 1
        assert DEFAULTS.SHRINK_INTERVAL > 0
        assert DEFAULTS.MIN_ARENA_HALF_WIDTH > 0
        assert DEFAULTS.MIN_ARENA_HALF_WIDTH < DEFAULTS.ARENA_HALF_WIDTH

    def test_gravity_zero_for_top_down(self) -> None:
        """Test that gravity is (0, 0) for top-down view."""
        assert DEFAULTS.GRAVITY == (0.0, 0.0)

    def test_collision_settings(self) -> None:
        """Test that collision settings are sensible."""
        assert DEFAULTS.COLLISION_ITERATIONS > 0
        assert DEFAULTS.COLLISION_BIAS >= 0

    def test_collision_bias_reasonable(self) -> None:
        """Test that COLLISION_BIAS uses Pymunk default (~0.0018), not 0.2."""
        assert DEFAULTS.COLLISION_BIAS < 0.01
        # Should be approximately pow(1.0 - 0.1, 60.0) ≈ 0.001821
        expected = pow(1.0 - 0.1, 60.0)
        assert abs(DEFAULTS.COLLISION_BIAS - expected) < 1e-10
