"""Tests for IceSheet class (square arena)."""

import pytest

from knockout.core.config import DEFAULTS
from knockout.core.ice_sheet import IceSheet


class TestIceSheet:
    """Tests for IceSheet boundary detection (square arena)."""

    def test_creation(self) -> None:
        """Test creating an ice sheet with default half-width."""
        sheet = IceSheet()
        assert sheet.half_width == DEFAULTS.ARENA_HALF_WIDTH
        assert sheet.initial_half_width == DEFAULTS.ARENA_HALF_WIDTH

    def test_creation_custom_half_width(self) -> None:
        """Test creating an ice sheet with custom half-width."""
        sheet = IceSheet(half_width=200.0)
        assert sheet.half_width == 200.0

    def test_is_inside_center(self) -> None:
        """Test that center position is inside."""
        sheet = IceSheet()
        assert sheet.is_inside((0.0, 0.0))

    def test_is_inside_near_edge(self) -> None:
        """Test positions near but inside edge."""
        sheet = IceSheet(half_width=100.0)
        assert sheet.is_inside((99.0, 0.0))
        assert sheet.is_inside((0.0, 99.0))
        assert sheet.is_inside((-99.0, 0.0))
        assert sheet.is_inside((0.0, -99.0))

    def test_is_inside_on_edge(self) -> None:
        """Test positions exactly on edge are inside."""
        sheet = IceSheet(half_width=100.0)
        assert sheet.is_inside((100.0, 0.0))
        assert sheet.is_inside((0.0, 100.0))
        assert sheet.is_inside((100.0, 100.0))  # corner

    def test_is_outside_past_edge(self) -> None:
        """Test positions outside the boundary."""
        sheet = IceSheet(half_width=100.0)
        assert not sheet.is_inside((101.0, 0.0))
        assert not sheet.is_inside((0.0, 101.0))
        assert not sheet.is_inside((200.0, 0.0))

    def test_is_inside_diagonal(self) -> None:
        """Test diagonal positions — square allows corners unlike circle."""
        sheet = IceSheet(half_width=100.0)
        # (70, 70) is inside a square with half_width=100
        assert sheet.is_inside((70.0, 70.0))
        # (99, 99) is inside a square (would be outside a circle)
        assert sheet.is_inside((99.0, 99.0))
        # (101, 0) is outside
        assert not sheet.is_inside((101.0, 0.0))

    def test_distance_to_edge_center(self) -> None:
        """Test distance from center equals half-width."""
        sheet = IceSheet(half_width=100.0)
        distance = sheet.distance_to_edge((0.0, 0.0))
        assert abs(distance - 100.0) < 1e-9

    def test_distance_to_edge_on_edge(self) -> None:
        """Test distance at edge is zero."""
        sheet = IceSheet(half_width=100.0)
        distance = sheet.distance_to_edge((100.0, 0.0))
        assert abs(distance) < 1e-9

    def test_distance_to_edge_inside(self) -> None:
        """Test distance for positions inside."""
        sheet = IceSheet(half_width=100.0)
        # At (50, 0): nearest wall is east at x=100, dist=50
        distance = sheet.distance_to_edge((50.0, 0.0))
        assert abs(distance - 50.0) < 1e-9

    def test_distance_to_edge_corner(self) -> None:
        """Test distance near a corner — min of two wall distances."""
        sheet = IceSheet(half_width=100.0)
        # At (90, 80): nearest wall is east (100-90=10) or north (100-80=20)
        distance = sheet.distance_to_edge((90.0, 80.0))
        assert abs(distance - 10.0) < 1e-9

    def test_distance_to_edge_outside(self) -> None:
        """Test distance for positions outside (negative)."""
        sheet = IceSheet(half_width=100.0)
        distance = sheet.distance_to_edge((110.0, 0.0))
        assert abs(distance + 10.0) < 1e-9  # -10.0

    def test_edge_distances_4dir_center(self) -> None:
        """Test directional distances from center."""
        sheet = IceSheet(half_width=100.0)
        distances = sheet.edge_distances_4dir((0.0, 0.0))

        assert abs(distances["north"] - 100.0) < 1e-6
        assert abs(distances["east"] - 100.0) < 1e-6
        assert abs(distances["south"] - 100.0) < 1e-6
        assert abs(distances["west"] - 100.0) < 1e-6

    def test_edge_distances_4dir_east_side(self) -> None:
        """Test directional distances from east side."""
        sheet = IceSheet(half_width=100.0)
        distances = sheet.edge_distances_4dir((50.0, 0.0))

        assert distances["east"] < distances["west"]
        assert abs(distances["north"] - distances["south"]) < 1e-6

    def test_edge_distances_4dir_accuracy(self) -> None:
        """Test that directional distances are accurate for square."""
        sheet = IceSheet(half_width=100.0)
        distances = sheet.edge_distances_4dir((50.0, 0.0))

        # East: hw - x = 100 - 50 = 50
        assert abs(distances["east"] - 50.0) < 1e-9
        # West: hw + x = 100 + 50 = 150
        assert abs(distances["west"] - 150.0) < 1e-9
        # North: hw - y = 100 - 0 = 100
        assert abs(distances["north"] - 100.0) < 1e-9
        # South: hw + y = 100 + 0 = 100
        assert abs(distances["south"] - 100.0) < 1e-9

    def test_edge_distances_4dir_off_center(self) -> None:
        """Test directional distances from an off-center position."""
        sheet = IceSheet(half_width=100.0)
        distances = sheet.edge_distances_4dir((30.0, -20.0))

        assert abs(distances["east"] - 70.0) < 1e-9   # 100 - 30
        assert abs(distances["west"] - 130.0) < 1e-9   # 100 + 30
        assert abs(distances["north"] - 120.0) < 1e-9  # 100 - (-20)
        assert abs(distances["south"] - 80.0) < 1e-9   # 100 + (-20)

    def test_get_spawn_positions_team_a(self) -> None:
        """Test spawn positions for Team A (left side)."""
        sheet = IceSheet(half_width=100.0)
        positions = sheet.get_spawn_positions(team_id=0)

        assert len(positions) == 3

        # All positions should be on left side (negative x)
        for x, y in positions:
            assert x < 0

        # All positions should be inside the ice sheet
        for pos in positions:
            assert sheet.is_inside(pos)

        # Y coordinates should be spread out
        y_coords = [y for _, y in positions]
        assert y_coords == [-10.0, 0.0, 10.0]

    def test_get_spawn_positions_team_b(self) -> None:
        """Test spawn positions for Team B (right side)."""
        sheet = IceSheet(half_width=100.0)
        positions = sheet.get_spawn_positions(team_id=1)

        assert len(positions) == 3

        # All positions should be on right side (positive x)
        for x, y in positions:
            assert x > 0

        # All positions should be inside the ice sheet
        for pos in positions:
            assert sheet.is_inside(pos)

        # Y coordinates should be spread out
        y_coords = [y for _, y in positions]
        assert y_coords == [-10.0, 0.0, 10.0]

    def test_spawn_positions_symmetric(self) -> None:
        """Test that Team A and B spawn positions are symmetric."""
        sheet = IceSheet(half_width=100.0)
        positions_a = sheet.get_spawn_positions(team_id=0)
        positions_b = sheet.get_spawn_positions(team_id=1)

        for (xa, ya), (xb, yb) in zip(positions_a, positions_b):
            assert abs(xa + xb) < 1e-9
            assert abs(ya - yb) < 1e-9

    def test_shrink(self) -> None:
        """Test that shrinking reduces half_width correctly."""
        sheet = IceSheet(half_width=100.0)
        scale = sheet.shrink(0.67, 30.0)
        assert abs(sheet.half_width - 67.0) < 1e-9
        assert sheet.initial_half_width == 100.0  # unchanged
        assert abs(scale - 0.67) < 1e-9

    def test_shrink_returns_scale_factor(self) -> None:
        """Test that shrink returns correct scale factor."""
        sheet = IceSheet(half_width=100.0)
        scale = sheet.shrink(0.5, 30.0)
        assert abs(scale - 0.5) < 1e-9

        # When clamped to minimum, scale should reflect actual ratio
        sheet2 = IceSheet(half_width=40.0)
        scale2 = sheet2.shrink(0.5, 30.0)
        assert abs(scale2 - 30.0 / 40.0) < 1e-9

    def test_shrink_multiple(self) -> None:
        """Test multiple shrinks compound correctly."""
        sheet = IceSheet(half_width=100.0)
        scale1 = sheet.shrink(0.67, 30.0)
        assert abs(sheet.half_width - 67.0) < 1e-9
        assert abs(scale1 - 0.67) < 1e-9
        scale2 = sheet.shrink(0.67, 30.0)
        assert abs(sheet.half_width - 44.89) < 1e-9
        assert abs(scale2 - 44.89 / 67.0) < 1e-9

    def test_shrink_respects_minimum(self) -> None:
        """Test that shrinking respects minimum half-width."""
        sheet = IceSheet(half_width=100.0)
        # Shrink aggressively
        for _ in range(20):
            sheet.shrink(0.5, 30.0)
        assert sheet.half_width == 30.0

    def test_shrink_at_minimum_returns_one(self) -> None:
        """Test that shrinking at minimum returns scale=1.0 (no change)."""
        sheet = IceSheet(half_width=30.0)
        scale = sheet.shrink(0.5, 30.0)
        assert abs(scale - 1.0) < 1e-9

    def test_is_inside_after_shrink(self) -> None:
        """Test boundary check uses current (shrunk) half_width."""
        sheet = IceSheet(half_width=100.0)
        assert sheet.is_inside((80.0, 0.0))
        sheet.shrink(0.5, 30.0)  # half_width = 50
        assert not sheet.is_inside((80.0, 0.0))
        assert sheet.is_inside((49.0, 0.0))
