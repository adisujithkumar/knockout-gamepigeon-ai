"""Square ice sheet boundary with edge detection."""

from knockout.core.config import DEFAULTS


class IceSheet:
    """Square ice sheet boundary with edge detection.

    The arena is a square from (-half_width, -half_width) to
    (+half_width, +half_width). Supports shrinking over time.
    """

    def __init__(self, half_width: float = DEFAULTS.ARENA_HALF_WIDTH):
        """Initialize ice sheet with given half-width.

        Args:
            half_width: Half-width of the square ice sheet
        """
        self.half_width = half_width
        self.initial_half_width = half_width

    def is_inside(self, position: tuple[float, float]) -> bool:
        """Check if position is within ice sheet boundary.

        Args:
            position: (x, y) coordinates to check

        Returns:
            True if position is inside the square boundary
        """
        x, y = position
        return abs(x) <= self.half_width and abs(y) <= self.half_width

    def distance_to_edge(self, position: tuple[float, float]) -> float:
        """Calculate minimum distance to nearest edge.

        This is used by the heuristic agent with 40% weight in target scoring.

        Args:
            position: (x, y) coordinates

        Returns:
            Distance to nearest edge (positive inside, negative outside)
        """
        x, y = position
        return self.half_width - max(abs(x), abs(y))

    def edge_distances_4dir(self, position: tuple[float, float]) -> dict[str, float]:
        """Calculate distances to edge in 4 cardinal directions.

        For a square arena these are simple axis-aligned distances.

        Args:
            position: (x, y) coordinates

        Returns:
            Dict with keys: "north", "east", "south", "west"
        """
        x, y = position
        return {
            "north": self.half_width - y,
            "east": self.half_width - x,
            "south": self.half_width + y,
            "west": self.half_width + x,
        }

    def get_spawn_positions(self, team_id: int) -> list[tuple[float, float]]:
        """Get starting positions for a team.

        Team 0 (A): Left side (x < 0)
        Team 1 (B): Right side (x > 0)

        Args:
            team_id: 0 for Team A, 1 for Team B

        Returns:
            List of 3 (x, y) spawn positions
        """
        spawn_x = self.half_width * 0.6  # 60% from center
        spawn_y_spread = 10.0  # Scaled for smaller penguins (was 30.0)

        if team_id == 0:
            return [
                (-spawn_x, -spawn_y_spread),
                (-spawn_x, 0.0),
                (-spawn_x, spawn_y_spread),
            ]
        else:
            return [
                (spawn_x, -spawn_y_spread),
                (spawn_x, 0.0),
                (spawn_x, spawn_y_spread),
            ]

    def shrink(self, factor: float = DEFAULTS.SHRINK_FACTOR,
               min_half_width: float = DEFAULTS.MIN_ARENA_HALF_WIDTH) -> float:
        """Shrink the arena by the given factor.

        Args:
            factor: Multiply half_width by this (e.g. 0.67 = shrink by 33%)
            min_half_width: Floor value to prevent arena from vanishing

        Returns:
            Scale factor (new_half_width / old_half_width) for rescaling
            penguin positions.
        """
        old_hw = self.half_width
        self.half_width = max(self.half_width * factor, min_half_width)
        return self.half_width / old_hw
