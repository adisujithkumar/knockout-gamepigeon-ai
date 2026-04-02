"""Physics constants for the penguin knockout game."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Immutable physics/game constants."""

    # Timestep
    FIXED_DT: float = 1.0 / 60.0  # 60 Hz

    # Ice physics
    ICE_FRICTION: float = 0.03  # Real ice: 0.01-0.05
    DAMPING: float = 0.70  # Pymunk: fraction of velocity retained per SECOND
    # Per-step retention = 0.70^(1/60) ≈ 0.99407
    # Full-power travel ≈ 113 units (56% of arena width)
    # v0 = 400/10 = 40 units/sec (was 13) — fast launch, quick stop

    # Penguin properties
    PENGUIN_MASS: float = 10.0
    PENGUIN_RADIUS: float = 6.0  # ~6% of arena width (was 20.0)
    ELASTICITY: float = 0.8  # Collision bounciness (was 0.5)

    # Arena (square ice block)
    ARENA_HALF_WIDTH: float = 100.0  # Half-width of square arena

    # Shrinking
    SHRINK_FACTOR: float = 0.67  # Arena shrinks by this factor each shrink event
    SHRINK_INTERVAL: int = 5  # Shrink every N rounds
    MIN_ARENA_HALF_WIDTH: float = 30.0  # Floor so arena doesn't vanish

    # Actions
    MAX_LAUNCH_FORCE: float = 400.0  # Fast launch (v0=40), quick stop (D=0.70), ~113-unit travel

    # Pymunk settings
    GRAVITY: tuple[float, float] = (0.0, 0.0)  # Top-down view
    COLLISION_ITERATIONS: int = 10
    COLLISION_BIAS: float = pow(1.0 - 0.1, 60.0)  # BUG FIX #3: was 0.2

    # Settle detection
    SETTLE_SPEED_THRESHOLD: float = 2.0  # Penguins below this speed are "at rest"

    # Game constants
    POSSIBLE_AGENTS: tuple[str, ...] = (
        "penguin_0",
        "penguin_1",
        "penguin_2",
        "penguin_3",
        "penguin_4",
        "penguin_5",
    )
    TEAM_A_INDICES: tuple[int, ...] = (0, 1, 2)
    TEAM_B_INDICES: tuple[int, ...] = (3, 4, 5)


# Default singleton
DEFAULTS = GameConfig()
