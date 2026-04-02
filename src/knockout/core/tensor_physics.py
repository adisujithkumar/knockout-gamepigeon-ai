"""GPU-accelerated batched physics engine using PyTorch tensors.

Replaces Pymunk for RL training. All operations are vectorized across
the batch dimension -- no Python loops over environments.
"""

from __future__ import annotations

import math

import torch

from knockout.core.config import GameConfig, DEFAULTS


# ---- Standalone functions compilable by torch.compile -------------------

def _resolve_collisions_fn(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    alive: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    diameter: float,
    elasticity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve circle-circle elastic collisions -- one pass, no sync."""
    B = positions.shape[0]
    eps = 1e-8

    pos_i = positions[:, pair_i]  # (B, 15, 2)
    pos_j = positions[:, pair_j]
    vel_i = velocities[:, pair_i]
    vel_j = velocities[:, pair_j]

    both_alive = alive[:, pair_i] & alive[:, pair_j]  # (B, 15)

    delta = pos_j - pos_i  # (B, 15, 2)
    dist = torch.sqrt((delta * delta).sum(dim=-1) + eps)  # (B, 15)

    colliding = (dist < diameter) & both_alive

    n = delta / dist.unsqueeze(-1)  # collision normal (B, 15, 2)

    v_rel = vel_i - vel_j
    v_n = (v_rel * n).sum(dim=-1)  # relative velocity along normal (B, 15)

    resolve_mask = colliding & (v_n > 0)  # approaching pairs

    j_mag = ((1.0 + elasticity) * v_n / 2.0) * resolve_mask.float()

    impulse_vec = j_mag.unsqueeze(-1) * n  # (B, 15, 2)

    pi_exp = pair_i.unsqueeze(0).unsqueeze(-1).expand(B, -1, 2)
    pj_exp = pair_j.unsqueeze(0).unsqueeze(-1).expand(B, -1, 2)

    dv = torch.zeros_like(velocities)
    dv.scatter_add_(1, pi_exp, -impulse_vec)
    dv.scatter_add_(1, pj_exp, impulse_vec)
    velocities = velocities + dv

    overlap = ((diameter - dist) * colliding.float()).clamp(min=0.0)
    sep = (overlap / 2.0).unsqueeze(-1) * n

    dp = torch.zeros_like(positions)
    dp.scatter_add_(1, pi_exp, -sep)
    dp.scatter_add_(1, pj_exp, sep)
    positions = positions + dp

    return positions, velocities


def _physics_step_fn(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    alive: torch.Tensor,
    arena_hw: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    dt: float,
    damping: float,
    diameter: float,
    elasticity: float,
    collision_passes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single physics step: integrate, collide, boundary check."""
    alive_f = alive.unsqueeze(-1).float()

    # Integrate
    positions = positions + velocities * (dt * alive_f)
    # Damping
    velocities = velocities * (damping * alive_f)

    # Collision passes
    for _ in range(collision_passes):
        positions, velocities = _resolve_collisions_fn(
            positions, velocities, alive, pair_i, pair_j, diameter, elasticity,
        )

    # Boundary check
    hw = arena_hw.unsqueeze(-1)  # (B, 1)
    out_of_bounds = (positions[..., 0].abs() > hw) | (positions[..., 1].abs() > hw)
    alive = alive & ~out_of_bounds
    velocities = velocities * alive.unsqueeze(-1).float()

    return positions, velocities, alive


def _step_chunk_fn(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    alive: torch.Tensor,
    arena_hw: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    dt: float,
    damping: float,
    diameter: float,
    elasticity: float,
    collision_passes: int,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run *n_steps* physics steps as a single compiled graph.

    When traced by ``torch.compile`` the for-loop is unrolled, allowing
    the compiler to fuse kernels across iterations and dramatically
    reduce launch overhead on GPU.
    """
    for _ in range(n_steps):
        positions, velocities, alive = _physics_step_fn(
            positions, velocities, alive, arena_hw,
            pair_i, pair_j, dt, damping, diameter, elasticity,
            collision_passes,
        )
    return positions, velocities, alive


class TensorPhysicsEngine:
    """GPU-accelerated batched physics engine using PyTorch tensors.

    State representation:
        positions:  (batch, 6, 2)  -- x, y per penguin
        velocities: (batch, 6, 2)  -- vx, vy per penguin
        alive:      (batch, 6)     -- bool, whether each penguin is on the ice

    All operations are vectorized across the batch dimension.
    """

    # Pre-computed pair indices for 6 penguins (15 unique pairs)
    _PAIR_I = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
    _PAIR_J = [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]

    def __init__(
        self, config: GameConfig = DEFAULTS, device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.config = config

        # Precompute per-step damping.
        # Pymunk convention: DAMPING is fraction of velocity retained per *second*.
        # Per-step retention = DAMPING^(dt)  where dt = FIXED_DT = 1/60.
        self.damping_per_step: float = config.DAMPING ** config.FIXED_DT

        self.dt: float = config.FIXED_DT
        self.mass: float = config.PENGUIN_MASS
        self.radius: float = config.PENGUIN_RADIUS
        self.elasticity: float = config.ELASTICITY
        self.collision_diameter: float = 2.0 * config.PENGUIN_RADIUS
        self.settle_threshold: float = config.SETTLE_SPEED_THRESHOLD
        self._collision_passes: int = 3

        # Register pair indices as tensors once.
        self.pair_i = torch.tensor(self._PAIR_I, device=self.device, dtype=torch.long)
        self.pair_j = torch.tensor(self._PAIR_J, device=self.device, dtype=torch.long)

        # Compiled chunk functions keyed by chunk size (lazy init).
        self._compiled_chunks: dict[int, object] = {}

        # Spawn positions (same as IceSheet.get_spawn_positions)
        hw = config.ARENA_HALF_WIDTH
        spawn_x = hw * 0.6
        spread = 10.0
        # Team A: indices 0-2, Team B: indices 3-5
        self._spawn_positions = torch.tensor(
            [
                [-spawn_x, -spread],
                [-spawn_x, 0.0],
                [-spawn_x, spread],
                [spawn_x, -spread],
                [spawn_x, 0.0],
                [spawn_x, spread],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize *batch_size* environments.

        Returns:
            positions:  (batch, 6, 2)
            velocities: (batch, 6, 2)
            alive:      (batch, 6)  bool
        """
        positions = self._spawn_positions.unsqueeze(0).expand(batch_size, -1, -1).clone()
        velocities = torch.zeros(batch_size, 6, 2, dtype=torch.float32, device=self.device)
        alive = torch.ones(batch_size, 6, dtype=torch.bool, device=self.device)
        return positions, velocities, alive

    # ------------------------------------------------------------------
    # Impulse application
    # ------------------------------------------------------------------

    def apply_impulses(
        self,
        velocities: torch.Tensor,
        actions: torch.Tensor,
        alive: torch.Tensor,
    ) -> torch.Tensor:
        """Apply impulses to alive penguins.

        Args:
            velocities: (batch, 6, 2)
            actions:    (batch, 6, 2) -- [angle_degrees, power] per penguin
            alive:      (batch, 6) bool mask

        Returns:
            Updated velocities: (batch, 6, 2)
        """
        angle_deg = actions[..., 0].clamp(0.0, 360.0)
        power = actions[..., 1].clamp(0.0, self.config.MAX_LAUNCH_FORCE)

        angle_rad = angle_deg * (math.pi / 180.0)

        impulse_x = torch.cos(angle_rad) * power
        impulse_y = torch.sin(angle_rad) * power
        impulse = torch.stack([impulse_x, impulse_y], dim=-1)  # (B, 6, 2)

        mask = alive.unsqueeze(-1).float()  # (B, 6, 1)
        return velocities + (impulse / self.mass) * mask

    # ------------------------------------------------------------------
    # Collision resolution
    # ------------------------------------------------------------------

    def resolve_collisions(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        alive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve circle-circle elastic collisions in one pass.

        Delegates to the module-level function (which is compilable
        by ``torch.compile``).
        """
        return _resolve_collisions_fn(
            positions, velocities, alive,
            self.pair_i, self.pair_j,
            self.collision_diameter, self.elasticity,
        )

    # ------------------------------------------------------------------
    # Compiled chunk accessor
    # ------------------------------------------------------------------

    def _get_compiled_chunk(self, chunk_size: int):
        """Return compiled *_step_chunk_fn* for the given chunk size.

        On GPU, ``torch.compile`` unrolls the inner for-loop at trace
        time (since *n_steps* is a compile-time constant) and fuses
        kernels across iterations, cutting per-step launch overhead to
        nearly zero.  On CPU, compilation has negligible benefit so
        the plain function is used.
        """
        fn = self._compiled_chunks.get(chunk_size)
        if fn is not None:
            return fn

        if self.device.type != "cpu":
            try:
                fn = torch.compile(_step_chunk_fn, dynamic=True)
            except Exception:
                fn = _step_chunk_fn
        else:
            fn = _step_chunk_fn

        self._compiled_chunks[chunk_size] = fn
        return fn

    # ------------------------------------------------------------------
    # Single physics step
    # ------------------------------------------------------------------

    def step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        alive: torch.Tensor,
        arena_half_width: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One physics step: integrate, collide, check boundaries.

        Args:
            positions:        (batch, 6, 2)
            velocities:       (batch, 6, 2)
            alive:            (batch, 6)
            arena_half_width: (batch,) or scalar

        Returns:
            (new_positions, new_velocities, new_alive)
        """
        # Normalise arena_half_width to a (B,) tensor so that the compiled
        # graph sees a consistent signature.
        if isinstance(arena_half_width, (int, float)):
            hw_t = torch.full(
                (positions.shape[0],), arena_half_width,
                device=positions.device, dtype=torch.float32,
            )
        else:
            hw_t = arena_half_width

        return _physics_step_fn(
            positions, velocities, alive, hw_t,
            self.pair_i, self.pair_j,
            self.dt, self.damping_per_step,
            self.collision_diameter, self.elasticity,
            self._collision_passes,
        )

    # ------------------------------------------------------------------
    # Multi-step (reduced Python-loop overhead)
    # ------------------------------------------------------------------

    def step_n(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        alive: torch.Tensor,
        arena_half_width: torch.Tensor | float,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run *n* physics steps without intermediate settle checks.

        On GPU, this delegates to a ``torch.compile``-d chunk function
        that fuses all *n* iterations into a single compiled graph.
        """
        if isinstance(arena_half_width, (int, float)):
            hw_t = torch.full(
                (positions.shape[0],), arena_half_width,
                device=positions.device, dtype=torch.float32,
            )
        else:
            hw_t = arena_half_width

        chunk_fn = self._get_compiled_chunk(n)
        return chunk_fn(
            positions, velocities, alive, hw_t,
            self.pair_i, self.pair_j,
            self.dt, self.damping_per_step,
            self.collision_diameter, self.elasticity,
            self._collision_passes, n,
        )

    # ------------------------------------------------------------------
    # Step until settled
    # ------------------------------------------------------------------

    def step_until_settled(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        alive: torch.Tensor,
        arena_half_width: torch.Tensor | float,
        max_steps: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step physics until all penguins settle or are eliminated.

        Already-settled environments are frozen so extra iterations do
        not perturb them.  To reduce Python-loop overhead, settle is
        checked every ``check_interval`` sub-steps.  On GPU, the
        ``active.any()`` sync that breaks the loop is performed only
        every ``sync_interval`` checks to amortise the cost.

        Returns:
            (positions, velocities, alive, steps_taken)
            steps_taken: (batch,) int tensor
        """
        B = positions.shape[0]
        device = self.device
        check_interval = 10  # check settle every N sub-steps
        # On GPU, only sync to host every *sync_interval* settle-checks
        # to amortise the ~200 us cost of device-to-host transfer.
        is_gpu = device.type != "cpu"
        sync_interval = 5 if is_gpu else 1
        steps_taken = torch.zeros(B, dtype=torch.long, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)

        # Frozen snapshots for settled envs
        frozen_pos = positions.clone()
        frozen_vel = velocities.clone()
        frozen_alive = alive.clone()

        total_stepped = 0
        checks_since_sync = 0
        while total_stepped < max_steps:
            # Determine how many sub-steps to run before next check
            chunk = min(check_interval, max_steps - total_stepped)

            positions, velocities, alive = self.step_n(
                positions, velocities, alive, arena_half_width, chunk,
            )
            steps_taken += chunk * active.long()
            total_stepped += chunk

            # Restore frozen state for already-settled environments
            # (uses torch.where -- no sync needed)
            settled_mask = ~active
            sm62 = settled_mask.unsqueeze(-1).unsqueeze(-1)
            sm6 = settled_mask.unsqueeze(-1)
            positions = torch.where(sm62, frozen_pos, positions)
            velocities = torch.where(sm62, frozen_vel, velocities)
            alive = torch.where(sm6, frozen_alive, alive)

            # Check settle: max alive-penguin speed per env
            speeds_sq = (velocities * velocities).sum(dim=-1)  # (B, 6)
            max_speed_sq = torch.where(
                alive, speeds_sq, torch.zeros_like(speeds_sq)
            ).max(dim=-1).values
            newly_settled = active & (max_speed_sq < self.settle_threshold ** 2)

            # Snapshot newly-settled envs (no sync -- purely GPU ops)
            ns62 = newly_settled.unsqueeze(-1).unsqueeze(-1)
            ns6 = newly_settled.unsqueeze(-1)
            frozen_pos = torch.where(ns62, positions, frozen_pos)
            frozen_vel = torch.where(ns62, velocities, frozen_vel)
            frozen_alive = torch.where(ns6, alive, frozen_alive)

            active = active & ~newly_settled

            # Only sync to host periodically to check termination
            checks_since_sync += 1
            if checks_since_sync >= sync_interval:
                checks_since_sync = 0
                if not active.any():
                    break

        return positions, velocities, alive, steps_taken

    # ------------------------------------------------------------------
    # Rescale (after arena shrink)
    # ------------------------------------------------------------------

    def rescale_penguins(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        alive: torch.Tensor,
        scale_factor: torch.Tensor | float,
        new_half_width: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rescale alive penguin positions and velocities after arena shrink.

        Positions are clamped to stay inside the new arena boundary.
        """
        alive_b = alive.unsqueeze(-1)  # (B, 6, 1) bool

        if isinstance(scale_factor, (int, float)):
            sf = scale_factor
        else:
            sf = scale_factor.unsqueeze(-1).unsqueeze(-1)

        if isinstance(new_half_width, (int, float)):
            hw = new_half_width
        else:
            hw = new_half_width.unsqueeze(-1).unsqueeze(-1)

        new_pos = torch.where(alive_b, positions * sf, positions)
        new_vel = torch.where(alive_b, velocities * sf, velocities)

        margin_f = 0.005
        if isinstance(hw, (int, float)):
            limit = hw - max(0.01, hw * margin_f)
        else:
            limit = hw - torch.clamp(hw * margin_f, min=0.01)

        new_pos = torch.where(alive_b, new_pos.clamp(-limit, limit), new_pos)

        return new_pos, new_vel
