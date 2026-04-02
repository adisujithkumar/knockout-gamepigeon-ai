"""Tests for the GPU-accelerated batched tensor physics engine."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.tensor_physics import TensorPhysicsEngine


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def engine() -> TensorPhysicsEngine:
    """CPU tensor physics engine with default config."""
    return TensorPhysicsEngine(config=DEFAULTS, device="cpu")


@pytest.fixture
def batch4(engine: TensorPhysicsEngine):
    """4-environment reset state."""
    return engine.reset(4)


# ======================================================================
# Reset
# ======================================================================

class TestReset:
    def test_reset_shapes(self, engine: TensorPhysicsEngine) -> None:
        pos, vel, alive = engine.reset(8)
        assert pos.shape == (8, 6, 2)
        assert vel.shape == (8, 6, 2)
        assert alive.shape == (8, 6)
        assert alive.dtype == torch.bool

    def test_reset_positions(self, engine: TensorPhysicsEngine) -> None:
        """Spawn positions must match IceSheet.get_spawn_positions."""
        pos, _, _ = engine.reset(1)
        hw = DEFAULTS.ARENA_HALF_WIDTH
        spawn_x = hw * 0.6
        spread = 10.0

        expected = torch.tensor([[
            [-spawn_x, -spread],
            [-spawn_x, 0.0],
            [-spawn_x, spread],
            [spawn_x, -spread],
            [spawn_x, 0.0],
            [spawn_x, spread],
        ]])
        assert torch.allclose(pos, expected)

    def test_reset_velocities_zero(self, engine: TensorPhysicsEngine) -> None:
        _, vel, _ = engine.reset(3)
        assert (vel == 0).all()

    def test_reset_all_alive(self, engine: TensorPhysicsEngine) -> None:
        _, _, alive = engine.reset(5)
        assert alive.all()


# ======================================================================
# Impulse application
# ======================================================================

class TestImpulses:
    def test_impulse_right(self, engine: TensorPhysicsEngine) -> None:
        """Angle=0 (right), power=400 => vx += 400/10 = 40."""
        _, vel, alive = engine.reset(1)
        actions = torch.zeros(1, 6, 2)
        actions[0, 0, 0] = 0.0   # angle
        actions[0, 0, 1] = 400.0  # power
        vel = engine.apply_impulses(vel, actions, alive)
        assert abs(vel[0, 0, 0].item() - 40.0) < 1e-5
        assert abs(vel[0, 0, 1].item()) < 1e-5

    def test_impulse_up(self, engine: TensorPhysicsEngine) -> None:
        """Angle=90 => vy += power/mass."""
        _, vel, alive = engine.reset(1)
        actions = torch.zeros(1, 6, 2)
        actions[0, 1, 0] = 90.0
        actions[0, 1, 1] = 200.0
        vel = engine.apply_impulses(vel, actions, alive)
        assert abs(vel[0, 1, 0].item()) < 1e-4
        assert abs(vel[0, 1, 1].item() - 20.0) < 1e-4

    def test_impulse_dead_penguin_ignored(self, engine: TensorPhysicsEngine) -> None:
        """Dead penguins should not gain velocity."""
        _, vel, alive = engine.reset(1)
        alive[0, 2] = False
        actions = torch.zeros(1, 6, 2)
        actions[0, 2, 0] = 45.0
        actions[0, 2, 1] = 400.0
        vel = engine.apply_impulses(vel, actions, alive)
        assert (vel[0, 2] == 0).all()

    def test_impulse_clipping(self, engine: TensorPhysicsEngine) -> None:
        """Power > MAX_LAUNCH_FORCE should be clipped."""
        _, vel, alive = engine.reset(1)
        actions = torch.zeros(1, 6, 2)
        actions[0, 0, 0] = 0.0
        actions[0, 0, 1] = 9999.0  # way over max
        vel = engine.apply_impulses(vel, actions, alive)
        expected_v = DEFAULTS.MAX_LAUNCH_FORCE / DEFAULTS.PENGUIN_MASS
        assert abs(vel[0, 0, 0].item() - expected_v) < 1e-4


# ======================================================================
# Damping
# ======================================================================

class TestDamping:
    def test_damping_one_step(self, engine: TensorPhysicsEngine) -> None:
        """After one step, velocity should be multiplied by damping_per_step."""
        pos, vel, alive = engine.reset(1)
        vel[0, 0, 0] = 40.0  # vx = 40
        pos2, vel2, alive2 = engine.step(pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH)

        expected_v = 40.0 * engine.damping_per_step
        # The step also integrates position, but velocity should be damped
        assert abs(vel2[0, 0, 0].item() - expected_v) < 0.5  # tolerance for collision passes

    def test_damping_multiple_steps(self, engine: TensorPhysicsEngine) -> None:
        """After N steps, velocity ~ v0 * damping^N."""
        pos, vel, alive = engine.reset(1)
        # Place penguin at center to avoid boundary issues
        pos[0, 0] = torch.tensor([0.0, 0.0])
        vel[0, 0, 0] = 40.0
        alive_mask = torch.zeros(1, 6, dtype=torch.bool)
        alive_mask[0, 0] = True  # only one penguin alive — no collisions

        for _ in range(60):
            pos, vel, alive_mask = engine.step(pos, vel, alive_mask, DEFAULTS.ARENA_HALF_WIDTH)

        # After 60 steps (1 second), velocity should be ~ v0 * DAMPING
        expected = 40.0 * DEFAULTS.DAMPING
        assert abs(vel[0, 0, 0].item() - expected) < 1.0


# ======================================================================
# Boundary elimination
# ======================================================================

class TestBoundary:
    def test_boundary_elimination(self, engine: TensorPhysicsEngine) -> None:
        """Penguin at (101, 0) with hw=100 should be eliminated."""
        pos, vel, alive = engine.reset(1)
        pos[0, 0] = torch.tensor([101.0, 0.0])
        _, _, alive2 = engine.step(pos, vel, alive, 100.0)
        assert not alive2[0, 0].item()

    def test_boundary_inside(self, engine: TensorPhysicsEngine) -> None:
        """Penguin at (99, 0) with hw=100 should stay alive."""
        pos, vel, alive = engine.reset(1)
        pos[0, 0] = torch.tensor([99.0, 0.0])
        _, _, alive2 = engine.step(pos, vel, alive, 100.0)
        assert alive2[0, 0].item()

    def test_boundary_y_axis(self, engine: TensorPhysicsEngine) -> None:
        """Penguin outside on y axis should be eliminated."""
        pos, vel, alive = engine.reset(1)
        pos[0, 0] = torch.tensor([0.0, 105.0])
        _, _, alive2 = engine.step(pos, vel, alive, 100.0)
        assert not alive2[0, 0].item()

    def test_boundary_per_env_hw(self, engine: TensorPhysicsEngine) -> None:
        """Per-environment half-widths should be respected."""
        pos, vel, alive = engine.reset(2)
        pos[0, 0] = torch.tensor([55.0, 0.0])
        pos[1, 0] = torch.tensor([55.0, 0.0])
        hw = torch.tensor([50.0, 60.0])
        _, _, alive2 = engine.step(pos, vel, alive, hw)
        assert not alive2[0, 0].item()  # 55 > 50 => eliminated
        assert alive2[1, 0].item()      # 55 < 60 => alive


# ======================================================================
# Collision
# ======================================================================

class TestCollision:
    def test_collision_detection(self, engine: TensorPhysicsEngine) -> None:
        """Two penguins heading toward each other should exchange velocity."""
        pos, vel, alive = engine.reset(1)
        # Kill all but two penguins to isolate the collision
        alive[0] = False
        alive[0, 0] = True
        alive[0, 1] = True

        r = DEFAULTS.PENGUIN_RADIUS
        # Place them just outside collision range, moving towards each other
        pos[0, 0] = torch.tensor([0.0, 0.0])
        pos[0, 1] = torch.tensor([2 * r + 1.0, 0.0])  # slightly apart
        vel[0, 0] = torch.tensor([20.0, 0.0])   # moving right
        vel[0, 1] = torch.tensor([-20.0, 0.0])  # moving left

        # Step enough for them to collide
        for _ in range(10):
            pos, vel, alive = engine.step(pos, vel, alive, 200.0)

        # After elastic collision of equal masses head-on, they should
        # have roughly swapped velocities (modulo damping and elasticity)
        # The key check: penguin 0 should now move leftward
        assert vel[0, 0, 0].item() < 0  # reversed direction
        assert vel[0, 1, 0].item() > 0  # reversed direction

    def test_collision_elastic(self, engine: TensorPhysicsEngine) -> None:
        """Check elasticity reduces relative speed."""
        pos, vel, alive = engine.reset(1)
        alive[0] = False
        alive[0, 0] = True
        alive[0, 1] = True

        r = DEFAULTS.PENGUIN_RADIUS
        # Place overlapping so collision happens immediately
        pos[0, 0] = torch.tensor([0.0, 0.0])
        pos[0, 1] = torch.tensor([2 * r - 1.0, 0.0])  # overlapping
        vel[0, 0] = torch.tensor([20.0, 0.0])
        vel[0, 1] = torch.tensor([-20.0, 0.0])

        # One step resolves the collision
        pos2, vel2, alive2 = engine.step(pos, vel, alive, 200.0)

        # With elasticity < 1, relative speed should decrease
        rel_speed_before = 40.0  # |20 - (-20)|
        rel_speed_after = abs(vel2[0, 0, 0].item() - vel2[0, 1, 0].item())
        assert rel_speed_after < rel_speed_before

    def test_collision_only_alive(self, engine: TensorPhysicsEngine) -> None:
        """Dead penguins should not participate in collisions."""
        pos, vel, alive = engine.reset(1)
        alive[0] = False
        alive[0, 0] = True
        # penguin 1 is dead but overlapping
        alive[0, 1] = False
        pos[0, 0] = torch.tensor([0.0, 0.0])
        pos[0, 1] = torch.tensor([1.0, 0.0])  # overlapping
        vel[0, 0] = torch.tensor([10.0, 0.0])

        vel_before = vel[0, 0, 0].item()
        pos2, vel2, alive2 = engine.step(pos, vel, alive, 200.0)

        # Velocity should only change due to damping, not collision
        expected = vel_before * engine.damping_per_step
        assert abs(vel2[0, 0, 0].item() - expected) < 0.5


# ======================================================================
# Settle detection
# ======================================================================

class TestSettle:
    def test_stationary_is_settled(self, engine: TensorPhysicsEngine) -> None:
        """Penguins with zero velocity settle on the first check."""
        pos, vel, alive = engine.reset(1)
        # All velocities are zero from reset -- should settle on first
        # settle check (after one chunk of sub-steps).
        pos2, vel2, alive2, steps = engine.step_until_settled(
            pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH
        )
        # Steps taken equals the check_interval (first chunk settles it)
        assert steps[0].item() <= 10

    def test_step_until_settled_returns(self, engine: TensorPhysicsEngine) -> None:
        """Moving penguin should eventually settle due to damping."""
        pos, vel, alive = engine.reset(1)
        pos[0, 0] = torch.tensor([0.0, 0.0])
        vel[0, 0] = torch.tensor([30.0, 0.0])
        alive_only = torch.zeros(1, 6, dtype=torch.bool)
        alive_only[0, 0] = True

        pos2, vel2, alive2, steps = engine.step_until_settled(
            pos, vel, alive_only, DEFAULTS.ARENA_HALF_WIDTH
        )
        speed = torch.sqrt((vel2[0, 0] ** 2).sum()).item()
        assert speed < DEFAULTS.SETTLE_SPEED_THRESHOLD
        assert steps[0].item() > 1


# ======================================================================
# Batching correctness
# ======================================================================

class TestBatching:
    def test_batch_independence(self, engine: TensorPhysicsEngine) -> None:
        """Environment 0's physics must not affect environment 1."""
        # Run env 1 alone (no impulse) as reference
        pos_ref, vel_ref, alive_ref = engine.reset(1)
        pos_ref_out, vel_ref_out, alive_ref_out, _ = engine.step_until_settled(
            pos_ref, vel_ref, alive_ref, DEFAULTS.ARENA_HALF_WIDTH
        )

        # Run both envs together: env 0 gets impulse, env 1 does not
        pos, vel, alive = engine.reset(2)
        actions = torch.zeros(2, 6, 2)
        actions[0, 0, 0] = 0.0
        actions[0, 0, 1] = 400.0
        vel = engine.apply_impulses(vel, actions, alive)

        pos2, vel2, alive2, _ = engine.step_until_settled(
            pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH
        )

        # Env 1 output should match the reference (env run in isolation).
        # step_until_settled freezes settled envs, so there should be
        # no cross-env contamination.
        assert torch.allclose(pos2[1], pos_ref_out[0], atol=0.01)

    def test_batch_shapes(self, engine: TensorPhysicsEngine) -> None:
        """All outputs must have correct batch dimensions."""
        B = 16
        pos, vel, alive = engine.reset(B)
        pos2, vel2, alive2 = engine.step(pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH)
        assert pos2.shape == (B, 6, 2)
        assert vel2.shape == (B, 6, 2)
        assert alive2.shape == (B, 6)

    def test_large_batch(self, engine: TensorPhysicsEngine) -> None:
        """1000 environments should run without error."""
        pos, vel, alive = engine.reset(1000)
        actions = torch.rand(1000, 6, 2) * torch.tensor([360.0, 400.0])
        vel = engine.apply_impulses(vel, actions, alive)
        pos2, vel2, alive2, steps = engine.step_until_settled(
            pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH
        )
        assert pos2.shape == (1000, 6, 2)
        assert alive2.shape == (1000, 6)


# ======================================================================
# Device tests
# ======================================================================

class TestDevice:
    def test_cpu_device(self) -> None:
        eng = TensorPhysicsEngine(device="cpu")
        pos, vel, alive = eng.reset(2)
        assert pos.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_gpu_device(self) -> None:
        eng = TensorPhysicsEngine(device="cuda")
        pos, vel, alive = eng.reset(2)
        assert pos.device.type == "cuda"
        pos2, vel2, alive2 = eng.step(pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH)
        assert pos2.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_cpu_gpu_consistency(self) -> None:
        """Same initial state should give similar results on CPU and GPU."""
        eng_cpu = TensorPhysicsEngine(device="cpu")
        eng_gpu = TensorPhysicsEngine(device="cuda")

        pos_c, vel_c, alive_c = eng_cpu.reset(4)
        pos_g, vel_g, alive_g = eng_gpu.reset(4)

        actions = torch.rand(4, 6, 2) * torch.tensor([360.0, 400.0])
        vel_c = eng_cpu.apply_impulses(vel_c, actions, alive_c)
        vel_g = eng_gpu.apply_impulses(vel_g, actions.to("cuda"), alive_g)

        pos_c2, vel_c2, alive_c2, _ = eng_cpu.step_until_settled(
            pos_c, vel_c, alive_c, DEFAULTS.ARENA_HALF_WIDTH
        )
        pos_g2, vel_g2, alive_g2, _ = eng_gpu.step_until_settled(
            pos_g, vel_g, alive_g, DEFAULTS.ARENA_HALF_WIDTH
        )

        assert torch.allclose(pos_c2, pos_g2.cpu(), atol=1.0)
        assert (alive_c2 == alive_g2.cpu()).all()


# ======================================================================
# Rescale
# ======================================================================

class TestRescale:
    def test_rescale_positions(self, engine: TensorPhysicsEngine) -> None:
        pos, vel, alive = engine.reset(1)
        orig_pos = pos.clone()
        scale = 0.67
        new_hw = DEFAULTS.ARENA_HALF_WIDTH * scale
        pos2, vel2 = engine.rescale_penguins(pos, vel, alive, scale, new_hw)
        # Positions should be approximately scaled
        assert torch.allclose(pos2[0, 0], orig_pos[0, 0] * scale, atol=1.0)

    def test_rescale_clamps(self, engine: TensorPhysicsEngine) -> None:
        pos, vel, alive = engine.reset(1)
        pos[0, 0] = torch.tensor([95.0, 95.0])
        scale = 0.5
        new_hw = 50.0
        pos2, _ = engine.rescale_penguins(pos, vel, alive, scale, new_hw)
        # After rescale (95*0.5=47.5) should be inside 50-margin
        assert pos2[0, 0, 0].item() < new_hw
        assert pos2[0, 0, 1].item() < new_hw


# ======================================================================
# Comparison with Pymunk
# ======================================================================

class TestVsPymunk:
    """Compare tensor engine against the Pymunk PhysicsEngine."""

    def _run_pymunk_impulse(self, angle: float, power: float):
        """Run a single impulse in Pymunk and return final position."""
        from knockout.core.physics_engine import PhysicsEngine
        eng = PhysicsEngine(seed=0)
        eng.initialize_game()
        eng.apply_actions({"penguin_0": (angle, power)})
        eng.step_until_settled()
        p = eng.penguins["penguin_0"]
        return p.position, p.alive

    def _run_tensor_impulse(self, angle: float, power: float):
        """Run same impulse in tensor engine and return final position."""
        eng = TensorPhysicsEngine(device="cpu")
        pos, vel, alive = eng.reset(1)
        actions = torch.zeros(1, 6, 2)
        actions[0, 0, 0] = angle
        actions[0, 0, 1] = power
        vel = eng.apply_impulses(vel, actions, alive)
        pos2, vel2, alive2, _ = eng.step_until_settled(
            pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH
        )
        return (pos2[0, 0, 0].item(), pos2[0, 0, 1].item()), alive2[0, 0].item()

    def test_vs_pymunk_simple(self) -> None:
        """Single rightward impulse on penguin_0."""
        pymunk_pos, pymunk_alive = self._run_pymunk_impulse(0.0, 200.0)
        tensor_pos, tensor_alive = self._run_tensor_impulse(0.0, 200.0)

        assert pymunk_alive == tensor_alive
        # Positions should be within 10% of arena width
        tolerance = DEFAULTS.ARENA_HALF_WIDTH * 0.10
        assert abs(pymunk_pos[0] - tensor_pos[0]) < tolerance
        assert abs(pymunk_pos[1] - tensor_pos[1]) < tolerance

    def test_vs_pymunk_diagonal(self) -> None:
        """Diagonal impulse."""
        pymunk_pos, pymunk_alive = self._run_pymunk_impulse(45.0, 300.0)
        tensor_pos, tensor_alive = self._run_tensor_impulse(45.0, 300.0)

        assert pymunk_alive == tensor_alive
        tolerance = DEFAULTS.ARENA_HALF_WIDTH * 0.15
        assert abs(pymunk_pos[0] - tensor_pos[0]) < tolerance
        assert abs(pymunk_pos[1] - tensor_pos[1]) < tolerance

    def test_vs_pymunk_collision(self) -> None:
        """Two penguins colliding: compare alive counts."""
        from knockout.core.physics_engine import PhysicsEngine

        # Pymunk version: two penguins launched at each other
        eng = PhysicsEngine(seed=0)
        eng.initialize_game()
        eng.apply_actions({
            "penguin_0": (0.0, 400.0),   # right
            "penguin_3": (180.0, 400.0),  # left
        })
        eng.step_until_settled()
        pymunk_alive_a = eng.get_alive_count(0)
        pymunk_alive_b = eng.get_alive_count(1)

        # Tensor version
        tengine = TensorPhysicsEngine(device="cpu")
        pos, vel, alive = tengine.reset(1)
        actions = torch.zeros(1, 6, 2)
        actions[0, 0, 0] = 0.0
        actions[0, 0, 1] = 400.0
        actions[0, 3, 0] = 180.0
        actions[0, 3, 1] = 400.0
        vel = tengine.apply_impulses(vel, actions, alive)
        pos2, vel2, alive2, _ = tengine.step_until_settled(
            pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH
        )
        tensor_alive_a = alive2[0, :3].sum().item()
        tensor_alive_b = alive2[0, 3:].sum().item()

        # Alive counts should match
        assert tensor_alive_a == pymunk_alive_a
        assert tensor_alive_b == pymunk_alive_b

    def test_vs_pymunk_game(self) -> None:
        """Run a multi-round game with identical actions, compare alive counts."""
        from knockout.core.physics_engine import PhysicsEngine

        rng = np.random.default_rng(123)
        rounds = 5

        # Pymunk game
        pymunk_eng = PhysicsEngine(seed=0)
        pymunk_eng.initialize_game()

        # Tensor game
        tengine = TensorPhysicsEngine(device="cpu")
        pos, vel, alive = tengine.reset(1)

        match_count = 0
        for r in range(rounds):
            # Generate random actions for all 6 penguins
            angles = rng.uniform(0, 360, size=6)
            powers = rng.uniform(0, 400, size=6)

            # Apply in Pymunk
            pymunk_actions = {}
            for i in range(6):
                agent_id = f"penguin_{i}"
                if agent_id in pymunk_eng.penguins and pymunk_eng.penguins[agent_id].alive:
                    pymunk_actions[agent_id] = (angles[i], powers[i])
            pymunk_eng.apply_actions(pymunk_actions)
            pymunk_eng.step_until_settled()

            # Apply in tensor
            t_actions = torch.zeros(1, 6, 2)
            for i in range(6):
                t_actions[0, i, 0] = angles[i]
                t_actions[0, i, 1] = powers[i]
            vel = tengine.apply_impulses(vel, t_actions, alive)
            pos, vel, alive, _ = tengine.step_until_settled(
                pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH
            )

            # Compare alive counts
            pymunk_a = pymunk_eng.get_alive_count(0)
            pymunk_b = pymunk_eng.get_alive_count(1)
            tensor_a = alive[0, :3].sum().item()
            tensor_b = alive[0, 3:].sum().item()

            if pymunk_a == tensor_a and pymunk_b == tensor_b:
                match_count += 1

        # At least 60% of rounds should match (physics are similar, not identical)
        assert match_count >= rounds * 0.6, (
            f"Only {match_count}/{rounds} rounds had matching alive counts"
        )
