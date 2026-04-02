"""Tests for PhysicsEngine class."""

import numpy as np

from knockout.core.config import DEFAULTS
from knockout.core.physics_engine import PhysicsEngine


class TestPhysicsEngine:
    """Tests for deterministic physics simulation."""

    def test_creation(self) -> None:
        """Test creating physics engine."""
        engine = PhysicsEngine()
        assert engine.space is not None
        assert engine.ice_sheet is not None
        assert len(engine.penguins) == 0
        assert engine.step_count == 0

    def test_creation_with_seed(self) -> None:
        """Test creating physics engine with seed."""
        engine = PhysicsEngine(seed=42)
        assert engine.space is not None

    def test_initialize_game_creates_six_penguins(self) -> None:
        """Test that initialize_game creates 6 penguins."""
        engine = PhysicsEngine()
        engine.initialize_game()

        assert len(engine.penguins) == 6
        assert "penguin_0" in engine.penguins
        assert "penguin_5" in engine.penguins

    def test_initialize_game_team_assignments(self) -> None:
        """Test that penguins are assigned to correct teams."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Team A: penguins 0-2
        assert engine.penguins["penguin_0"].team_id == 0
        assert engine.penguins["penguin_1"].team_id == 0
        assert engine.penguins["penguin_2"].team_id == 0

        # Team B: penguins 3-5
        assert engine.penguins["penguin_3"].team_id == 1
        assert engine.penguins["penguin_4"].team_id == 1
        assert engine.penguins["penguin_5"].team_id == 1

    def test_initialize_game_spawn_positions(self) -> None:
        """Test that penguins spawn at correct positions."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Team A on left (negative x)
        for i in range(3):
            x, _ = engine.penguins[f"penguin_{i}"].position
            assert x < 0

        # Team B on right (positive x)
        for i in range(3, 6):
            x, _ = engine.penguins[f"penguin_{i}"].position
            assert x > 0

    def test_initialize_game_all_alive(self) -> None:
        """Test that all penguins start alive."""
        engine = PhysicsEngine()
        engine.initialize_game()

        for penguin in engine.penguins.values():
            assert penguin.alive

    def test_initialize_game_resets_step_count(self) -> None:
        """Test that initialize_game resets step counter."""
        engine = PhysicsEngine()
        engine.initialize_game()
        engine.step(10)

        assert engine.step_count == 10

        engine.initialize_game()
        assert engine.step_count == 0

    def test_apply_actions_single_penguin(self) -> None:
        """Test applying action to single penguin."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Apply action to penguin_0
        actions = {"penguin_0": (45.0, 300.0)}
        engine.apply_actions(actions)

        # Penguin should have velocity
        vx, vy = engine.penguins["penguin_0"].velocity
        assert vx > 0 or vy > 0

    def test_apply_actions_multiple_penguins(self) -> None:
        """Test applying actions to multiple penguins simultaneously."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Apply actions to all penguins
        actions = {
            f"penguin_{i}": (float(i * 30), 200.0) for i in range(6)
        }
        engine.apply_actions(actions)

        # All penguins should have velocity
        for penguin in engine.penguins.values():
            vx, vy = penguin.velocity
            assert vx != 0 or vy != 0

    def test_apply_actions_clips_to_valid_range(self) -> None:
        """Test that actions are clipped to valid range."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Try invalid actions (should be clipped)
        actions = {
            "penguin_0": (400.0, 1000.0),  # Both out of range
            "penguin_1": (-10.0, -50.0),  # Both negative
        }
        engine.apply_actions(actions)

        # Should not crash, actions should be clipped

    def test_step_increments_counter(self) -> None:
        """Test that step increments step counter."""
        engine = PhysicsEngine()
        engine.initialize_game()

        assert engine.step_count == 0

        engine.step(1)
        assert engine.step_count == 1

        engine.step(5)
        assert engine.step_count == 6

    def test_step_updates_positions(self) -> None:
        """Test that physics step updates penguin positions."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Launch penguin
        engine.apply_actions({"penguin_0": (0.0, 300.0)})

        # Get initial position
        x0, y0 = engine.penguins["penguin_0"].position

        # Step physics
        engine.step(10)

        # Position should have changed
        x1, y1 = engine.penguins["penguin_0"].position
        assert abs(x1 - x0) > 0.1 or abs(y1 - y0) > 0.1

    def test_elimination_when_leaving_boundary(self) -> None:
        """Test that penguins are eliminated when they leave the ice sheet."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Get initial position
        initial_x, initial_y = engine.penguins["penguin_0"].position

        # Launch penguin directly outward from center with maximum power.
        # Compute angle in 0-360 range (penguin_0 spawns on the left side).
        angle_rad = np.arctan2(initial_y, initial_x)
        angle = float(np.degrees(angle_rad)) % 360.0

        # Launch with max power in direction away from center
        engine.apply_actions(
            {"penguin_0": (angle, engine.config.MAX_LAUNCH_FORCE)}
        )

        # Step until penguin leaves boundary (should happen fairly quickly)
        for _ in range(1000):
            engine.step(1)
            if not engine.penguins["penguin_0"].alive:
                break

        # Penguin should be eliminated
        assert not engine.penguins["penguin_0"].alive

    def test_get_alive_count_team_a(self) -> None:
        """Test counting alive penguins for Team A."""
        engine = PhysicsEngine()
        engine.initialize_game()

        count = engine.get_alive_count(team_id=0)
        assert count == 3

    def test_get_alive_count_team_b(self) -> None:
        """Test counting alive penguins for Team B."""
        engine = PhysicsEngine()
        engine.initialize_game()

        count = engine.get_alive_count(team_id=1)
        assert count == 3

    def test_get_alive_count_after_elimination(self) -> None:
        """Test alive count decreases after elimination."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Eliminate one penguin from Team A
        engine.penguins["penguin_0"].eliminate(engine.space)

        count_a = engine.get_alive_count(team_id=0)
        count_b = engine.get_alive_count(team_id=1)

        assert count_a == 2
        assert count_b == 3

    def test_is_game_over_initially_false(self) -> None:
        """Test that game is not over initially."""
        engine = PhysicsEngine()
        engine.initialize_game()

        assert not engine.is_game_over()

    def test_is_game_over_when_team_eliminated(self) -> None:
        """Test that game is over when one team is fully eliminated."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Eliminate all of Team A
        for i in range(3):
            engine.penguins[f"penguin_{i}"].eliminate(engine.space)

        assert engine.is_game_over()

    def test_get_winner_initially_none(self) -> None:
        """Test that winner is None initially."""
        engine = PhysicsEngine()
        engine.initialize_game()

        assert engine.get_winner() is None

    def test_get_winner_team_a(self) -> None:
        """Test winner detection when Team A wins."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Eliminate all of Team B
        for i in range(3, 6):
            engine.penguins[f"penguin_{i}"].eliminate(engine.space)

        winner = engine.get_winner()
        assert winner == 0  # Team A wins

    def test_get_winner_team_b(self) -> None:
        """Test winner detection when Team B wins."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Eliminate all of Team A
        for i in range(3):
            engine.penguins[f"penguin_{i}"].eliminate(engine.space)

        winner = engine.get_winner()
        assert winner == 1  # Team B wins

    def test_get_winner_draw(self) -> None:
        """Test winner detection for draw."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Eliminate all penguins
        for penguin in engine.penguins.values():
            penguin.eliminate(engine.space)

        winner = engine.get_winner()
        assert winner == -1  # Draw

    def test_physics_runs_1000_steps_without_crash(self) -> None:
        """Test that physics can run for 1000+ steps without crashing."""
        engine = PhysicsEngine()
        engine.initialize_game()

        # Step 1000 times
        engine.step(1000)

        # Should complete without error
        assert engine.step_count == 1000

    def test_deterministic_physics_exact_equality(self) -> None:
        """CRITICAL: Same seed must produce identical results.

        This is the most important test - determinism is required for replay.
        """
        seed = 42

        # Engine 1
        engine1 = PhysicsEngine(seed=seed)
        engine1.initialize_game()
        engine1.apply_actions({"penguin_0": (45.0, 300.0)})
        engine1.step(100)

        # Engine 2 (same seed)
        engine2 = PhysicsEngine(seed=seed)
        engine2.initialize_game()
        engine2.apply_actions({"penguin_0": (45.0, 300.0)})
        engine2.step(100)

        # Positions must be EXACTLY equal (not approximately)
        pos1 = engine1.penguins["penguin_0"].position
        pos2 = engine2.penguins["penguin_0"].position

        assert pos1 == pos2, f"Determinism violated: {pos1} != {pos2}"

        # Velocities must be EXACTLY equal
        vel1 = engine1.penguins["penguin_0"].velocity
        vel2 = engine2.penguins["penguin_0"].velocity

        assert vel1 == vel2, f"Determinism violated: {vel1} != {vel2}"

    def test_deterministic_physics_all_penguins(self) -> None:
        """Test determinism for all penguins after complex simulation."""
        seed = 123

        # Engine 1
        engine1 = PhysicsEngine(seed=seed)
        engine1.initialize_game()
        actions = {f"penguin_{i}": (float(i * 45), 250.0) for i in range(6)}
        engine1.apply_actions(actions)
        engine1.step(50)

        # Engine 2 (same seed)
        engine2 = PhysicsEngine(seed=seed)
        engine2.initialize_game()
        engine2.apply_actions(actions)
        engine2.step(50)

        # All penguins should match exactly
        for i in range(6):
            agent_id = f"penguin_{i}"
            pos1 = engine1.penguins[agent_id].position
            pos2 = engine2.penguins[agent_id].position
            assert pos1 == pos2, f"Penguin {i} position mismatch: {pos1} != {pos2}"

    def test_deterministic_physics_with_collisions(self) -> None:
        """Test determinism with penguin collisions."""
        seed = 789

        # Engine 1
        engine1 = PhysicsEngine(seed=seed)
        engine1.initialize_game()
        # Launch two penguins toward each other
        engine1.apply_actions({
            "penguin_0": (0.0, 400.0),  # East
            "penguin_5": (180.0, 400.0),  # West
        })
        engine1.step(200)

        # Engine 2 (same seed)
        engine2 = PhysicsEngine(seed=seed)
        engine2.initialize_game()
        engine2.apply_actions({
            "penguin_0": (0.0, 400.0),
            "penguin_5": (180.0, 400.0),
        })
        engine2.step(200)

        # Both penguins should match exactly after collision
        for agent_id in ["penguin_0", "penguin_5"]:
            pos1 = engine1.penguins[agent_id].position
            pos2 = engine2.penguins[agent_id].position
            assert pos1 == pos2, f"{agent_id} collision determinism failed"

    def test_rng_isolation(self) -> None:
        """Test that creating engine2 does not affect engine1's behavior.

        BUG FIX #2 verification: Uses instance-level RNG instead of
        global np.random.seed(), so engines don't pollute each other.
        """
        # Create engine1 and run some steps
        engine1 = PhysicsEngine(seed=42)
        engine1.initialize_game()
        engine1.apply_actions({"penguin_0": (45.0, 300.0)})
        engine1.step(50)

        # Capture engine1 state
        pos1_before = engine1.penguins["penguin_0"].position

        # Now create engine2 with a DIFFERENT seed (this would have
        # corrupted engine1 if using global np.random.seed)
        engine2 = PhysicsEngine(seed=999)
        engine2.initialize_game()

        # Run engine1 reference: fresh engine with same seed, same actions
        engine1_ref = PhysicsEngine(seed=42)
        engine1_ref.initialize_game()
        engine1_ref.apply_actions({"penguin_0": (45.0, 300.0)})
        engine1_ref.step(50)

        # engine1 and engine1_ref should match exactly
        pos1_ref = engine1_ref.penguins["penguin_0"].position
        assert pos1_before == pos1_ref, (
            f"RNG isolation failed: engine1 state affected by engine2 creation. "
            f"{pos1_before} != {pos1_ref}"
        )

    def test_collision_bias_applied(self) -> None:
        """Test that the fixed collision_bias is applied to the space.

        BUG FIX #3 verification: collision_bias should be ~0.0018 (Pymunk default),
        not the old buggy value of 0.2.
        """
        engine = PhysicsEngine()

        # Verify space has correct collision_bias
        assert engine.space.collision_bias == DEFAULTS.COLLISION_BIAS

        # Verify it's close to the Pymunk default (~0.0018)
        expected = pow(1.0 - 0.1, 60.0)
        assert abs(engine.space.collision_bias - expected) < 1e-10

        # Verify it's NOT the old buggy value
        assert engine.space.collision_bias != 0.2
        assert engine.space.collision_bias < 0.01


class TestSettleDetection:
    """Tests for the settle-based stepping mechanic."""

    def test_are_all_settled_initially(self) -> None:
        """Penguins at rest (just spawned) should be settled."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        assert engine.are_all_settled(), "Freshly spawned penguins should be settled"

    def test_are_all_settled_after_launch(self) -> None:
        """Penguins with velocity should NOT be settled."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        engine.apply_actions({"penguin_0": (0.0, 400.0)})
        engine.step(1)

        assert not engine.are_all_settled(), "Moving penguin should not be settled"

    def test_step_until_settled(self) -> None:
        """Launch a penguin, call step_until_settled, verify it eventually settles."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        engine.apply_actions({"penguin_0": (0.0, 300.0)})
        steps = engine.step_until_settled()

        # Should have taken more than 1 step to settle
        assert steps > 1, f"Expected multiple steps to settle, got {steps}"

        # After settling, all penguins should be at rest (or eliminated)
        assert engine.are_all_settled(), "Penguins should be settled after step_until_settled"

    def test_step_until_settled_respects_max_steps(self) -> None:
        """Verify that max_steps cap works."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        # Launch with high power so it takes many steps to settle
        engine.apply_actions({"penguin_0": (0.0, 500.0)})

        max_cap = 5
        steps = engine.step_until_settled(max_steps=max_cap)

        assert steps == max_cap, f"Expected exactly {max_cap} steps, got {steps}"


class TestRescalePenguins:
    """Tests for penguin position/velocity rescaling after arena shrink."""

    def test_rescale_preserves_normalized_positions(self) -> None:
        """Rescaling penguins should preserve their relative positions."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        old_hw = engine.ice_sheet.half_width

        # Record normalized positions before rescale
        normalized_before = {}
        for agent_id, penguin in engine.penguins.items():
            x, y = penguin.position
            normalized_before[agent_id] = (x / old_hw, y / old_hw)

        # Shrink arena and rescale
        scale = engine.ice_sheet.shrink(0.67, 30.0)
        engine.rescale_penguins(scale)

        new_hw = engine.ice_sheet.half_width

        # Normalized positions should be identical
        for agent_id, penguin in engine.penguins.items():
            if penguin.alive:
                x, y = penguin.position
                norm_x, norm_y = x / new_hw, y / new_hw
                expected_x, expected_y = normalized_before[agent_id]
                assert abs(norm_x - expected_x) < 1e-9, (
                    f"{agent_id} norm_x changed: {norm_x} vs {expected_x}"
                )
                assert abs(norm_y - expected_y) < 1e-9, (
                    f"{agent_id} norm_y changed: {norm_y} vs {expected_y}"
                )

    def test_rescale_keeps_penguins_inside(self) -> None:
        """After rescale, all alive penguins should still be inside the arena."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        # Move a penguin near the edge
        engine.penguins["penguin_0"].body.position = (80.0, 70.0)

        scale = engine.ice_sheet.shrink(0.67, 30.0)
        engine.rescale_penguins(scale)

        for penguin in engine.penguins.values():
            if penguin.alive:
                assert engine.ice_sheet.is_inside(penguin.position), (
                    f"{penguin.agent_id} outside after rescale: {penguin.position}"
                )

    def test_rescale_scales_velocity(self) -> None:
        """Velocity should be scaled by the same factor as position."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        # Give penguin a velocity
        engine.penguins["penguin_0"].body.velocity = (100.0, 50.0)

        scale = engine.ice_sheet.shrink(0.5, 30.0)
        engine.rescale_penguins(scale)

        vx, vy = engine.penguins["penguin_0"].velocity
        assert abs(vx - 100.0 * scale) < 1e-9
        assert abs(vy - 50.0 * scale) < 1e-9

    def test_rescale_ignores_eliminated_penguins(self) -> None:
        """Eliminated penguins should not be affected by rescaling."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        engine.penguins["penguin_0"].eliminate(engine.space)

        # Should not crash
        scale = engine.ice_sheet.shrink(0.67, 30.0)
        engine.rescale_penguins(scale)

        assert not engine.penguins["penguin_0"].alive
