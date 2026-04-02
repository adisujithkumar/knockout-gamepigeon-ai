"""Regression tests verifying all bug fixes from the consolidation."""

import numpy as np
import pymunk
import pytest

from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.penguin import Penguin
from knockout.core.physics_engine import PhysicsEngine
from knockout.env.observations import ObservationBuilder
from knockout.agents.heuristic_agent import HeuristicAgent


class TestBugFix1_WorldImpulse:
    """Bug #1: apply_impulse_at_world_point instead of local_point."""

    def test_impulse_direction_independent_of_rotation(self):
        """Impulse direction should not depend on body rotation."""
        results = []
        for angle_offset in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            space = pymunk.Space()
            penguin = Penguin("test", 0, (0.0, 0.0), space)
            penguin.body.angle = angle_offset

            # Apply launch at 0 degrees (east)
            penguin.apply_launch(0.0, 300.0)
            vx, vy = penguin.velocity
            results.append((vx, vy))

        # All should have same velocity direction (positive x, ~zero y)
        for vx, vy in results:
            assert vx > 25.0, f"X velocity too low: {vx} (rotation affected direction)"
            assert abs(vy) < 1.0, f"Y velocity too high: {vy} (rotation affected direction)"


class TestBugFix2_RNGIsolation:
    """Bug #2: Instance RNG instead of global np.random.seed."""

    def test_rng_isolation_between_engines(self):
        """Creating engine with seed should not affect other engines."""
        # Create first engine with seed 42
        engine1 = PhysicsEngine(seed=42)
        engine1.initialize_game()
        engine1.apply_actions({"penguin_0": (45.0, 300.0)})
        engine1.step(50)

        # Create another engine with different seed (should NOT affect engine1)
        engine2 = PhysicsEngine(seed=999)

        # Engine1 should still produce same results as a fresh run
        engine1_fresh = PhysicsEngine(seed=42)
        engine1_fresh.initialize_game()
        engine1_fresh.apply_actions({"penguin_0": (45.0, 300.0)})
        engine1_fresh.step(50)

        # Positions should match exactly
        assert engine1.penguins["penguin_0"].position == engine1_fresh.penguins["penguin_0"].position

    def test_no_global_seed_pollution(self):
        """Verify np.random global state is not modified."""
        # Set global seed
        np.random.seed(123)
        val_before = np.random.random()

        # Reset global seed
        np.random.seed(123)

        # Creating an engine should NOT affect global state
        engine = PhysicsEngine(seed=456)

        val_after = np.random.random()
        assert val_before == val_after, "PhysicsEngine polluted global np.random state"


class TestBugFix3_CollisionBias:
    """Bug #3: COLLISION_BIAS should be Pymunk default, not 0.2."""

    def test_collision_bias_is_pymunk_default(self):
        """Collision bias should be ~0.0018 (not 0.2)."""
        expected = pow(1.0 - 0.1, 60.0)
        assert abs(DEFAULTS.COLLISION_BIAS - expected) < 1e-10
        assert DEFAULTS.COLLISION_BIAS < 0.01, f"COLLISION_BIAS too high: {DEFAULTS.COLLISION_BIAS}"

    def test_collision_bias_applied_to_space(self):
        """Space should use the corrected collision bias."""
        engine = PhysicsEngine(seed=42)
        assert abs(engine.space.collision_bias - DEFAULTS.COLLISION_BIAS) < 1e-10


class TestBugFix4_ObservationBounds:
    """Bug #4: All observation values must be within [-1, 1]."""

    def test_observation_bounds_under_extreme_conditions(self):
        """Observations stay in [-1, 1] even with extreme physics states."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Launch all penguins with max force in various directions
        for i in range(6):
            engine.apply_actions({f"penguin_{i}": (float(i * 60), 500.0)})

        # Run for many steps - penguins may go outside arena
        engine.step(300)

        for agent_id, penguin in engine.penguins.items():
            if penguin.alive:
                obs = builder.build_observation(agent_id, engine.penguins, timestep=300)
                assert np.all(obs >= -1.0), f"{agent_id}: obs min={obs.min()}"
                assert np.all(obs <= 1.0), f"{agent_id}: obs max={obs.max()}"


class TestBugFix5_RelativeVelocity:
    """Bug #5: Relative velocity should be truly relative (other - ego)."""

    def test_relative_velocity_same_direction(self):
        """Two penguins moving same direction at same speed -> rel_vel ~ 0."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Launch penguin_0 and penguin_3 in same direction at same speed
        engine.apply_actions({
            "penguin_0": (0.0, 300.0),
            "penguin_3": (0.0, 300.0),
        })
        # Don't step - just check immediate velocities

        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # Enemy relative velocity features:
        # Enemies start at index 42 (after ego 14 + allies 28).
        # Per-penguin features: 14. Rel_vel_x is index 11 within the block.
        # First enemy (closest) rel_vel_x: obs[42 + 11] = obs[53]
        # Both have same launch parameters -> same velocity -> rel_vel ~ 0
        enemy_rel_vel_x = obs[42 + 11]
        assert abs(enemy_rel_vel_x) < 0.1, f"Relative velocity not near zero: {enemy_rel_vel_x}"

    def test_relative_velocity_opposite_direction(self):
        """Two penguins moving opposite directions -> rel_vel ~ 2x."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Launch ego east, enemy west
        engine.apply_actions({
            "penguin_0": (0.0, 300.0),    # East
            "penguin_3": (180.0, 300.0),   # West
        })

        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # The relative velocity in x should be large negative
        # (enemy moving west relative to ego moving east)
        enemy_rel_vel_x = obs[42 + 11]
        assert enemy_rel_vel_x < -0.1, f"Relative velocity not negative enough: {enemy_rel_vel_x}"


class TestBugFix6_PredictionDamping:
    """Bug #6: Prediction uses geometric sum for damped displacement."""

    def test_prediction_uses_geometric_sum(self):
        """Verify prediction formula: v * dt * (1 - d^N) / (1 - d)."""
        agent = HeuristicAgent("test", seed=42)

        target = {
            "position": (0.0, 0.0),
            "velocity": (100.0, 0.0),
            "distance_to_ego": 50.0,
            "distance_to_edge": 50.0,
            "alive": True,
        }

        pred = agent._predict_position(target, prediction_steps=10)

        # Expected: v * dt * (1 - d^N) / (1 - d)
        # Pymunk damping is per-second; per-step retention = damping^dt
        dt = DEFAULTS.FIXED_DT  # 1/60
        d = DEFAULTS.DAMPING ** dt  # per-step retention factor
        N = 10
        expected_displacement = 100.0 * dt * (1.0 - d**N) / (1.0 - d)
        expected_x = 0.0 + expected_displacement

        assert abs(pred[0] - expected_x) < 0.01, f"Prediction wrong: {pred[0]} vs {expected_x}"
        assert abs(pred[1] - 0.0) < 0.01, f"Y should be 0: {pred[1]}"


class TestBugFix8_MutualKnockoutDraw:
    """Bug #8: Game must wait for all physics to resolve before declaring winner.

    When the last two penguins (one per team) collide and BOTH fly off the map,
    the game should report a draw, not a premature win for either team.
    """

    def _setup_staggered_elimination(self):
        """Create a scenario where both penguins are aimed off-map but at
        different distances from their respective edges.

        penguin_0 is 5 units from the left edge (dies first).
        penguin_3 is 20 units from the right edge (dies a bit later).
        Both are launched outward with max force, so both will be
        eliminated -- but penguin_0 exits first.  Without the timing
        fix, the simulation would stop when penguin_0 dies and
        incorrectly declare penguin_3 (Team B) the winner.
        """
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        # Eliminate everyone except penguin_0 and penguin_3
        for pid in ("penguin_1", "penguin_2", "penguin_4", "penguin_5"):
            engine.penguins[pid].eliminate(engine.space)

        hw = engine.ice_sheet.half_width

        # penguin_0: very near left edge, aimed left (off the map quickly)
        engine.penguins["penguin_0"].body.position = (-(hw - 5), 0.0)
        engine.penguins["penguin_0"].body.velocity = (0.0, 0.0)

        # penguin_3: further from right edge, aimed right (off the map later)
        engine.penguins["penguin_3"].body.position = ((hw - 20), 0.0)
        engine.penguins["penguin_3"].body.velocity = (0.0, 0.0)

        return engine

    def test_mutual_knockout_is_draw_via_step_until_settled(self):
        """Staggered elimination: both penguins fly off -> draw.

        penguin_0 is eliminated first (closer to edge), but penguin_3 is
        also heading off the map.  The simulation must continue after
        penguin_0 dies so that penguin_3 also gets eliminated, producing
        a correct draw.
        """
        engine = self._setup_staggered_elimination()

        max_force = engine.config.MAX_LAUNCH_FORCE
        engine.apply_actions({
            "penguin_0": (180.0, max_force),  # West (off left edge)
            "penguin_3": (0.0, max_force),    # East (off right edge)
        })

        engine.step_until_settled()

        # Both should be eliminated
        assert not engine.penguins["penguin_0"].alive, "penguin_0 should be eliminated"
        assert not engine.penguins["penguin_3"].alive, "penguin_3 should be eliminated"

        # Game should be over and result should be a draw
        assert engine.is_game_over()
        assert engine.get_winner() == -1, f"Expected draw (-1), got {engine.get_winner()}"

    def test_mutual_knockout_is_draw_via_env(self):
        """Same staggered-elimination scenario through PenguinEnv.

        Rewards should be 0 for all agents (draw).
        """
        from knockout.env.penguin_env import PenguinEnv

        env = PenguinEnv(seed=42, settle_mode=True)
        env.reset()

        # Eliminate everyone except penguin_0 and penguin_3
        for pid in ("penguin_1", "penguin_2", "penguin_4", "penguin_5"):
            env.physics_engine.penguins[pid].eliminate(env.physics_engine.space)

        hw = env.physics_engine.ice_sheet.half_width
        env.physics_engine.penguins["penguin_0"].body.position = (-(hw - 5), 0.0)
        env.physics_engine.penguins["penguin_0"].body.velocity = (0.0, 0.0)
        env.physics_engine.penguins["penguin_3"].body.position = ((hw - 20), 0.0)
        env.physics_engine.penguins["penguin_3"].body.velocity = (0.0, 0.0)

        max_force = env.config.MAX_LAUNCH_FORCE
        actions = {
            "penguin_0": np.array([180.0, max_force], dtype=np.float32),
            "penguin_3": np.array([0.0, max_force], dtype=np.float32),
        }
        # Supply no-ops for eliminated agents so env doesn't complain
        for pid in ("penguin_1", "penguin_2", "penguin_4", "penguin_5"):
            actions[pid] = np.array([0.0, 0.0], dtype=np.float32)

        _, rewards, terminations, _, infos = env.step(actions)

        # Game should be over
        assert all(terminations.values()), "Game should be over after mutual knockout"

        # All rewards should be 0 (draw)
        for agent_id, r in rewards.items():
            assert r == 0.0, f"{agent_id} reward should be 0.0 (draw), got {r}"

        # Winner should be -1 (draw)
        for info in infos.values():
            assert info["winner"] == -1, f"Expected draw (-1), got {info['winner']}"

    def test_step_until_settled_does_not_stop_at_first_elimination(self):
        """Verify physics keeps running after one team has 0 alive penguins.

        Before the fix, step_until_settled() would stop as soon as
        is_game_over() returned True (one team at 0).  This test ensures
        the simulation continues so the remaining moving penguin can also
        fall off, converting a false win into a correct draw.
        """
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()

        # Only keep one per team
        for pid in ("penguin_1", "penguin_2", "penguin_4", "penguin_5"):
            engine.penguins[pid].eliminate(engine.space)

        hw = engine.ice_sheet.half_width

        # penguin_0: near left edge, launched further left (off the map)
        engine.penguins["penguin_0"].body.position = (-hw * 0.92, 0.0)
        engine.penguins["penguin_0"].body.velocity = (0.0, 0.0)

        # penguin_3: near right edge, launched further right (off the map)
        engine.penguins["penguin_3"].body.position = (hw * 0.92, 0.0)
        engine.penguins["penguin_3"].body.velocity = (0.0, 0.0)

        max_force = engine.config.MAX_LAUNCH_FORCE
        # penguin_0 aimed left (180 deg), penguin_3 aimed right (0 deg)
        engine.apply_actions({
            "penguin_0": (180.0, max_force),
            "penguin_3": (0.0, max_force),
        })

        engine.step_until_settled()

        # Both should be eliminated — draw
        assert not engine.penguins["penguin_0"].alive
        assert not engine.penguins["penguin_3"].alive
        assert engine.get_winner() == -1


class TestBugFix7_UtilityWeights:
    """Bug #7: Target utility weights must sum to 1.0."""

    def test_utility_has_three_factors(self):
        """Utility should use edge(0.4) + distance(0.25) + vel_toward_edge(0.35) = 1.0."""
        agent = HeuristicAgent("test", seed=42)

        enemy = {
            "position": (50.0, 0.0),
            "velocity": (10.0, 0.0),  # Moving toward edge
            "distance_to_ego": 100.0,
            "distance_to_edge": 50.0,
            "alive": True,
        }
        ego_state = {"position": (0.0, 0.0)}

        utility = agent._calculate_target_utility(enemy, ego_state)

        # Utility should be between 0 and 1 (weights sum to 1.0, each factor in [0,1])
        assert 0.0 <= utility <= 1.0, f"Utility out of range: {utility}"

    def test_utility_rewards_edge_proximity_and_velocity(self):
        """Enemy near edge moving outward should have higher utility."""
        agent = HeuristicAgent("test", seed=42)
        ego_state = {"position": (0.0, 0.0)}

        # Enemy near edge, moving toward edge
        dangerous_enemy = {
            "position": (80.0, 0.0),
            "velocity": (50.0, 0.0),  # Moving toward edge
            "distance_to_ego": 80.0,
            "distance_to_edge": 20.0,
            "alive": True,
        }

        # Enemy in center, stationary
        safe_enemy = {
            "position": (10.0, 0.0),
            "velocity": (0.0, 0.0),
            "distance_to_ego": 10.0,
            "distance_to_edge": 90.0,
            "alive": True,
        }

        utility_dangerous = agent._calculate_target_utility(dangerous_enemy, ego_state)
        utility_safe = agent._calculate_target_utility(safe_enemy, ego_state)

        assert utility_dangerous > utility_safe, \
            f"Dangerous enemy should have higher utility: {utility_dangerous} vs {utility_safe}"
