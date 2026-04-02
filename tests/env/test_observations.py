"""Tests for observation builder."""

import numpy as np
import pytest

from knockout.env.observations import ObservationBuilder
from knockout.core.physics_engine import PhysicsEngine


class TestObservationBuilder:
    """Test observation construction."""

    def test_observation_dimension(self):
        """Test that observations have correct dimension (89)."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        assert obs.shape == (89,), f"Expected shape (89,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_observation_dimensions_breakdown(self):
        """Test that observation dimensions add up correctly."""
        builder = ObservationBuilder()

        # Verify dimension breakdown
        expected = (
            builder.FEATURES_PER_PENGUIN * (1 + builder.NUM_ALLIES + builder.NUM_ENEMIES)
            + builder.GLOBAL_FEATURES
        )
        assert expected == 89
        assert builder.TOTAL_DIM == 89

    def test_ego_observation_structure(self):
        """Test that ego penguin features are extracted correctly."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # First 14 features are ego features
        ego_features = obs[:14]
        assert len(ego_features) == 14

        # Check that relative features are zero for ego (indices 9-13)
        assert ego_features[9] == 0.0  # relative_position_x
        assert ego_features[10] == 0.0  # relative_position_y
        assert ego_features[11] == 0.0  # relative_velocity_x
        assert ego_features[12] == 0.0  # relative_velocity_y
        assert ego_features[13] == 0.0  # distance_to_ego

    def test_ally_features_sorted_by_distance(self):
        """Test that ally features are sorted by distance to ego."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Get observations for penguin_0 (Team A)
        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # Allies are features 14-41 (2 allies * 14 features each)
        ally1_distance = obs[14 + 13]  # distance_to_ego for ally 1
        ally2_distance = obs[28 + 13]  # distance_to_ego for ally 2

        # Verify sorting (closest first)
        assert ally1_distance <= ally2_distance

    def test_enemy_features_sorted_by_distance(self):
        """Test that enemy features are sorted by distance to ego."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Get observations for penguin_0 (Team A)
        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # Enemies are features 42-83 (3 enemies * 14 features each)
        enemy1_distance = obs[42 + 13]  # distance_to_ego for enemy 1
        enemy2_distance = obs[56 + 13]  # distance_to_ego for enemy 2
        enemy3_distance = obs[70 + 13]  # distance_to_ego for enemy 3

        # Verify sorting (closest first)
        assert enemy1_distance <= enemy2_distance
        assert enemy2_distance <= enemy3_distance

    def test_global_features_range(self):
        """Test that global features are in valid range."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # Global features are last 5 elements
        global_features = obs[-5:]

        # All should be normalized between 0 and 1
        assert np.all(global_features >= 0.0)
        assert np.all(global_features <= 1.0)

        # Check team counts
        team_a_alive_norm = global_features[0]
        team_b_alive_norm = global_features[1]

        # At start, all teams should be at full strength (1.0)
        assert team_a_alive_norm == 1.0
        assert team_b_alive_norm == 1.0

    def test_dead_penguin_observation(self):
        """Test that dead penguins get zero observations."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Eliminate penguin_0
        engine.penguins["penguin_0"].eliminate(engine.space)

        # Get observation for dead penguin
        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # Should be all zeros
        assert np.allclose(obs, 0.0)

    def test_observation_normalization(self):
        """Test that observations are normalized to reasonable ranges."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Apply some actions to create movement
        actions = {
            f"penguin_{i}": (float(i * 45), 300.0) for i in range(6)
        }  # Different angles
        engine.apply_actions(actions)
        engine.step(num_steps=10)

        obs = builder.build_observation("penguin_0", engine.penguins, timestep=10)

        # With bug fix #4, ALL features should be in [-1, 1]
        assert np.all(obs >= -1.0 - 1e-6)
        assert np.all(obs <= 1.0 + 1e-6)

    def test_observation_determinism(self):
        """Test that observations are deterministic with same seed."""
        builder = ObservationBuilder()

        # Create two engines with same seed
        engine1 = PhysicsEngine(seed=42)
        engine1.initialize_game()

        engine2 = PhysicsEngine(seed=42)
        engine2.initialize_game()

        # Get observations
        obs1 = builder.build_observation("penguin_0", engine1.penguins, timestep=0)
        obs2 = builder.build_observation("penguin_0", engine2.penguins, timestep=0)

        # Should be identical
        assert np.allclose(obs1, obs2)

    def test_observation_changes_after_actions(self):
        """Test that observations change after actions are applied."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Get initial observation
        obs_before = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # Apply action
        engine.apply_actions({"penguin_0": (90.0, 300.0)})
        engine.step(num_steps=10)

        # Get observation after action
        obs_after = builder.build_observation("penguin_0", engine.penguins, timestep=10)

        # Observations should be different
        assert not np.allclose(obs_before, obs_after)

    def test_teammate_identification(self):
        """Test that teammates are correctly identified."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Penguin 0 is on Team A (with penguins 1, 2)
        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # Allies should be features 14-41 (2 allies)
        # Enemies should be features 42-83 (3 enemies)

        # Check alive status for allies (feature index 8 within each 14-feature block)
        ally1_alive = obs[14 + 8]  # First ally
        ally2_alive = obs[28 + 8]  # Second ally

        # At start, all should be alive
        assert ally1_alive == 1.0
        assert ally2_alive == 1.0

    def test_zero_padding_for_eliminated_penguins(self):
        """Test that eliminated penguins are zero-padded in observations."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Eliminate an ally (penguin_1)
        engine.penguins["penguin_1"].eliminate(engine.space)

        # Get observation for penguin_0
        obs = builder.build_observation("penguin_0", engine.penguins, timestep=0)

        # One of the ally slots should have alive=0
        ally1_alive = obs[14 + 8]  # First ally
        ally2_alive = obs[28 + 8]  # Second ally

        # At least one should be dead (or moved to end due to sorting)
        # Due to sorting, dead penguins go to the end, so ally2 should be dead
        assert ally2_alive == 0.0 or ally1_alive == 0.0

    def test_observation_for_all_agents(self):
        """Test that all agents can get observations."""
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Get observations for all 6 agents
        for i in range(6):
            agent_id = f"penguin_{i}"
            obs = builder.build_observation(agent_id, engine.penguins, timestep=0)

            assert obs.shape == (89,)
            assert obs.dtype == np.float32
            assert not np.all(obs == 0)  # Should not be all zeros (alive)

    def test_observation_bounds_within_declared_space(self):
        """BUG FIX #4 validation: all observations stay within [-1, 1] even under extreme conditions.

        Launch penguins with max force in all directions, step 200 times.
        Verify ALL observation values are within [-1, 1].
        """
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Apply max force in various directions to create extreme velocities
        max_force = 500.0
        actions = {
            "penguin_0": (0.0, max_force),
            "penguin_1": (90.0, max_force),
            "penguin_2": (180.0, max_force),
            "penguin_3": (270.0, max_force),
            "penguin_4": (45.0, max_force),
            "penguin_5": (135.0, max_force),
        }

        # Step many times to create extreme positions/velocities
        for step in range(200):
            engine.apply_actions(actions)
            engine.step(num_steps=1)

            # Check observations for all alive agents
            for i in range(6):
                agent_id = f"penguin_{i}"
                if engine.penguins[agent_id].alive:
                    obs = builder.build_observation(agent_id, engine.penguins, timestep=step)
                    assert np.all(obs >= -1.0 - 1e-6), (
                        f"Step {step}, {agent_id}: obs min={obs.min()} at index {obs.argmin()}"
                    )
                    assert np.all(obs <= 1.0 + 1e-6), (
                        f"Step {step}, {agent_id}: obs max={obs.max()} at index {obs.argmax()}"
                    )

    def test_relative_velocity_is_truly_relative(self):
        """BUG FIX #5 validation: relative velocity is computed as (other_vel - ego_vel).

        Two penguins moving in same direction at same speed should have ~0 relative velocity.
        Two penguins moving in opposite directions should have ~2x relative velocity.
        """
        engine = PhysicsEngine(seed=42)
        engine.initialize_game()
        builder = ObservationBuilder()

        # Launch penguin_0 and penguin_1 (same team, allies) in the SAME direction
        # with the same force
        same_dir_actions = {
            "penguin_0": (0.0, 300.0),   # Both go right
            "penguin_1": (0.0, 300.0),
            "penguin_2": (0.0, 0.0),     # Stationary
            "penguin_3": (0.0, 0.0),
            "penguin_4": (0.0, 0.0),
            "penguin_5": (0.0, 0.0),
        }
        engine.apply_actions(same_dir_actions)
        engine.step(num_steps=5)

        # Get observation for penguin_0 - ally (penguin_1) should have near-zero rel velocity
        obs = builder.build_observation("penguin_0", engine.penguins, timestep=5)

        # Find the ally that is penguin_1 (closest ally since both are moving together)
        # Ally 1 features start at index 14, rel_vel_x is at offset 11, rel_vel_y at offset 12
        ally1_rel_vel_x = obs[14 + 11]
        ally1_rel_vel_y = obs[14 + 12]

        # Since both move in same direction at same speed, relative velocity should be ~0
        assert abs(ally1_rel_vel_x) < 0.1, (
            f"Same-direction rel_vel_x should be ~0, got {ally1_rel_vel_x}"
        )
        assert abs(ally1_rel_vel_y) < 0.1, (
            f"Same-direction rel_vel_y should be ~0, got {ally1_rel_vel_y}"
        )

        # Now test opposite directions
        engine2 = PhysicsEngine(seed=42)
        engine2.initialize_game()

        # Launch penguin_0 right, penguin_3 (enemy) left
        opposite_actions = {
            "penguin_0": (0.0, 300.0),    # Go right
            "penguin_1": (0.0, 0.0),
            "penguin_2": (0.0, 0.0),
            "penguin_3": (180.0, 300.0),  # Go left (opposite)
            "penguin_4": (0.0, 0.0),
            "penguin_5": (0.0, 0.0),
        }
        engine2.apply_actions(opposite_actions)
        engine2.step(num_steps=5)

        # Get observation for penguin_0
        obs2 = builder.build_observation("penguin_0", engine2.penguins, timestep=5)

        # Find enemy penguin_3 relative velocity
        # Enemy features start at index 42
        # We need to find which enemy slot penguin_3 is in (sorted by distance)
        # penguin_3 is closest enemy if it moved toward penguin_0
        # The rel_vel_x should be significantly negative (penguin_3 moves left relative to
        # penguin_0 who moves right, so relative vel = left_vel - right_vel which is very negative)
        # Check that the CLOSEST enemy has substantial relative velocity
        enemy1_rel_vel_x = obs2[42 + 11]

        # Relative velocity should be substantial (not zero like the buggy version would give)
        # With opposite directions, rel_vel should be roughly 2x individual velocity
        assert abs(enemy1_rel_vel_x) > 0.1, (
            f"Opposite-direction rel_vel_x should be substantial, got {enemy1_rel_vel_x}"
        )
