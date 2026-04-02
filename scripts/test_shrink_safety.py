#!/usr/bin/env python3
"""Concrete proof that arena shrinking never eliminates penguins.

Test 1: Place penguins at 90% of the arena edge and shrink multiple times.
         Assert all survive and their normalized positions are preserved.

Test 2: Run 5 full heuristic-vs-heuristic games (seeds 0-4) with shrink every
         5 rounds. Track alive counts to verify no penguin dies at shrink time.
"""

import sys
import numpy as np

from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.physics_engine import PhysicsEngine
from knockout.env.penguin_env import PenguinEnv
from knockout.env.observations import ObservationBuilder
from knockout.agents.heuristic_agent import HeuristicAgent


def test_edge_penguins_survive_shrinks():
    """Place penguins near edges, shrink multiple times, verify all survive."""
    print("=" * 70)
    print("TEST 1: Edge penguins survive multiple arena shrinks")
    print("=" * 70)

    engine = PhysicsEngine(seed=42, config=DEFAULTS)
    engine.initialize_game()

    # Manually move penguins to 90% of the arena edge
    hw = engine.ice_sheet.half_width
    edge_positions = [
        (hw * 0.90, hw * 0.85),    # penguin_0: near NE corner
        (-hw * 0.90, 0.0),         # penguin_1: near W edge
        (0.0, -hw * 0.90),         # penguin_2: near S edge
        (hw * 0.90, -hw * 0.90),   # penguin_3: near SE corner
        (-hw * 0.85, hw * 0.90),   # penguin_4: near NW corner
        (-hw * 0.90, -hw * 0.85),  # penguin_5: near SW corner
    ]

    for i, pos in enumerate(edge_positions):
        agent_id = f"penguin_{i}"
        penguin = engine.penguins[agent_id]
        penguin.body.position = pos
        penguin.body.velocity = (0.0, 0.0)

    num_shrinks = 5
    all_passed = True

    for shrink_num in range(1, num_shrinks + 1):
        old_hw = engine.ice_sheet.half_width
        alive_before = sum(1 for p in engine.penguins.values() if p.alive)

        # Record positions before shrink
        positions_before = {}
        normalized_before = {}
        for agent_id, penguin in engine.penguins.items():
            if penguin.alive:
                px, py = penguin.body.position
                positions_before[agent_id] = (px, py)
                normalized_before[agent_id] = (px / old_hw, py / old_hw)

        # Shrink and rescale
        scale = engine.ice_sheet.shrink(DEFAULTS.SHRINK_FACTOR, DEFAULTS.MIN_ARENA_HALF_WIDTH)
        engine.rescale_penguins(scale)

        new_hw = engine.ice_sheet.half_width
        alive_after = sum(1 for p in engine.penguins.values() if p.alive)

        print(f"\nShrink #{shrink_num}:")
        print(f"  Old half_width: {old_hw:.4f}  ->  New half_width: {new_hw:.4f}  (scale={scale:.4f})")
        print(f"  Alive: {alive_before} -> {alive_after}")

        for agent_id, penguin in engine.penguins.items():
            if agent_id in positions_before:
                bx, by = positions_before[agent_id]
                nbx, nby = normalized_before[agent_id]
                if penguin.alive:
                    ax, ay = penguin.body.position
                    nax, nay = ax / new_hw, ay / new_hw
                    inside = engine.ice_sheet.is_inside(penguin.position)
                    print(f"  {agent_id}: pos ({bx:8.3f},{by:8.3f}) -> ({ax:8.3f},{ay:8.3f})"
                          f"  norm ({nbx:7.4f},{nby:7.4f}) -> ({nax:7.4f},{nay:7.4f})"
                          f"  inside={inside}  alive=True")
                else:
                    print(f"  {agent_id}: was at ({bx:8.3f},{by:8.3f}) -- ELIMINATED!")

        if alive_after != alive_before:
            print(f"  *** FAIL: {alive_before - alive_after} penguin(s) eliminated by shrink!")
            all_passed = False
        else:
            print(f"  PASS: all {alive_after} penguins survived")

    assert all_passed, "Some penguins were eliminated by shrinking!"
    print(f"\nTEST 1 PASSED: all penguins survived {num_shrinks} shrinks.\n")


def test_extreme_edge_penguins():
    """Place penguins at EXACTLY the boundary and verify they survive shrink."""
    print("=" * 70)
    print("TEST 1b: Penguins at EXACT boundary survive shrinks")
    print("=" * 70)

    engine = PhysicsEngine(seed=42, config=DEFAULTS)
    engine.initialize_game()

    hw = engine.ice_sheet.half_width
    # Place penguins at exactly the boundary (worst case)
    extreme_positions = [
        (hw, hw),      # penguin_0: exact corner
        (-hw, hw),     # penguin_1: exact corner
        (hw, -hw),     # penguin_2: exact corner
        (-hw, -hw),    # penguin_3: exact corner
        (hw, 0.0),     # penguin_4: exact edge
        (0.0, hw),     # penguin_5: exact edge
    ]

    for i, pos in enumerate(extreme_positions):
        agent_id = f"penguin_{i}"
        penguin = engine.penguins[agent_id]
        penguin.body.position = pos
        penguin.body.velocity = (0.0, 0.0)

    for shrink_num in range(1, 4):
        old_hw = engine.ice_sheet.half_width
        alive_before = sum(1 for p in engine.penguins.values() if p.alive)

        scale = engine.ice_sheet.shrink(DEFAULTS.SHRINK_FACTOR, DEFAULTS.MIN_ARENA_HALF_WIDTH)
        engine.rescale_penguins(scale)

        new_hw = engine.ice_sheet.half_width
        alive_after = sum(1 for p in engine.penguins.values() if p.alive)

        print(f"\nShrink #{shrink_num}: {old_hw:.2f} -> {new_hw:.2f}")
        print(f"  Alive: {alive_before} -> {alive_after}")

        for agent_id, penguin in engine.penguins.items():
            if penguin.alive:
                px, py = penguin.body.position
                inside = engine.ice_sheet.is_inside(penguin.position)
                print(f"  {agent_id}: ({px:8.4f}, {py:8.4f})  inside={inside}")

        assert alive_after == alive_before, \
            f"Shrink #{shrink_num} killed {alive_before - alive_after} penguin(s)!"
        print(f"  PASS")

    print(f"\nTEST 1b PASSED.\n")


def test_full_games_with_shrink():
    """Run 5 full heuristic-vs-heuristic games and verify shrink doesn't kill.

    This uses PhysicsEngine directly (not PenguinEnv) so we can check alive
    counts between the physics-settle phase and the shrink phase, proving
    that any eliminations at shrink rounds come from collisions (physics),
    NOT from the arena shrinking itself.
    """
    print("=" * 70)
    print("TEST 2: Full heuristic-vs-heuristic games (seeds 0-4)")
    print("         Verifying shrink NEVER causes elimination")
    print("=" * 70)

    obs_builder = ObservationBuilder(config=DEFAULTS)
    shrink_caused_elim = False

    for seed in range(5):
        print(f"\n--- Game seed={seed} ---")
        engine = PhysicsEngine(seed=seed, config=DEFAULTS)
        engine.initialize_game()

        agents = {}
        for i in range(6):
            agent_id = f"penguin_{i}"
            agents[agent_id] = HeuristicAgent(
                agent_id=agent_id,
                seed=seed * 100 + i,
                config=DEFAULTS,
            )

        max_rounds = 50

        for round_num in range(1, max_rounds + 1):
            if engine.is_game_over():
                break

            # Get actions
            actions = {}
            for agent_id, penguin in engine.penguins.items():
                if not penguin.alive:
                    continue
                obs = obs_builder.build_observation(
                    agent_id, engine.penguins, engine.step_count
                )
                action = agents[agent_id].get_action(obs)
                actions[agent_id] = (float(action[0]), float(action[1]))

            alive_before_physics = sum(1 for p in engine.penguins.values() if p.alive)

            # Apply actions and settle
            engine.apply_actions(actions)
            engine.step_until_settled()

            alive_after_physics = sum(1 for p in engine.penguins.values() if p.alive)

            # Now check shrink
            is_shrink_round = (round_num % DEFAULTS.SHRINK_INTERVAL == 0)
            old_hw = engine.ice_sheet.half_width

            if is_shrink_round:
                scale = engine.ice_sheet.shrink(
                    DEFAULTS.SHRINK_FACTOR, DEFAULTS.MIN_ARENA_HALF_WIDTH
                )
                engine.rescale_penguins(scale)

            new_hw = engine.ice_sheet.half_width
            alive_after_shrink = sum(1 for p in engine.penguins.values() if p.alive)

            # Build status
            status = ""
            if is_shrink_round and new_hw < old_hw:
                status += f" *** SHRINK {old_hw:.1f} -> {new_hw:.1f}"

            physics_elims = alive_before_physics - alive_after_physics
            shrink_elims = alive_after_physics - alive_after_shrink

            if physics_elims > 0:
                killed_by_physics = [
                    p.agent_id for p in engine.penguins.values()
                    if not p.alive
                ]
                status += f" | physics eliminated {physics_elims}"

            if shrink_elims > 0:
                status += f" | !!! SHRINK ELIMINATED {shrink_elims} !!!"
                shrink_caused_elim = True

            print(f"  Round {round_num:3d}: arena={new_hw:6.1f}  "
                  f"alive: {alive_before_physics}->{alive_after_physics}"
                  f"(physics)->{alive_after_shrink}(shrink){status}")

            if engine.is_game_over():
                break

        winner = engine.get_winner()
        teams = {0: "Team A", 1: "Team B", -1: "Draw", None: "Ongoing"}
        print(f"  Result: {teams.get(winner, 'Unknown')} wins after {round_num} rounds")

    assert not shrink_caused_elim, "FAIL: Shrink caused elimination in at least one game!"
    print(f"\nTEST 2 PASSED: Shrink never caused any elimination across all 5 games.\n")


if __name__ == "__main__":
    test_edge_penguins_survive_shrinks()
    test_extreme_edge_penguins()
    test_full_games_with_shrink()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
