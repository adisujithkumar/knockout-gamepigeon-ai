"""Non-interactive game viewer for watching AI vs AI matches."""

from __future__ import annotations

from typing import Optional

import numpy as np

from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.physics_engine import PhysicsEngine


class GameViewer:
    """Watch mode: run a game and optionally render it."""

    def __init__(
        self,
        config: GameConfig = DEFAULTS,
        seed: int = 42,
        physics_steps_per_action: int = 10,
        max_actions: int = 200,
    ):
        self.config = config
        self.seed = seed
        self.physics_steps_per_action = physics_steps_per_action
        self.max_actions = max_actions

    def run_headless(self, team_a_fn=None, team_b_fn=None):
        """Run a game without visualization. Returns winner (0, 1, or -1).

        team_a_fn/team_b_fn: callable(engine) -> dict of agent_id -> (angle, power)
        If None, uses random actions.
        """
        rng = np.random.default_rng(self.seed)
        engine = PhysicsEngine(seed=self.seed, config=self.config)
        engine.initialize_game()

        for action_step in range(self.max_actions):
            if engine.is_game_over():
                break

            actions = {}
            for agent_id, penguin in engine.penguins.items():
                if not penguin.alive:
                    continue
                if penguin.team_id == 0 and team_a_fn:
                    team_actions = team_a_fn(engine)
                    if agent_id in team_actions:
                        actions[agent_id] = team_actions[agent_id]
                        continue
                elif penguin.team_id == 1 and team_b_fn:
                    team_actions = team_b_fn(engine)
                    if agent_id in team_actions:
                        actions[agent_id] = team_actions[agent_id]
                        continue
                # Default: random
                angle = float(rng.uniform(0, 360))
                power = float(rng.uniform(0, self.config.MAX_LAUNCH_FORCE))
                actions[agent_id] = (angle, power)

            engine.apply_actions(actions)
            engine.step_until_settled()

            # Arena shrink after settling (round number = action_step + 1)
            round_num = action_step + 1
            if round_num % self.config.SHRINK_INTERVAL == 0:
                scale = engine.ice_sheet.shrink(
                    self.config.SHRINK_FACTOR,
                    self.config.MIN_ARENA_HALF_WIDTH,
                )
                engine.rescale_penguins(scale)

        winner = engine.get_winner()
        return winner if winner is not None else -1

    def run_visual(self, team_a_fn=None, team_b_fn=None):
        """Run a game with Pygame visualization."""
        from knockout.visualization.renderer import Renderer

        rng = np.random.default_rng(self.seed)
        engine = PhysicsEngine(seed=self.seed, config=self.config)
        engine.initialize_game()

        renderer = Renderer(config=self.config)

        try:
            for action_step in range(self.max_actions):
                if renderer.check_quit():
                    break
                if engine.is_game_over():
                    # Show final frame for a bit
                    for _ in range(60):
                        renderer.draw_frame(engine)
                        renderer.tick()
                        if renderer.check_quit():
                            break
                    break

                actions = {}
                for agent_id, penguin in engine.penguins.items():
                    if not penguin.alive:
                        continue
                    if penguin.team_id == 0 and team_a_fn:
                        team_actions = team_a_fn(engine)
                        if agent_id in team_actions:
                            actions[agent_id] = team_actions[agent_id]
                            continue
                    elif penguin.team_id == 1 and team_b_fn:
                        team_actions = team_b_fn(engine)
                        if agent_id in team_actions:
                            actions[agent_id] = team_actions[agent_id]
                            continue
                    angle = float(rng.uniform(0, 360))
                    power = float(rng.uniform(0, self.config.MAX_LAUNCH_FORCE))
                    actions[agent_id] = (angle, power)

                engine.apply_actions(actions)

                # Animate settling: step one physics frame at a time and render
                hud_text = f"Action: {action_step + 1}/{self.max_actions}"
                settled = False
                settle_steps = 0
                while not settled and settle_steps < 3000:
                    engine.step(num_steps=1)
                    settle_steps += 1
                    renderer.draw_frame(
                        engine,
                        extra_text=f"{hud_text} - Settling...",
                        actions=actions,
                    )
                    # Render every 2nd frame to keep it fast but visible
                    if settle_steps % 2 == 0:
                        renderer.tick()
                    if renderer.check_quit():
                        renderer.close()
                        return engine.get_winner() or -1
                    if engine.are_all_settled():
                        settled = True

                # Arena shrink after settling (round number = action_step + 1)
                round_num = action_step + 1
                if round_num % self.config.SHRINK_INTERVAL == 0:
                    scale = engine.ice_sheet.shrink(
                        self.config.SHRINK_FACTOR,
                        self.config.MIN_ARENA_HALF_WIDTH,
                    )
                    engine.rescale_penguins(scale)

                # Brief pause after settling so viewer can see the result
                for _ in range(30):
                    renderer.draw_frame(
                        engine,
                        extra_text=f"{hud_text} - Settled",
                        actions=actions,
                    )
                    renderer.tick()
                    if renderer.check_quit():
                        renderer.close()
                        return engine.get_winner() or -1
        finally:
            renderer.close()

        winner = engine.get_winner()
        return winner if winner is not None else -1


def main():
    """Entry point for knockout-watch command."""
    viewer = GameViewer(seed=42)
    winner = viewer.run_headless()
    teams = {0: "Team A", 1: "Team B", -1: "Draw"}
    print(f"Game result: {teams.get(winner, 'Unknown')}")
