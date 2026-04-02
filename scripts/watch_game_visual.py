"""Run a visual game with Pygame."""

from knockout.visualization.game_viewer import GameViewer


def main():
    viewer = GameViewer(seed=42, physics_steps_per_action=5)
    winner = viewer.run_visual()
    teams = {0: "Team A", 1: "Team B", -1: "Draw"}
    print(f"Game result: {teams.get(winner, 'Unknown')}")


if __name__ == "__main__":
    main()
