"""Run a headless game and print results."""

from knockout.visualization.game_viewer import GameViewer


def main():
    viewer = GameViewer(seed=42)
    winner = viewer.run_headless()
    teams = {0: "Team A", 1: "Team B", -1: "Draw"}
    print(f"Game result: {teams.get(winner, 'Unknown')}")


if __name__ == "__main__":
    main()
