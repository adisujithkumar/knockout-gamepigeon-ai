"""Agent tournament benchmark."""

from knockout.training.evaluation import round_robin
from knockout.agents.random_agent import RandomAgent
from knockout.agents.heuristic_agent import HeuristicAgent


def main():
    agents = {
        "random": RandomAgent("r", seed=42),
        "heuristic": HeuristicAgent("h", seed=42),
    }
    results = round_robin(agents, n_games=50)
    print("Tournament Results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["win_rate"], reverse=True):
        print(f"  {name}: {r['wins']}W/{r['losses']}L/{r['draws']}D ({r['win_rate']:.1%})")


if __name__ == "__main__":
    main()
