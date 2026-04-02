"""Watch a visual game: Heuristic agents (Team A) vs Random agents (Team B)."""

from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.agents.random_agent import RandomAgent
from knockout.env.observations import ObservationBuilder
from knockout.core.physics_engine import PhysicsEngine
from knockout.visualization.game_viewer import GameViewer


def main():
    obs_builder = ObservationBuilder()

    # Create agents for Team A (heuristic) - penguins 0, 1, 2
    team_a_agents = {
        f"penguin_{i}": HeuristicAgent(agent_id=f"penguin_{i}", seed=100 + i)
        for i in range(3)
    }

    # Create agents for Team B (random) - penguins 3, 4, 5
    team_b_agents = {
        f"penguin_{i}": RandomAgent(agent_id=f"penguin_{i}", seed=200 + i)
        for i in range(3, 6)
    }

    def team_a_fn(engine: PhysicsEngine) -> dict[str, tuple[float, float]]:
        actions = {}
        for agent_id, agent in team_a_agents.items():
            penguin = engine.penguins.get(agent_id)
            if penguin is None or not penguin.alive:
                continue
            obs = obs_builder.build_observation(agent_id, engine.penguins, engine.step_count)
            action = agent.get_action(obs)
            actions[agent_id] = (float(action[0]), float(action[1]))
        return actions

    def team_b_fn(engine: PhysicsEngine) -> dict[str, tuple[float, float]]:
        actions = {}
        for agent_id, agent in team_b_agents.items():
            penguin = engine.penguins.get(agent_id)
            if penguin is None or not penguin.alive:
                continue
            obs = obs_builder.build_observation(agent_id, engine.penguins, engine.step_count)
            action = agent.get_action(obs)
            actions[agent_id] = (float(action[0]), float(action[1]))
        return actions

    viewer = GameViewer(seed=42, physics_steps_per_action=5)
    print("Starting game: Heuristic (Team A) vs Random (Team B)")
    winner = viewer.run_visual(team_a_fn=team_a_fn, team_b_fn=team_b_fn)

    teams = {0: "Team A (Heuristic)", 1: "Team B (Random)", -1: "Draw"}
    print(f"Game result: {teams.get(winner, 'Unknown')} wins!")


if __name__ == "__main__":
    main()
