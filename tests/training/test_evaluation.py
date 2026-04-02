"""Tests for evaluation harness."""

import pytest

from knockout.agents.random_agent import RandomAgent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.training.evaluation import run_match, round_robin


class TestEvaluation:
    """Tests for evaluation functions."""

    def test_run_match(self):
        """Test single match returns expected result dict."""
        agent_a = RandomAgent("a", seed=1)
        agent_b = RandomAgent("b", seed=2)

        team_a = {f"penguin_{i}": agent_a for i in range(3)}
        team_b = {f"penguin_{i}": agent_b for i in range(3, 6)}

        result = run_match(team_a, team_b, seed=42)

        assert "winner" in result
        assert "steps" in result
        assert "team_a_alive" in result
        assert "team_b_alive" in result
        assert result["winner"] in (0, 1, -1)
        assert result["steps"] > 0
        assert 0 <= result["team_a_alive"] <= 3
        assert 0 <= result["team_b_alive"] <= 3

    def test_run_match_deterministic(self):
        """Test same seed produces same result."""
        agent_a = RandomAgent("a", seed=10)
        agent_b = RandomAgent("b", seed=20)

        team_a = {f"penguin_{i}": agent_a for i in range(3)}
        team_b = {f"penguin_{i}": agent_b for i in range(3, 6)}

        result1 = run_match(team_a, team_b, seed=42)

        # Reset agent RNGs
        agent_a = RandomAgent("a", seed=10)
        agent_b = RandomAgent("b", seed=20)
        team_a = {f"penguin_{i}": agent_a for i in range(3)}
        team_b = {f"penguin_{i}": agent_b for i in range(3, 6)}

        result2 = run_match(team_a, team_b, seed=42)

        assert result1["winner"] == result2["winner"]
        assert result1["steps"] == result2["steps"]

    def test_round_robin_two_agents(self):
        """Test round robin with two agents runs to completion."""
        agents = {
            "random": RandomAgent("r", seed=42),
            "heuristic": HeuristicAgent("h", seed=42),
        }

        results = round_robin(agents, n_games=2)

        assert "random" in results
        assert "heuristic" in results

        # Each agent plays 2 games as A and 2 as B = 4 games total
        assert results["random"]["games"] == 4
        assert results["heuristic"]["games"] == 4

    def test_round_robin_results_structure(self):
        """Test round robin returns correct result structure."""
        agents = {
            "a": RandomAgent("a", seed=1),
            "b": RandomAgent("b", seed=2),
        }

        results = round_robin(agents, n_games=1)

        for name in ["a", "b"]:
            r = results[name]
            assert "wins" in r
            assert "losses" in r
            assert "draws" in r
            assert "games" in r
            assert "win_rate" in r
            assert r["wins"] + r["losses"] + r["draws"] == r["games"]
            assert 0.0 <= r["win_rate"] <= 1.0
