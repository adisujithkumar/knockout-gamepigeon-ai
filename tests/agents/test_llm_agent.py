"""Tests for LLM agent and game state text conversion.

All tests use mocked LLM clients -- no real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from knockout.agents.game_state_text import obs_to_text, _parse_block, FEATURES_PER_PENGUIN
from knockout.agents.llm_agent import LLMAgent, parse_action
from knockout.core.config import DEFAULTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(response_text: str) -> MagicMock:
    """Create a mock Anthropic client that returns the given text."""
    client = MagicMock()
    content_block = MagicMock()
    content_block.text = response_text
    response = MagicMock()
    response.content = [content_block]
    client.messages.create.return_value = response
    return client


def _make_alive_obs() -> np.ndarray:
    """Create a realistic 89-dim observation where all penguins are alive.

    Ego at (-42, 5.3), allies and enemies at various positions.
    """
    obs = np.zeros(89, dtype=np.float32)
    hw = DEFAULTS.ARENA_HALF_WIDTH  # 100

    # Ego (indices 0-13): position (-42, 5.3), alive
    obs[0] = -42.0 / hw      # pos_x
    obs[1] = 5.3 / hw        # pos_y
    obs[2] = 0.0             # vel_x
    obs[3] = 0.0             # vel_y
    obs[4] = 42.0 / hw       # dist_center
    obs[5] = 58.0 / hw       # dist_edge
    obs[6] = 0.0             # speed
    obs[7] = 0.0             # heading
    obs[8] = 1.0             # alive

    # Ally 1 (indices 14-27): position (-38.5, -8.2), alive
    obs[14] = -38.5 / hw
    obs[15] = -8.2 / hw
    obs[18] = 38.5 / hw
    obs[19] = 61.5 / hw
    obs[22] = 1.0            # alive

    # Ally 2 (indices 28-41): dead (all zeros except alive=0)
    # Already zero

    # Enemy 1 (indices 42-55): position (55.2, -12.0), alive
    # block offsets: 0=pos_x, 1=pos_y, 4=dist_center, 5=dist_edge, 8=alive
    obs[42] = 55.2 / hw       # pos_x
    obs[43] = -12.0 / hw      # pos_y
    obs[46] = 55.2 / hw       # dist_center
    obs[47] = 44.8 / hw       # dist_edge
    obs[50] = 1.0             # alive  (42 + 8)

    # Enemy 2 (indices 56-69): position (60.0, 0.0), alive
    obs[56] = 60.0 / hw       # pos_x
    obs[57] = 0.0             # pos_y
    obs[60] = 60.0 / hw       # dist_center
    obs[61] = 40.0 / hw       # dist_edge
    obs[64] = 1.0             # alive  (56 + 8)

    # Enemy 3 (indices 70-83): position (48.7, 15.3), alive
    obs[70] = 48.7 / hw       # pos_x
    obs[71] = 15.3 / hw       # pos_y
    obs[74] = 48.7 / hw       # dist_center
    obs[75] = 51.3 / hw       # dist_edge
    obs[78] = 1.0             # alive  (70 + 8)

    # Global (indices 84-88)
    obs[84] = 2.0 / 3.0      # team_a_alive (2)
    obs[85] = 3.0 / 3.0      # team_b_alive (3)
    obs[86] = 2.0 / 3.0      # ego_team_alive
    obs[87] = 3.0 / 3.0      # opp_team_alive
    obs[88] = 0.0             # timestep

    return obs


# ---------------------------------------------------------------------------
# obs_to_text tests
# ---------------------------------------------------------------------------

class TestObsToText:

    def test_obs_to_text_format(self):
        """Verify text output is well-formatted with positions and alive status."""
        obs = _make_alive_obs()
        text = obs_to_text(obs, "penguin_0")

        assert "penguin_0" in text
        assert "Team A" in text
        assert "YOUR POSITION" in text
        assert "ALLIES" in text
        assert "ENEMIES" in text
        assert "TEAM STATUS" in text
        assert "ELIMINATED" in text  # Ally 2 is dead
        assert "[CLOSEST]" in text

    def test_obs_to_text_denormalization(self):
        """Verify positions are in game coordinates, not normalized [-1,1]."""
        obs = _make_alive_obs()
        text = obs_to_text(obs, "penguin_0")

        # Ego position should be denormalized: (-42.0, 5.3)
        assert "-42.0" in text
        assert "5.3" in text

        # Enemy positions should be denormalized
        assert "55.2" in text  # enemy 1 x
        assert "60.0" in text  # enemy 2 x

        # Normalized values should NOT appear (e.g., -0.42, 0.053)
        assert "-0.42" not in text
        assert "0.053" not in text

    def test_obs_to_text_contains_action_format(self):
        """Verify the prompt tells the LLM about angle/power format."""
        obs = _make_alive_obs()
        text = obs_to_text(obs, "penguin_0")

        assert "angle" in text.lower()
        assert "power" in text.lower()
        assert "0=right" in text or "0 =right" in text or "0=right" in text.replace(" ", "")

    def test_obs_to_text_team_b_agent(self):
        """Team B agents should show 'Team B'."""
        obs = _make_alive_obs()
        # Adjust alive flag for penguin_3 perspective
        obs[8] = 1.0
        text = obs_to_text(obs, "penguin_3")
        assert "Team B" in text

    def test_dead_agent_handling(self):
        """Zero observation (dead agent) produces reasonable text."""
        obs = np.zeros(89, dtype=np.float32)
        text = obs_to_text(obs, "penguin_0")

        assert "ELIMINATED" in text
        assert "penguin_0" in text
        assert "No action required" in text


# ---------------------------------------------------------------------------
# parse_action tests
# ---------------------------------------------------------------------------

class TestParseAction:

    def test_parse_json_action(self):
        """Parse JSON format: {"angle": 45, "power": 300}."""
        text = 'I will aim right. {"angle": 45, "power": 300}'
        result = parse_action(text)

        assert result is not None
        assert result[0] == pytest.approx(45.0)
        assert result[1] == pytest.approx(300.0)

    def test_parse_json_with_floats(self):
        """Parse JSON with float values."""
        text = '{"angle": 123.5, "power": 275.0}'
        result = parse_action(text)

        assert result is not None
        assert result[0] == pytest.approx(123.5)
        assert result[1] == pytest.approx(275.0)

    def test_parse_json_reversed_keys(self):
        """Parse JSON with power before angle."""
        text = '{"power": 200, "angle": 90}'
        result = parse_action(text)

        assert result is not None
        assert result[0] == pytest.approx(90.0)
        assert result[1] == pytest.approx(200.0)

    def test_parse_natural_language(self):
        """Parse natural language: 'launch at 45 degrees with 300 power'."""
        text = "I'll launch at 45 degrees with 300 power to knock them off."
        result = parse_action(text)

        assert result is not None
        assert result[0] == pytest.approx(45.0)
        assert result[1] == pytest.approx(300.0)

    def test_parse_natural_language_angle_power(self):
        """Parse 'angle 180 power 400'."""
        text = "angle 180 power 400"
        result = parse_action(text)

        assert result is not None
        assert result[0] == pytest.approx(180.0)
        assert result[1] == pytest.approx(400.0)

    def test_parse_failure_returns_none(self):
        """Unparseable text returns None."""
        assert parse_action("I don't know what to do") is None
        assert parse_action("") is None
        assert parse_action("just some random text") is None

    def test_parse_json_embedded_in_text(self):
        """JSON embedded in reasoning text."""
        text = (
            "The closest enemy is near the right edge. "
            'I should aim right with full power. {"angle": 0, "power": 400}'
        )
        result = parse_action(text)

        assert result is not None
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# LLMAgent tests (mocked client)
# ---------------------------------------------------------------------------

class TestLLMAgent:

    def test_get_action_returns_correct_shape(self):
        """LLM agent returns np.ndarray of shape (2,)."""
        client = _make_mock_client('{"angle": 90, "power": 200}')
        agent = LLMAgent("penguin_0", client=client)

        obs = _make_alive_obs()
        action = agent.get_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert action[0] == pytest.approx(90.0)
        assert action[1] == pytest.approx(200.0)

    def test_action_bounds_clamped(self):
        """Parsed actions are clamped to valid ranges."""
        client = _make_mock_client('{"angle": 500, "power": 9999}')
        agent = LLMAgent("penguin_0", client=client)

        obs = _make_alive_obs()
        action = agent.get_action(obs)

        assert action[0] <= 360.0
        assert action[1] <= DEFAULTS.MAX_LAUNCH_FORCE

    def test_parse_failure_fallback(self):
        """Invalid LLM response falls back to heuristic agent."""
        client = _make_mock_client("I have no idea what to do lol")
        agent = LLMAgent("penguin_0", client=client, seed=42)

        obs = _make_alive_obs()
        action = agent.get_action(obs)

        # Should still return a valid action (from heuristic)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert 0.0 <= action[0] <= 360.0
        assert 0.0 <= action[1] <= DEFAULTS.MAX_LAUNCH_FORCE
        assert agent.fallback_count == 1

    def test_api_error_fallback(self):
        """API errors fall back to heuristic agent."""
        client = MagicMock()
        client.messages.create.side_effect = Exception("API Error")
        agent = LLMAgent("penguin_0", client=client, seed=42)

        obs = _make_alive_obs()
        action = agent.get_action(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert agent.fallback_count == 1

    def test_dead_agent_returns_noop(self):
        """Dead agent (zero obs) returns [0, 0] without calling LLM."""
        client = _make_mock_client("should not be called")
        agent = LLMAgent("penguin_0", client=client)

        obs = np.zeros(89, dtype=np.float32)
        action = agent.get_action(obs)

        assert action[0] == pytest.approx(0.0)
        assert action[1] == pytest.approx(0.0)
        # LLM should not have been called
        client.messages.create.assert_not_called()

    def test_llm_receives_state_text(self):
        """Verify the LLM is called with the game state text."""
        client = _make_mock_client('{"angle": 45, "power": 300}')
        agent = LLMAgent("penguin_0", client=client)

        obs = _make_alive_obs()
        agent.get_action(obs)

        # Check the LLM was called
        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        user_msg = messages[0]["content"]
        assert "penguin_0" in user_msg
        assert "YOUR POSITION" in user_msg

    def test_reset_clears_reasoning(self):
        """Reset clears per-episode state."""
        client = _make_mock_client('{"angle": 45, "power": 300}')
        agent = LLMAgent("penguin_0", client=client)

        obs = _make_alive_obs()
        agent.get_action(obs)
        assert agent.last_reasoning != ""

        agent.reset()
        assert agent.last_reasoning == ""

    def test_no_api_key_raises(self):
        """Creating agent without API key or client raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            env = dict(__builtins__={})  # not used, just for clarity
            with pytest.raises(ValueError, match="No API key"):
                LLMAgent("penguin_0")

    def test_call_count_tracking(self):
        """Agent tracks number of LLM calls."""
        client = _make_mock_client('{"angle": 0, "power": 100}')
        agent = LLMAgent("penguin_0", client=client)

        obs = _make_alive_obs()
        agent.get_action(obs)
        agent.get_action(obs)
        assert agent.call_count == 2
