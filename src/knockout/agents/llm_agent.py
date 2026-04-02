"""LLM-powered agent for penguin knockout game.

Uses the Anthropic SDK to call Claude for action decisions.
The LLM sees a text description of the game state and responds
with an angle and power.
"""

from __future__ import annotations

import json
import logging
import os
import re

import numpy as np

from knockout.agents.base import Agent
from knockout.agents.game_state_text import obs_to_text
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.core.config import GameConfig, DEFAULTS

logger = logging.getLogger(__name__)

# Default system prompt that teaches the LLM how to play.
DEFAULT_SYSTEM_PROMPT = """\
You are playing a 3v3 penguin knockout game on a square ice arena.

RULES:
- Each round, every penguin simultaneously chooses an angle and power to launch.
- After launching, penguins slide on ice with friction until they stop.
- A penguin that slides off the arena edge is eliminated.
- The team that eliminates all opposing penguins wins.
- The arena shrinks every 5 rounds.

COORDINATE SYSTEM:
- The arena is a square centered at (0, 0).
- Positive X = right, Positive Y = up.
- Angle 0 = right, 90 = up, 180 = left, 270 = down.

STRATEGY TIPS:
- Aim to knock enemies off the edge while keeping yourself safe.
- Enemies near the edge are easier to eliminate.
- Be careful not to launch yourself off the edge.
- Higher power = farther travel. Max power is 400.
- Consider the positions of all penguins before choosing your action.

RESPONSE FORMAT:
Respond with a brief reasoning (1-2 sentences), then your action as JSON:
{"angle": <degrees 0-360>, "power": <newtons 0-400>}
"""


class LLMAgent(Agent):
    """Agent that uses an LLM to decide actions.

    Converts the 89-dim observation to readable text, sends it to the
    LLM, and parses the response back to [angle, power].

    On parse failure, falls back to a heuristic agent action.
    """

    def __init__(
        self,
        agent_id: str,
        client: object | None = None,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: str | None = None,
        tools_tier: int = 0,
        config: GameConfig = DEFAULTS,
        seed: int | None = None,
        api_key: str | None = None,
    ):
        """Initialize the LLM agent.

        Args:
            agent_id: e.g. "penguin_0"
            client: anthropic.Anthropic client instance. If None, one is
                created using the api_key or ANTHROPIC_API_KEY env var.
            model: Model identifier to use for completions.
            system_prompt: Custom system prompt. Uses DEFAULT_SYSTEM_PROMPT
                if not provided.
            tools_tier: 0 = raw state only (tool tiers added later).
            config: Game configuration.
            seed: Random seed for the fallback heuristic agent.
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY
                env var if not provided.
        """
        super().__init__(agent_id)
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.tools_tier = tools_tier
        self.config = config

        # Build or accept the Anthropic client
        if client is not None:
            self.client = client
        else:
            import anthropic

            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "No API key provided. Pass api_key= or set "
                    "ANTHROPIC_API_KEY environment variable."
                )
            self.client = anthropic.Anthropic(api_key=key)

        # Fallback heuristic for parse failures
        self._fallback = HeuristicAgent(agent_id, seed=seed, config=config)

        # Logging
        self.last_reasoning: str = ""
        self.last_raw_response: str = ""
        self.fallback_count: int = 0
        self.call_count: int = 0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Convert obs to text, call LLM, parse response to [angle, power].

        Args:
            observation: 89-dimensional observation vector.

        Returns:
            Action array [angle_degrees, power_newtons].
        """
        self.call_count += 1

        # Convert observation to readable text
        state_text = obs_to_text(observation, self.agent_id, self.config)

        # Check if dead (zero observation)
        if np.allclose(observation, 0.0):
            logger.info("[%s] Agent is eliminated, returning no-op.", self.agent_id)
            self.last_reasoning = "Eliminated, no action."
            return np.array([0.0, 0.0], dtype=np.float32)

        # Call LLM
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=self.system_prompt,
                messages=[{"role": "user", "content": state_text}],
            )
            raw_text = response.content[0].text
            self.last_raw_response = raw_text
            logger.info("[%s] LLM response: %s", self.agent_id, raw_text)
        except Exception as e:
            logger.warning("[%s] LLM call failed: %s. Using fallback.", self.agent_id, e)
            self.fallback_count += 1
            return self._fallback.get_action(observation)

        # Parse action from response
        action = parse_action(raw_text)
        if action is not None:
            angle, power = action
            # Clamp to valid ranges
            angle = float(np.clip(angle, 0.0, 360.0))
            power = float(np.clip(power, 0.0, self.config.MAX_LAUNCH_FORCE))
            self.last_reasoning = raw_text
            return np.array([angle, power], dtype=np.float32)

        # Parse failed -- fall back to heuristic
        logger.warning(
            "[%s] Could not parse action from LLM response. Using fallback. "
            "Response was: %s",
            self.agent_id,
            raw_text,
        )
        self.last_reasoning = f"[FALLBACK] {raw_text}"
        self.fallback_count += 1
        return self._fallback.get_action(observation)

    def reset(self) -> None:
        """Reset per-episode state."""
        self.last_reasoning = ""
        self.last_raw_response = ""


def parse_action(text: str) -> tuple[float, float] | None:
    """Parse angle and power from LLM response text.

    Supports two formats:
    1. JSON: {"angle": 45, "power": 300}
    2. Natural language: "launch at 45 degrees with power 300"

    Args:
        text: Raw LLM response text.

    Returns:
        (angle, power) tuple, or None if parsing fails.
    """
    # Strategy 1: Try to find JSON object with angle and power
    json_match = re.search(r'\{[^{}]*"angle"\s*:\s*[\d.]+[^{}]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "angle" in data and "power" in data:
                return (float(data["angle"]), float(data["power"]))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Also try with keys in other order
    json_match = re.search(r'\{[^{}]*"power"\s*:\s*[\d.]+[^{}]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "angle" in data and "power" in data:
                return (float(data["angle"]), float(data["power"]))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 2: Natural language patterns
    # "angle 45 power 300", "45 degrees with power 300", etc.
    angle_patterns = [
        r'angle\s*(?:of\s*)?(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*degrees',
        r'at\s+(\d+(?:\.\d+)?)\s*(?:deg|degree)',
        r'direction\s*(?:of\s*)?(\d+(?:\.\d+)?)',
    ]
    power_patterns = [
        r'power\s*(?:of\s*)?(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:power|newtons?|N\b)',
        r'force\s*(?:of\s*)?(\d+(?:\.\d+)?)',
        r'strength\s*(?:of\s*)?(\d+(?:\.\d+)?)',
    ]

    angle_val = None
    power_val = None

    for pattern in angle_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            angle_val = float(m.group(1))
            break

    for pattern in power_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            power_val = float(m.group(1))
            break

    if angle_val is not None and power_val is not None:
        return (angle_val, power_val)

    return None
