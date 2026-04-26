"""LLM Reward Architect: zero-knowledge reward discovery via LLM iteration.

An LLM observes game statistics and writes Python reward functions.
It iterates based on training outcomes, generating rich
(prompt, reasoning, code, outcome) tuples for foundation model training data.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from knockout.core.config import GameConfig, DEFAULTS

logger = logging.getLogger(__name__)


# ---- Observation feature map ------------------------------------------------

_PENGUIN_FEATURES = [
    ("position_x", "normalised x position", (-1.0, 1.0)),
    ("position_y", "normalised y position", (-1.0, 1.0)),
    ("velocity_x", "normalised x velocity", (-1.0, 1.0)),
    ("velocity_y", "normalised y velocity", (-1.0, 1.0)),
    ("distance_from_center", "Chebyshev distance from center, normalised", (0.0, 1.0)),
    ("distance_to_edge", "distance to nearest boundary, normalised", (-1.0, 1.0)),
    ("speed", "scalar speed, normalised", (0.0, 1.0)),
    ("heading", "velocity angle / pi", (-1.0, 1.0)),
    ("alive", "binary alive flag", (0.0, 1.0)),
    ("relative_position_x", "x offset from ego agent, normalised", (-1.0, 1.0)),
    ("relative_position_y", "y offset from ego agent, normalised", (-1.0, 1.0)),
    ("relative_velocity_x", "x velocity relative to ego, normalised", (-1.0, 1.0)),
    ("relative_velocity_y", "y velocity relative to ego, normalised", (-1.0, 1.0)),
    ("distance_to_ego", "Euclidean distance to ego, normalised", (0.0, 1.0)),
]

_GLOBAL_FEATURES = [
    ("team_a_alive_count", "fraction of team A agents alive", (0.0, 1.0)),
    ("team_b_alive_count", "fraction of team B agents alive", (0.0, 1.0)),
    ("ego_team_alive_count", "fraction of ego team alive", (0.0, 1.0)),
    ("opponent_team_alive_count", "fraction of opponent team alive", (0.0, 1.0)),
    ("timestep", "normalised simulation timestep", (0.0, 1.0)),
]


def build_obs_description() -> str:
    """Build a zero-knowledge feature description for the LLM.

    Does NOT mention 'knockout', 'penguin', 'ice', or game mechanics.
    """
    lines: list[str] = []
    idx = 0

    groups = [
        ("ego_agent", "Self-agent features", 1),
        ("ally", "Allied agent features (sorted by proximity)", 2),
        ("opponent", "Opponent agent features (sorted by proximity)", 3),
    ]

    for group_name, group_desc, count in groups:
        for agent_i in range(count):
            label = f"{group_name}_{agent_i}" if count > 1 else group_name
            for name, desc, (lo, hi) in _PENGUIN_FEATURES:
                lines.append(
                    f"  Feature {idx}: {label}.{name} -- {desc} (range [{lo}, {hi}])"
                )
                idx += 1

    lines.append("")
    lines.append("  Global state features:")
    for name, desc, (lo, hi) in _GLOBAL_FEATURES:
        lines.append(
            f"  Feature {idx}: global.{name} -- {desc} (range [{lo}, {hi}])"
        )
        idx += 1

    assert idx == 89, f"Expected 89 features, got {idx}"

    header = (
        "The environment has a {obs_dim}-dimensional observation vector and "
        "a 2-dimensional continuous action vector.\n"
        "Observation layout:\n"
    ).format(obs_dim=89)

    return header + "\n".join(lines)


# ---- Config ------------------------------------------------------------------


@dataclass(frozen=True)
class LLMArchitectConfig:
    """Configuration for the LLM Reward Architect outer loop."""

    num_iterations: int = 10
    episodes_per_eval: int = 500
    steps_per_iteration: int = 1_000_000
    llm_model: str = "claude-sonnet-4-20250514"
    num_candidates: int = 3
    candidate_eval_steps: int = 200_000
    temperature: float = 0.7
    max_retries_per_candidate: int = 3
    output_dir: str = "runs/llm_architect"
    device: str = "cpu"
    num_envs: int = 64


# ---- Game statistics collector -----------------------------------------------


class GameStatsCollector:
    """Collects statistics from trajectories for LLM analysis.

    Runs a policy in the TensorVecEnv and records aggregate statistics
    without revealing game-specific semantics.
    """

    def collect(
        self,
        env: Any,
        policy: Callable[[np.ndarray], np.ndarray],
        num_episodes: int,
    ) -> dict[str, Any]:
        """Collect game statistics by running the policy.

        Args:
            env: A TensorVecEnv (or compatible) instance.
            policy: Callable (N*3, 89) -> (N*3, 2) returning raw actions.
            num_episodes: Minimum number of completed episodes to collect.

        Returns:
            Dictionary of aggregate statistics.
        """
        obs, masks = env.reset()

        episodes_done = 0
        total_wins = 0
        total_losses = 0
        total_draws = 0
        episode_lengths: list[int] = []
        round_counters = np.zeros(env.num_envs, dtype=np.int32)

        # Accumulate per-feature statistics
        all_obs: list[np.ndarray] = []
        win_terminal_obs: list[np.ndarray] = []
        loss_terminal_obs: list[np.ndarray] = []
        alive_counts_at_end: list[tuple[int, int]] = []

        max_steps = max(num_episodes * 50, 5000)  # safety cap
        for _ in range(max_steps):
            if episodes_done >= num_episodes:
                break

            obs_np = np.asarray(obs)
            flat_obs = obs_np.reshape(-1, 89)
            all_obs.append(flat_obs.copy())

            raw_actions = policy(flat_obs)
            actions_3d = raw_actions.reshape(env.num_envs, 3, 2)

            obs, rewards, dones, masks, infos = env.step(actions_3d)
            round_counters += 1

            for i, info in enumerate(infos):
                if dones[i] and "winner" in info:
                    episodes_done += 1
                    episode_lengths.append(int(round_counters[i]))
                    round_counters[i] = 0

                    if info["winner"] == 0:
                        total_wins += 1
                        win_terminal_obs.append(obs_np[i].reshape(-1, 89).copy())
                    elif info["winner"] == 1:
                        total_losses += 1
                        loss_terminal_obs.append(obs_np[i].reshape(-1, 89).copy())
                    else:
                        total_draws += 1

                    alive_counts_at_end.append(
                        (info.get("team_a_alive", 0), info.get("team_b_alive", 0))
                    )

        total_games = max(total_wins + total_losses + total_draws, 1)

        # Compute per-feature statistics
        all_obs_arr = np.concatenate(all_obs, axis=0) if all_obs else np.zeros((1, 89))
        feature_means = all_obs_arr.mean(axis=0).tolist()
        feature_stds = all_obs_arr.std(axis=0).tolist()

        # Win vs loss feature comparison
        win_feature_means = None
        loss_feature_means = None
        if win_terminal_obs:
            win_arr = np.concatenate(win_terminal_obs, axis=0)
            win_feature_means = win_arr.mean(axis=0).tolist()
        if loss_terminal_obs:
            loss_arr = np.concatenate(loss_terminal_obs, axis=0)
            loss_feature_means = loss_arr.mean(axis=0).tolist()

        avg_team_a_alive = (
            np.mean([a for a, _ in alive_counts_at_end])
            if alive_counts_at_end else 0.0
        )
        avg_team_b_alive = (
            np.mean([b for _, b in alive_counts_at_end])
            if alive_counts_at_end else 0.0
        )

        return {
            "total_episodes": episodes_done,
            "win_rate": total_wins / total_games,
            "loss_rate": total_losses / total_games,
            "draw_rate": total_draws / total_games,
            "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "median_episode_length": float(np.median(episode_lengths)) if episode_lengths else 0.0,
            "avg_team_a_alive_at_end": float(avg_team_a_alive),
            "avg_team_b_alive_at_end": float(avg_team_b_alive),
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "win_terminal_feature_means": win_feature_means,
            "loss_terminal_feature_means": loss_feature_means,
        }


# ---- Reward function sandbox -------------------------------------------------


class RewardFunctionSandbox:
    """Safely compiles and validates LLM-generated reward functions.

    The generated function signature:
        def reward(obs: torch.Tensor) -> torch.Tensor
    where obs is (B, 89) and return is (B,).

    Safety: AST-level whitelist blocks file I/O, network, exec/eval, subprocess.
    Rewards are clipped to [-MAX_REWARD_MAGNITUDE, MAX_REWARD_MAGNITUDE].
    """

    ALLOWED_IMPORTS = frozenset({"math", "torch"})
    MAX_REWARD_MAGNITUDE = 10.0

    # AST node types and names that are forbidden
    _BLOCKED_NAMES = frozenset({
        "open", "exec", "eval", "compile", "__import__",
        "subprocess", "os", "sys", "shutil", "pathlib",
        "socket", "http", "urllib", "requests",
        "globals", "locals", "vars", "dir",
        "getattr", "setattr", "delattr",
        "breakpoint", "exit", "quit",
    })

    _BLOCKED_MODULES = frozenset({
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "http", "urllib", "requests", "io",
        "ctypes", "importlib", "builtins", "signal",
        "multiprocessing", "threading",
    })

    def validate_ast(self, code: str) -> list[str]:
        """Parse and validate the AST of the reward function code.

        Returns a list of error strings (empty means valid).
        """
        errors: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"SyntaxError: {e}"]

        for node in ast.walk(tree):
            # Block dangerous calls
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in self._BLOCKED_NAMES:
                    errors.append(f"Blocked function call: {func.id}")
                elif isinstance(func, ast.Attribute) and func.attr in self._BLOCKED_NAMES:
                    errors.append(f"Blocked attribute call: {func.attr}")

            # Block forbidden imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module not in self.ALLOWED_IMPORTS:
                        errors.append(f"Blocked import: {alias.name}")

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    if root_module not in self.ALLOWED_IMPORTS:
                        errors.append(f"Blocked import from: {node.module}")

        return errors

    def compile_reward_function(self, code: str) -> Callable:
        """Compile LLM-generated code into a callable reward function.

        Args:
            code: Python source code containing a function named 'reward'.

        Returns:
            A callable reward(obs: Tensor) -> Tensor.

        Raises:
            ValueError: If the code is invalid or unsafe.
        """
        # AST validation
        ast_errors = self.validate_ast(code)
        if ast_errors:
            raise ValueError(
                f"Code failed safety validation: {'; '.join(ast_errors)}"
            )

        # Compile in a restricted namespace
        namespace: dict[str, Any] = {"math": math, "torch": torch}
        try:
            exec(code, namespace)  # noqa: S102
        except Exception as e:
            raise ValueError(f"Code execution error: {e}") from e

        if "reward" not in namespace:
            raise ValueError("Code must define a function named 'reward'")

        raw_fn = namespace["reward"]
        if not callable(raw_fn):
            raise ValueError("'reward' must be callable")

        # Wrap with magnitude clipping
        max_mag = self.MAX_REWARD_MAGNITUDE

        def clipped_reward(obs: torch.Tensor) -> torch.Tensor:
            result = raw_fn(obs)
            return result.clamp(-max_mag, max_mag)

        return clipped_reward

    def validate_output(
        self,
        reward_fn: Callable,
        sample_obs: torch.Tensor | None = None,
    ) -> bool:
        """Run the compiled reward function on sample data and check output.

        Args:
            reward_fn: Compiled reward function.
            sample_obs: Optional (B, 89) tensor. If None, random data is used.

        Returns:
            True if the function produces valid output.

        Raises:
            ValueError: If the function produces invalid output.
        """
        if sample_obs is None:
            sample_obs = torch.randn(32, 89)

        B = sample_obs.shape[0]

        try:
            result = reward_fn(sample_obs)
        except Exception as e:
            raise ValueError(f"Reward function raised error on sample input: {e}") from e

        if not isinstance(result, torch.Tensor):
            raise ValueError(
                f"Reward function must return a torch.Tensor, got {type(result)}"
            )

        if result.shape != (B,):
            raise ValueError(
                f"Reward function must return shape ({B},), got {result.shape}"
            )

        if torch.isnan(result).any():
            raise ValueError("Reward function produced NaN values")

        if torch.isinf(result).any():
            raise ValueError("Reward function produced infinite values")

        return True


# ---- LLM reward generator ---------------------------------------------------


_SYSTEM_PROMPT = """\
You are a reward function engineer for a multi-agent reinforcement learning system.

Your task is to write a Python reward shaping function that, when added to a \
sparse terminal reward (+1 win, -1 loss), produces agents that win more often.

The function signature is:
    def reward(obs: torch.Tensor) -> torch.Tensor
where obs has shape (B, 89) and the return has shape (B,).

You may only use 'torch' and 'math' as imports. The function must be pure \
(no side effects, no file I/O, no network calls).

Output rewards should be small in magnitude (typically -1 to +1). They will \
be clipped to [-10, 10].

Respond with your reasoning, then the complete Python code block."""

_INITIAL_USER_PROMPT = """\
Design an initial reward shaping function for this environment.

{obs_description}

You have NO information about the game rules or objectives yet. \
Design a reward function based on the observation structure alone. \
Use spatial and relational features to encourage potentially useful behaviors.

Write your reasoning, then a single Python code block with:
```python
import torch

def reward(obs: torch.Tensor) -> torch.Tensor:
    ...
```"""

_EVOLVE_USER_PROMPT = """\
Here is the current reward function:

```python
{previous_code}
```

Training metrics with this reward:
- Win rate vs random opponent: {win_rate:.1%}
- Average episode length: {avg_episode_length:.1f} rounds
- Average ego team alive at end: {avg_team_a_alive:.2f} / 3
- Average opponent team alive at end: {avg_team_b_alive:.2f} / 3

Game statistics from {total_episodes} episodes:
- Feature means (across all timesteps): {feature_summary}
- Feature means at WIN terminal states: {win_summary}
- Feature means at LOSS terminal states: {loss_summary}

{obs_description}

Analyze what the current reward function does well and poorly. \
Then write an improved version. Focus on behaviors that correlate with winning.

Write your reasoning, then a single Python code block with:
```python
import torch

def reward(obs: torch.Tensor) -> torch.Tensor:
    ...
```"""


def _format_feature_summary(means: list[float] | None, top_n: int = 15) -> str:
    """Format top-N most informative features as a compact string."""
    if means is None:
        return "(no data)"
    # Show features with largest absolute mean (most signal)
    indexed = sorted(enumerate(means), key=lambda x: abs(x[1]), reverse=True)
    parts = [f"feat[{i}]={v:.3f}" for i, v in indexed[:top_n]]
    return ", ".join(parts)


class LLMRewardGenerator:
    """Calls LLM to generate and evolve reward functions."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._obs_description = build_obs_description()

    def generate_initial(self) -> list[str]:
        """Generate initial reward function candidates.

        Returns:
            List of Python code strings, each defining a 'reward' function.
        """
        responses = []
        for attempt in range(self.max_retries):
            response_text = self._call_llm(
                _SYSTEM_PROMPT,
                _INITIAL_USER_PROMPT.format(obs_description=self._obs_description),
            )
            code = self._extract_code(response_text)
            if code is not None:
                responses.append(code)
                if len(responses) >= 1:
                    break
            else:
                logger.warning(
                    "Failed to extract code from LLM response (attempt %d/%d)",
                    attempt + 1, self.max_retries,
                )

        if not responses:
            raise RuntimeError(
                "LLM failed to produce valid code after all retries"
            )
        return responses

    def evolve(
        self,
        previous_code: str,
        previous_metrics: dict[str, Any],
        game_stats: dict[str, Any],
    ) -> tuple[list[str], str, str]:
        """Evolve a reward function based on training results.

        Args:
            previous_code: The Python source of the current reward function.
            previous_metrics: Training metrics (policy_loss, etc.).
            game_stats: Statistics from GameStatsCollector.

        Returns:
            Tuple of (code_candidates, prompt_text, response_text).
        """
        feature_summary = _format_feature_summary(
            game_stats.get("feature_means")
        )
        win_summary = _format_feature_summary(
            game_stats.get("win_terminal_feature_means")
        )
        loss_summary = _format_feature_summary(
            game_stats.get("loss_terminal_feature_means")
        )

        user_prompt = _EVOLVE_USER_PROMPT.format(
            previous_code=previous_code,
            win_rate=game_stats.get("win_rate", 0.0),
            avg_episode_length=game_stats.get("avg_episode_length", 0.0),
            avg_team_a_alive=game_stats.get("avg_team_a_alive_at_end", 0.0),
            avg_team_b_alive=game_stats.get("avg_team_b_alive_at_end", 0.0),
            total_episodes=game_stats.get("total_episodes", 0),
            feature_summary=feature_summary,
            win_summary=win_summary,
            loss_summary=loss_summary,
            obs_description=self._obs_description,
        )

        candidates: list[str] = []
        full_response = ""
        for attempt in range(self.max_retries):
            response_text = self._call_llm(_SYSTEM_PROMPT, user_prompt)
            full_response = response_text
            code = self._extract_code(response_text)
            if code is not None:
                candidates.append(code)
                if len(candidates) >= 1:
                    break
            else:
                logger.warning(
                    "Failed to extract code on evolution attempt %d/%d",
                    attempt + 1, self.max_retries,
                )

        return candidates, user_prompt, full_response

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Anthropic API with retry and backoff.

        Returns the text content of the response.

        Raises:
            RuntimeError: If the API key is not set or all retries fail.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Use --fallback-reward to run without an API key."
            )

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                message = client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                # Extract text from content blocks
                parts = []
                for block in message.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return "\n".join(parts)
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "LLM API call failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1, self.max_retries, e, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"LLM API call failed after {self.max_retries} retries: {last_error}"
        )

    @staticmethod
    def _extract_code(response: str) -> str | None:
        """Extract Python code from an LLM response.

        Looks for ```python ... ``` blocks and returns the last one
        (which is typically the final/complete version).
        """
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            return None
        # Return the last code block (most likely the final version)
        return matches[-1].strip()


# ---- Fallback reward generator -----------------------------------------------

# Hardcoded reward function templates that simulate LLM discovery progression.
# Each iteration builds on the previous one, adding more reward terms.
_FALLBACK_REWARD_TEMPLATES = [
    # Iteration 0: basic survival — reward for staying near center
    (
        "import torch\n\n"
        "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
        "    # Reward for low distance_from_center (feature 4) -- basic survival\n"
        "    return -obs[:, 4] * 0.5\n"
    ),
    # Iteration 1: add reward for enemies being far from center
    (
        "import torch\n\n"
        "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
        "    # Reward for staying near center + opponents being far from center\n"
        "    center_bonus = -obs[:, 4] * 0.3\n"
        "    enemy_far = obs[:, 47] * 0.3  # opponent_0.distance_to_ego (proxy for spread)\n"
        "    return center_bonus + enemy_far\n"
    ),
    # Iteration 2: add penalty for being near edge
    (
        "import torch\n\n"
        "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
        "    # Multi-objective: center positioning + opponent distance + edge avoidance\n"
        "    center_bonus = -obs[:, 4] * 0.2\n"
        "    enemy_far = obs[:, 47] * 0.3\n"
        "    edge_penalty = (1.0 - obs[:, 5]) * 0.2  # feature 5 = distance_to_edge\n"
        "    return center_bonus + enemy_far + edge_penalty\n"
    ),
]


class FallbackRewardGenerator:
    """Generates reward functions from hardcoded templates when no API key is available.

    Simulates the iterative discovery progression that an LLM would make,
    using a fixed sequence of increasingly sophisticated reward functions.
    """

    def __init__(self) -> None:
        self._templates = list(_FALLBACK_REWARD_TEMPLATES)
        self._call_count = 0

    def generate_initial(self) -> list[str]:
        """Return the first fallback reward function template."""
        logger.info("FallbackRewardGenerator: returning template 0 (basic survival)")
        self._call_count = 0
        return [self._templates[0]]

    def evolve(
        self,
        previous_code: str,
        previous_metrics: dict[str, Any],
        game_stats: dict[str, Any],
    ) -> tuple[list[str], str, str]:
        """Return the next fallback template in the progression.

        Cycles through templates, returning to the last one once exhausted.
        """
        self._call_count += 1
        idx = min(self._call_count, len(self._templates) - 1)
        logger.info(
            "FallbackRewardGenerator: returning template %d (of %d)",
            idx, len(self._templates),
        )

        prompt_text = f"(fallback mode: template {idx})"
        response_text = (
            f"(fallback mode: using hardcoded reward template {idx}. "
            f"Previous win_rate={game_stats.get('win_rate', 0.0):.1%})"
        )
        return [self._templates[idx]], prompt_text, response_text


# ---- Outer loop trainer ------------------------------------------------------


class LLMArchitectTrainer:
    """Outer loop: generate -> train -> evaluate -> evolve -> repeat.

    Each iteration:
    1. Generate/evolve reward function candidates via LLM.
    2. Validate candidates in sandbox.
    3. Quick-train each candidate, pick the best by win rate.
    4. Full train with the best candidate.
    5. Collect game stats, save iteration artifacts.
    6. Feed stats back to LLM for next iteration.
    """

    def __init__(
        self,
        architect_config: LLMArchitectConfig | None = None,
        game_config: GameConfig = DEFAULTS,
        fallback_reward: bool = False,
    ) -> None:
        self.config = architect_config or LLMArchitectConfig()
        self.game_config = game_config

        self.sandbox = RewardFunctionSandbox()
        self.stats_collector = GameStatsCollector()
        self.ppo: Any = None  # Set during _train_with_reward

        # Use fallback generator when requested or when API key is absent
        if fallback_reward or not os.environ.get("ANTHROPIC_API_KEY"):
            if not fallback_reward:
                logger.warning(
                    "ANTHROPIC_API_KEY not set, falling back to "
                    "hardcoded reward templates."
                )
            logger.info("Using FallbackRewardGenerator (no LLM calls).")
            self.generator: LLMRewardGenerator | FallbackRewardGenerator = (
                FallbackRewardGenerator()
            )
        else:
            self.generator = LLMRewardGenerator(
                model=self.config.llm_model,
                temperature=self.config.temperature,
            )

        self.output_dir = Path(self.config.output_dir)
        self.history: list[dict[str, Any]] = []

    def run(self) -> list[dict[str, Any]]:
        """Run the full LLM Reward Architect loop.

        Returns:
            List of per-iteration result dicts.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        current_code: str | None = None
        current_metrics: dict[str, Any] = {}
        current_stats: dict[str, Any] = {}

        for iteration in range(self.config.num_iterations):
            logger.info("=== LLM Architect Iteration %d ===", iteration)
            iter_dir = self.output_dir / f"iteration_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # --- 1. Generate or evolve candidates ---
            prompt_text = ""
            response_text = ""
            if current_code is None:
                # First iteration: generate initial candidates
                logger.info("Generating initial reward function candidates...")
                candidates = self._generate_initial_candidates()
                prompt_text = "(initial generation)"
                response_text = "(initial generation)"
            else:
                # Evolve from previous iteration
                logger.info("Evolving reward function...")
                candidates, prompt_text, response_text = (
                    self._evolve_candidates(
                        current_code, current_metrics, current_stats
                    )
                )

            if not candidates:
                logger.error(
                    "No valid candidates at iteration %d, skipping.", iteration
                )
                continue

            # --- 2. Pick best candidate (quick eval) ---
            best_code, best_fn, best_win_rate = self._select_best_candidate(
                candidates
            )

            # --- 3. Full training with best candidate ---
            logger.info(
                "Full training with best candidate (win_rate=%.2f from quick eval)...",
                best_win_rate,
            )
            trainer_metrics = self._train_with_reward(
                best_fn, self.config.steps_per_iteration
            )

            # --- 4. Collect game stats ---
            logger.info("Collecting game statistics...")
            game_stats = self._collect_stats(best_fn)

            # --- 5. Save artifacts ---
            iteration_record = {
                "iteration": iteration,
                "prompt": prompt_text,
                "llm_reasoning": response_text,
                "reward_code": best_code,
                "metrics": {
                    **trainer_metrics,
                    "quick_eval_win_rate": best_win_rate,
                },
                "game_stats": game_stats,
            }

            self._save_iteration(iter_dir, iteration_record)
            self.history.append(iteration_record)

            # Save agent checkpoint per iteration
            ckpt_path = iter_dir / "agent_checkpoint.pt"
            self.ppo.agent.save(ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

            # Update state for next iteration
            current_code = best_code
            current_metrics = trainer_metrics
            current_stats = game_stats

            logger.info(
                "Iteration %d complete. Win rate: %.1f%%",
                iteration,
                game_stats.get("win_rate", 0.0) * 100,
            )

        return self.history

    def _generate_initial_candidates(self) -> list[str]:
        """Generate and validate initial reward function candidates."""
        valid: list[str] = []
        try:
            raw_candidates = self.generator.generate_initial()
        except RuntimeError as e:
            logger.error("Initial generation failed: %s", e)
            return valid

        for code in raw_candidates:
            if self._validate_candidate(code):
                valid.append(code)

        # If LLM didn't produce enough, add a simple fallback
        if not valid:
            fallback = (
                "import torch\n\n"
                "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
                "    # Fallback: encourage being near center (feature 4 = distance_from_center)\n"
                "    return -obs[:, 4] * 0.1\n"
            )
            if self._validate_candidate(fallback):
                valid.append(fallback)
                logger.info("Using fallback reward function.")

        return valid

    def _evolve_candidates(
        self,
        previous_code: str,
        previous_metrics: dict[str, Any],
        game_stats: dict[str, Any],
    ) -> tuple[list[str], str, str]:
        """Evolve candidates from previous iteration."""
        valid: list[str] = []
        prompt_text = ""
        response_text = ""

        for _ in range(self.config.num_candidates):
            try:
                candidates, prompt_text, response_text = self.generator.evolve(
                    previous_code, previous_metrics, game_stats
                )
            except RuntimeError as e:
                logger.error("Evolution call failed: %s", e)
                continue

            for code in candidates:
                if self._validate_candidate(code):
                    valid.append(code)

        # Keep previous code as a baseline candidate
        if self._validate_candidate(previous_code):
            valid.append(previous_code)

        return valid, prompt_text, response_text

    def _validate_candidate(self, code: str) -> bool:
        """Validate a reward function candidate through the sandbox."""
        try:
            fn = self.sandbox.compile_reward_function(code)
            self.sandbox.validate_output(fn)
            return True
        except ValueError as e:
            logger.warning("Candidate validation failed: %s", e)
            return False

    def _select_best_candidate(
        self, candidates: list[str]
    ) -> tuple[str, Callable, float]:
        """Quick-train each candidate and pick the best by win rate.

        Returns (code, compiled_fn, win_rate).
        """
        if len(candidates) == 1:
            fn = self.sandbox.compile_reward_function(candidates[0])
            return candidates[0], fn, 0.0

        best_code = candidates[0]
        best_fn = self.sandbox.compile_reward_function(candidates[0])
        best_win_rate = -1.0

        for code in candidates:
            fn = self.sandbox.compile_reward_function(code)
            metrics = self._train_with_reward(
                fn, self.config.candidate_eval_steps
            )
            stats = self._collect_stats(fn, num_episodes=50)
            win_rate = stats.get("win_rate", 0.0)

            logger.info("Candidate win_rate=%.2f", win_rate)

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_code = code
                best_fn = fn

        return best_code, best_fn, best_win_rate

    def _train_with_reward(
        self,
        reward_fn: Callable,
        total_steps: int,
    ) -> dict[str, Any]:
        """Train PPO with a shaped reward function.

        The shaped reward is ADDED to the sparse terminal reward from the env.
        """
        from knockout.env.tensor_env import TensorVecEnv
        from knockout.training.ppo import PPOTrainer

        device = self.config.device
        num_envs = self.config.num_envs

        env = TensorVecEnv(
            num_envs=num_envs,
            config=self.game_config,
            device=device,
        )

        trainer = PPOTrainer(
            config=self.game_config,
            device=device,
            num_envs=num_envs,
            backend="tensor",
        )
        self.ppo = trainer  # Store for checkpoint saving

        # Override rollout collection to inject shaped reward
        steps_per_rollout = trainer.rollout_steps * num_envs * 3
        n_rollouts = max(total_steps // steps_per_rollout, 1)

        all_metrics: list[dict[str, float]] = []
        obs, masks = env.reset()

        for rollout_idx in range(n_rollouts):
            buffer = trainer.collect_rollout_vec(env)

            # Add shaped reward to the buffer's rewards
            obs_tensor = torch.as_tensor(
                buffer.observations[: buffer.pos], dtype=torch.float32, device=device
            )
            with torch.no_grad():
                shaped = reward_fn(obs_tensor).cpu().numpy()

            buffer.rewards[: buffer.pos] += shaped

            # Recompute GAE with updated rewards
            if hasattr(buffer, "compute_gae_structured"):
                buffer.compute_gae_structured(
                    gamma=trainer.gamma, gae_lambda=trainer.gae_lambda
                )
            else:
                buffer.compute_gae(
                    gamma=trainer.gamma, gae_lambda=trainer.gae_lambda
                )

            metrics = trainer.train_step(buffer)
            all_metrics.append(metrics)

        env.close()

        # Aggregate metrics
        if all_metrics:
            return {
                k: float(np.mean([m[k] for m in all_metrics]))
                for k in all_metrics[0]
            }
        return {}

    def _collect_stats(
        self,
        reward_fn: Callable,
        num_episodes: int | None = None,
    ) -> dict[str, Any]:
        """Collect game statistics using the current policy."""
        from knockout.env.tensor_env import TensorVecEnv

        if num_episodes is None:
            num_episodes = self.config.episodes_per_eval

        env = TensorVecEnv(
            num_envs=min(self.config.num_envs, 32),
            config=self.game_config,
            device=self.config.device,
        )

        def random_policy(obs: np.ndarray) -> np.ndarray:
            """Simple random policy for stats collection."""
            B = obs.shape[0]
            angles = np.random.uniform(0, 360, size=(B, 1)).astype(np.float32)
            powers = np.random.uniform(
                0, self.game_config.MAX_LAUNCH_FORCE, size=(B, 1)
            ).astype(np.float32)
            return np.concatenate([angles, powers], axis=-1)

        stats = self.stats_collector.collect(env, random_policy, num_episodes)
        env.close()
        return stats

    @staticmethod
    def _save_iteration(iter_dir: Path, record: dict[str, Any]) -> None:
        """Save iteration artifacts to disk."""
        # Save prompt
        (iter_dir / "prompt.txt").write_text(
            record.get("prompt", ""), encoding="utf-8"
        )

        # Save LLM response
        (iter_dir / "response.txt").write_text(
            record.get("llm_reasoning", ""), encoding="utf-8"
        )

        # Save reward function code
        (iter_dir / "reward_function.py").write_text(
            record.get("reward_code", ""), encoding="utf-8"
        )

        # Save metrics and stats as JSON
        serialisable = {
            "iteration": record["iteration"],
            "metrics": record.get("metrics", {}),
            "game_stats": {
                k: v
                for k, v in record.get("game_stats", {}).items()
                if k not in ("feature_means", "feature_stds",
                             "win_terminal_feature_means",
                             "loss_terminal_feature_means")
            },
        }
        (iter_dir / "metrics.json").write_text(
            json.dumps(serialisable, indent=2), encoding="utf-8"
        )

        # Save full record (including feature arrays) separately
        full_path = iter_dir / "full_record.json"
        try:
            full_path.write_text(
                json.dumps(record, indent=2, default=str), encoding="utf-8"
            )
        except TypeError:
            logger.warning("Could not serialise full record to JSON.")
