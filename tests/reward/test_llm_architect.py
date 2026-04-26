"""Tests for the LLM Reward Architect.

Tests sandbox safety, valid compilation, stats collector structure,
and the LLM integration with mock responses (no real API calls).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from knockout.reward.llm_architect import (
    GameStatsCollector,
    LLMArchitectConfig,
    LLMRewardGenerator,
    RewardFunctionSandbox,
    build_obs_description,
)


class TestRewardFunctionSandbox:
    """Tests for the reward function sandbox."""

    def setup_method(self) -> None:
        self.sandbox = RewardFunctionSandbox()

    # --- Dangerous code should be blocked ---

    def test_blocks_import_os(self) -> None:
        code = "import os\ndef reward(obs):\n    return obs[:, 0]"
        errors = self.sandbox.validate_ast(code)
        assert any("os" in e for e in errors)

    def test_blocks_import_subprocess(self) -> None:
        code = "import subprocess\ndef reward(obs):\n    return obs[:, 0]"
        errors = self.sandbox.validate_ast(code)
        assert any("subprocess" in e for e in errors)

    def test_blocks_exec(self) -> None:
        code = "def reward(obs):\n    exec('print(1)')\n    return obs[:, 0]"
        errors = self.sandbox.validate_ast(code)
        assert any("exec" in e for e in errors)

    def test_blocks_eval(self) -> None:
        code = "def reward(obs):\n    return eval('obs[:, 0]')"
        errors = self.sandbox.validate_ast(code)
        assert any("eval" in e for e in errors)

    def test_blocks_open(self) -> None:
        code = "def reward(obs):\n    open('/etc/passwd')\n    return obs[:, 0]"
        errors = self.sandbox.validate_ast(code)
        assert any("open" in e for e in errors)

    def test_blocks_import_sys(self) -> None:
        code = "import sys\ndef reward(obs):\n    return obs[:, 0]"
        errors = self.sandbox.validate_ast(code)
        assert any("sys" in e for e in errors)

    def test_blocks_import_socket(self) -> None:
        code = "import socket\ndef reward(obs):\n    return obs[:, 0]"
        errors = self.sandbox.validate_ast(code)
        assert any("socket" in e for e in errors)

    def test_blocks_dunder_import(self) -> None:
        code = "def reward(obs):\n    __import__('os')\n    return obs[:, 0]"
        errors = self.sandbox.validate_ast(code)
        assert any("__import__" in e for e in errors)

    def test_blocks_from_os_import(self) -> None:
        code = "from os import path\ndef reward(obs):\n    return obs[:, 0]"
        errors = self.sandbox.validate_ast(code)
        assert any("os" in e for e in errors)

    def test_blocks_import_io(self) -> None:
        code = "import io\ndef reward(obs):\n    return obs[:, 0]"
        errors = self.sandbox.validate_ast(code)
        assert any("io" in e for e in errors)

    # --- Valid code should be accepted ---

    def test_accepts_valid_reward_function(self) -> None:
        code = (
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return -obs[:, 4] * 0.1\n"
        )
        errors = self.sandbox.validate_ast(code)
        assert errors == []

    def test_accepts_math_import(self) -> None:
        code = (
            "import math\nimport torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.full((obs.shape[0],), math.pi * 0.01)\n"
        )
        errors = self.sandbox.validate_ast(code)
        assert errors == []

    def test_accepts_torch_operations(self) -> None:
        code = (
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    center_dist = obs[:, 4]\n"
            "    alive = obs[:, 8]\n"
            "    return (1.0 - center_dist) * alive * 0.1\n"
        )
        errors = self.sandbox.validate_ast(code)
        assert errors == []

    # --- Compilation tests ---

    def test_compile_valid_function(self) -> None:
        code = (
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return -obs[:, 4] * 0.1\n"
        )
        fn = self.sandbox.compile_reward_function(code)
        assert callable(fn)

    def test_compile_rejects_no_reward_function(self) -> None:
        code = (
            "import torch\n\n"
            "def not_reward(obs):\n"
            "    return obs[:, 0]\n"
        )
        with pytest.raises(ValueError, match="must define a function named 'reward'"):
            self.sandbox.compile_reward_function(code)

    def test_compile_rejects_syntax_error(self) -> None:
        code = "def reward(obs:\n    return obs"
        with pytest.raises(ValueError, match="SyntaxError"):
            self.sandbox.compile_reward_function(code)

    def test_compile_rejects_blocked_import(self) -> None:
        code = "import os\ndef reward(obs):\n    return obs[:, 0]"
        with pytest.raises(ValueError, match="safety validation"):
            self.sandbox.compile_reward_function(code)

    # --- Output validation ---

    def test_validate_output_correct_shape(self) -> None:
        code = (
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.zeros(obs.shape[0])\n"
        )
        fn = self.sandbox.compile_reward_function(code)
        assert self.sandbox.validate_output(fn) is True

    def test_validate_output_wrong_shape(self) -> None:
        code = (
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.zeros(obs.shape[0], 2)\n"
        )
        fn = self.sandbox.compile_reward_function(code)
        with pytest.raises(ValueError, match="shape"):
            self.sandbox.validate_output(fn)

    def test_reward_magnitude_clipping(self) -> None:
        code = (
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.full((obs.shape[0],), 100.0)\n"
        )
        fn = self.sandbox.compile_reward_function(code)
        result = fn(torch.randn(16, 89))
        assert result.max().item() <= RewardFunctionSandbox.MAX_REWARD_MAGNITUDE

    def test_validate_rejects_nan(self) -> None:
        code = (
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return torch.full((obs.shape[0],), float('nan'))\n"
        )
        fn = self.sandbox.compile_reward_function(code)
        with pytest.raises(ValueError, match="NaN"):
            self.sandbox.validate_output(fn)


class TestGameStatsCollector:
    """Tests for the game statistics collector."""

    def test_collect_returns_expected_keys(self) -> None:
        collector = GameStatsCollector()

        # Create a mock environment
        mock_env = MagicMock()
        mock_env.num_envs = 4

        obs = np.random.randn(4, 3, 89).astype(np.float32)
        masks = np.ones((4, 3), dtype=bool)
        mock_env.reset.return_value = (obs, masks)

        # Simulate done episodes
        call_count = [0]

        def mock_step(actions):
            call_count[0] += 1
            next_obs = np.random.randn(4, 3, 89).astype(np.float32)
            rewards = np.zeros((4, 3), dtype=np.float32)
            dones = np.zeros(4, dtype=bool)
            next_masks = np.ones((4, 3), dtype=bool)
            infos = [{} for _ in range(4)]

            # Make some episodes finish
            if call_count[0] % 5 == 0:
                for i in range(4):
                    dones[i] = True
                    infos[i] = {
                        "winner": 0 if i < 2 else 1,
                        "team_a_alive": 2,
                        "team_b_alive": 1 if i < 2 else 2,
                    }

            return next_obs, rewards, dones, next_masks, infos

        mock_env.step = mock_step

        def policy(obs):
            return np.random.randn(obs.shape[0], 2).astype(np.float32)

        stats = collector.collect(mock_env, policy, num_episodes=8)

        expected_keys = {
            "total_episodes", "win_rate", "loss_rate", "draw_rate",
            "avg_episode_length", "median_episode_length",
            "avg_team_a_alive_at_end", "avg_team_b_alive_at_end",
            "feature_means", "feature_stds",
            "win_terminal_feature_means", "loss_terminal_feature_means",
        }
        assert expected_keys.issubset(set(stats.keys()))
        assert stats["total_episodes"] >= 8
        assert 0.0 <= stats["win_rate"] <= 1.0

    def test_feature_means_length(self) -> None:
        collector = GameStatsCollector()

        mock_env = MagicMock()
        mock_env.num_envs = 2
        obs = np.random.randn(2, 3, 89).astype(np.float32)
        mock_env.reset.return_value = (obs, np.ones((2, 3), dtype=bool))

        call_count = [0]

        def mock_step(actions):
            call_count[0] += 1
            dones = np.array([call_count[0] >= 3, call_count[0] >= 3])
            infos = [
                {"winner": 0, "team_a_alive": 2, "team_b_alive": 0}
                if d else {} for d in dones
            ]
            return (
                np.random.randn(2, 3, 89).astype(np.float32),
                np.zeros((2, 3), dtype=np.float32),
                dones,
                np.ones((2, 3), dtype=bool),
                infos,
            )

        mock_env.step = mock_step

        stats = collector.collect(
            mock_env,
            lambda obs: np.random.randn(obs.shape[0], 2).astype(np.float32),
            num_episodes=2,
        )

        assert len(stats["feature_means"]) == 89
        assert len(stats["feature_stds"]) == 89


class TestLLMRewardGenerator:
    """Tests for LLM reward generation with mock API responses."""

    def test_extract_code_from_response(self) -> None:
        response = (
            "Here is my reasoning...\n\n"
            "```python\n"
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return -obs[:, 4] * 0.1\n"
            "```\n"
        )
        code = LLMRewardGenerator._extract_code(response)
        assert code is not None
        assert "def reward" in code
        assert "obs[:, 4]" in code

    def test_extract_code_takes_last_block(self) -> None:
        response = (
            "First draft:\n"
            "```python\nold_code\n```\n"
            "Final version:\n"
            "```python\n"
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return obs[:, 5]\n"
            "```\n"
        )
        code = LLMRewardGenerator._extract_code(response)
        assert code is not None
        assert "obs[:, 5]" in code
        assert "old_code" not in code

    def test_extract_code_returns_none_for_no_block(self) -> None:
        response = "No code blocks here."
        assert LLMRewardGenerator._extract_code(response) is None

    @patch.object(LLMRewardGenerator, "_call_llm")
    def test_generate_initial_with_mock(self, mock_call: MagicMock) -> None:
        mock_call.return_value = (
            "My reasoning:\n\n"
            "```python\n"
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return -obs[:, 4] * 0.1\n"
            "```\n"
        )

        generator = LLMRewardGenerator()
        candidates = generator.generate_initial()

        assert len(candidates) >= 1
        assert "def reward" in candidates[0]
        mock_call.assert_called_once()

    @patch.object(LLMRewardGenerator, "_call_llm")
    def test_evolve_with_mock(self, mock_call: MagicMock) -> None:
        mock_call.return_value = (
            "Improved version:\n\n"
            "```python\n"
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    center = obs[:, 4]\n"
            "    edge = obs[:, 5]\n"
            "    return (1.0 - center) * 0.05 + edge * 0.05\n"
            "```\n"
        )

        generator = LLMRewardGenerator()
        previous_code = (
            "import torch\n\n"
            "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
            "    return -obs[:, 4] * 0.1\n"
        )
        stats = {
            "win_rate": 0.45,
            "avg_episode_length": 20.0,
            "avg_team_a_alive_at_end": 1.5,
            "avg_team_b_alive_at_end": 1.2,
            "total_episodes": 100,
            "feature_means": [0.0] * 89,
            "win_terminal_feature_means": [0.1] * 89,
            "loss_terminal_feature_means": [-0.1] * 89,
        }

        candidates, prompt, response = generator.evolve(
            previous_code, {}, stats
        )

        assert len(candidates) >= 1
        assert "def reward" in candidates[0]
        assert "center" in candidates[0] or "edge" in candidates[0]
        mock_call.assert_called_once()

    @patch.object(LLMRewardGenerator, "_call_llm")
    def test_generate_retries_on_bad_response(self, mock_call: MagicMock) -> None:
        mock_call.side_effect = [
            "No code here.",  # First attempt: no code block
            "```python\nimport torch\n\ndef reward(obs: torch.Tensor) -> torch.Tensor:\n    return torch.zeros(obs.shape[0])\n```",
        ]

        generator = LLMRewardGenerator(max_retries=3)
        candidates = generator.generate_initial()

        assert len(candidates) == 1
        assert mock_call.call_count == 2


class TestObsDescription:
    """Tests for the observation description builder."""

    def test_build_obs_description_has_89_features(self) -> None:
        desc = build_obs_description()
        # Count "Feature N:" lines
        import re
        feature_lines = re.findall(r"Feature \d+:", desc)
        assert len(feature_lines) == 89

    def test_build_obs_description_no_game_names(self) -> None:
        desc = build_obs_description()
        desc_lower = desc.lower()
        for forbidden in ["knockout", "penguin", "ice"]:
            assert forbidden not in desc_lower, (
                f"Description should not mention '{forbidden}' (zero-knowledge)"
            )


class TestLLMArchitectConfig:
    """Tests for the config dataclass."""

    def test_defaults(self) -> None:
        config = LLMArchitectConfig()
        assert config.num_iterations == 10
        assert config.steps_per_iteration == 1_000_000
        assert config.num_candidates == 3
        assert config.temperature == 0.7

    def test_frozen(self) -> None:
        config = LLMArchitectConfig()
        with pytest.raises(AttributeError):
            config.num_iterations = 5  # type: ignore[misc]
