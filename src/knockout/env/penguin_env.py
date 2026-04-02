"""PettingZoo Parallel API environment for penguin knockout game."""

import functools
from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from knockout.env.observations import ObservationBuilder
from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.physics_engine import PhysicsEngine


class PenguinEnv(ParallelEnv):  # type: ignore[misc]
    """3v3 Penguin Knockout environment with PettingZoo Parallel API.

    This environment supports true simultaneous actions from all 6 agents.
    Teams:
        - Team A: penguin_0, penguin_1, penguin_2
        - Team B: penguin_3, penguin_4, penguin_5

    Observation Space: Box(89,) - see ObservationBuilder for details
    Action Space: Box(2,) - [angle_degrees, power_newtons]
        - angle_degrees: [0, 360]
        - power_newtons: [0, 500]

    Rewards: Sparse
        - +1.0 for winning team
        - -1.0 for losing team
        - 0.0 during game
    """

    metadata = {
        "name": "penguin_knockout_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int = 1000,
        physics_steps_per_action: int = 10,
        config: GameConfig = DEFAULTS,
        settle_mode: bool = True,
    ):
        """Initialize penguin knockout environment.

        Args:
            seed: Random seed for deterministic physics (None for non-deterministic)
            max_steps: Maximum number of environment steps before truncation
            physics_steps_per_action: Number of physics steps to execute per action
                (only used when settle_mode is False)
            config: Game configuration to use
            settle_mode: When True (default), each step runs physics until all
                penguins come to rest rather than a fixed number of sub-steps.
        """
        super().__init__()

        self.seed_value = seed
        self.max_steps = max_steps
        self.physics_steps_per_action = physics_steps_per_action
        self.config = config
        self.settle_mode = settle_mode

        # Initialize physics engine and observation builder
        self.physics_engine = PhysicsEngine(seed=seed, config=config)
        self.obs_builder = ObservationBuilder(config=config)

        # PettingZoo API requirements
        self.possible_agents = list(config.POSSIBLE_AGENTS)
        self.agents = self.possible_agents.copy()

        # Episode state
        self.current_step = 0
        self.round_number = 0
        self._cumulative_rewards: dict[str, float] = dict.fromkeys(self.possible_agents, 0.0)

    @functools.cache  # noqa: B019
    def observation_space(self, agent: str) -> gym.Space[np.ndarray]:
        """Observation space for a specific agent.

        Cached to ensure the same object is returned for PettingZoo API compliance.
        Note: functools.cache on methods can lead to memory leaks, but this is required
        for PettingZoo API compliance (must return same object).
        """
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ObservationBuilder.TOTAL_DIM,),
            dtype=np.float32,
        )

    @functools.cache  # noqa: B019
    def action_space(self, agent: str) -> gym.Space[np.ndarray]:
        """Action space for a specific agent.

        Cached to ensure the same object is returned for PettingZoo API compliance.
        Note: functools.cache on methods can lead to memory leaks, but this is required
        for PettingZoo API compliance (must return same object).
        """
        return gym.spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([360.0, self.config.MAX_LAUNCH_FORCE], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def observation_spaces(self) -> dict[str, gym.Space[np.ndarray]]:
        """Observation spaces for all agents."""
        return {agent: self.observation_space(agent) for agent in self.possible_agents}

    @property
    def action_spaces(self) -> dict[str, gym.Space[np.ndarray]]:
        """Action spaces for all agents."""
        return {agent: self.action_space(agent) for agent in self.possible_agents}

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset the environment.

        Args:
            seed: Random seed for deterministic physics
            options: Additional options (unused)

        Returns:
            observations: Dict mapping agent_id -> observation array
            infos: Dict mapping agent_id -> info dict
        """
        # Update seed if provided
        if seed is not None:
            self.seed_value = seed

        # Reset physics engine
        self.physics_engine = PhysicsEngine(seed=self.seed_value, config=self.config)
        self.physics_engine.initialize_game()

        # Reset episode state
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.round_number = 0
        self._cumulative_rewards = dict.fromkeys(self.possible_agents, 0.0)

        # Build observations
        observations = self._get_observations()
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}

        return observations, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Execute one environment step with all agents acting simultaneously.

        Args:
            actions: Dict mapping agent_id -> action array [angle, power]

        Returns:
            observations: Dict mapping agent_id -> observation array
            rewards: Dict mapping agent_id -> reward (sparse: +1/-1/0)
            terminations: Dict mapping agent_id -> done (game over)
            truncations: Dict mapping agent_id -> truncated (max steps)
            infos: Dict mapping agent_id -> info dict
        """
        # Convert actions to physics engine format
        physics_actions = {}
        for agent_id in self.agents:
            if agent_id in actions:
                action = actions[agent_id]
                angle, power = float(action[0]), float(action[1])
                physics_actions[agent_id] = (angle, power)
            else:
                # Default action if not provided (no-op)
                physics_actions[agent_id] = (0.0, 0.0)

        # Apply all actions simultaneously (critical for simultaneity!)
        self.physics_engine.apply_actions(physics_actions)

        # Step physics simulation
        if self.settle_mode:
            self.physics_engine.step_until_settled()
        else:
            self.physics_engine.step(num_steps=self.physics_steps_per_action)
        self.current_step += 1
        self.round_number += 1

        # Shrink arena at regular intervals (after physics settling, before observations)
        if self.round_number > 0 and self.round_number % self.config.SHRINK_INTERVAL == 0:
            scale = self.physics_engine.ice_sheet.shrink(
                self.config.SHRINK_FACTOR,
                self.config.MIN_ARENA_HALF_WIDTH,
            )
            # Rescale all alive penguin positions/velocities so their
            # normalized coordinates within the arena are preserved.
            # Shrinking should NEVER eliminate penguins.
            self.physics_engine.rescale_penguins(scale)

        # Check game state
        game_over = self.physics_engine.is_game_over()
        truncated = self.current_step >= self.max_steps

        # Get observations
        observations = self._get_observations()

        # Calculate rewards (sparse: only at game end)
        rewards = self._calculate_rewards(game_over)

        # Update cumulative rewards
        for agent in self.agents:
            self._cumulative_rewards[agent] += rewards[agent]

        # Build termination/truncation dicts
        terminations = dict.fromkeys(self.agents, game_over)
        truncations = dict.fromkeys(self.agents, truncated)

        # Infos
        infos = self._build_infos(game_over)

        # Remove dead agents from active agents list
        if game_over or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Build observations for all agents.

        Returns:
            Dict mapping agent_id -> 89-dim observation array
        """
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self.obs_builder.build_observation(
                ego_id=agent_id,
                penguins=self.physics_engine.penguins,
                timestep=self.physics_engine.step_count,
            )
        return observations

    def _calculate_rewards(self, game_over: bool) -> dict[str, float]:
        """Calculate sparse rewards.

        Rewards are only non-zero at game end:
        - +1.0 for winning team
        - -1.0 for losing team
        - 0.0 during game or draw

        Args:
            game_over: Whether the game has ended

        Returns:
            Dict mapping agent_id -> reward
        """
        rewards = dict.fromkeys(self.agents, 0.0)

        if game_over:
            winner = self.physics_engine.get_winner()
            if winner == -1:
                # Draw - no rewards
                pass
            elif winner is not None and winner >= 0:  # Type guard for mypy
                # Assign rewards based on winner
                for agent_id in self.agents:
                    # Extract team from agent_id (e.g., "penguin_0" -> team 0)
                    agent_idx = int(agent_id.split("_")[1])
                    agent_team = 0 if agent_idx in self.config.TEAM_A_INDICES else 1

                    if agent_team == winner:
                        rewards[agent_id] = 1.0
                    else:
                        rewards[agent_id] = -1.0

        return rewards

    def _build_infos(self, game_over: bool) -> dict[str, dict[str, Any]]:
        """Build info dicts for all agents.

        Args:
            game_over: Whether the game has ended

        Returns:
            Dict mapping agent_id -> info dict
        """
        infos = {}
        for agent_id in self.agents:
            info: dict[str, Any] = {
                "step_count": self.physics_engine.step_count,
                "current_step": self.current_step,
                "current_arena_size": self.physics_engine.ice_sheet.half_width,
            }

            if game_over:
                winner = self.physics_engine.get_winner()
                info["winner"] = winner if winner is not None else -1
                info["team_a_alive"] = self.physics_engine.get_alive_count(team_id=0)
                info["team_b_alive"] = self.physics_engine.get_alive_count(team_id=1)

            infos[agent_id] = info

        return infos

    def render(self) -> np.ndarray | None:
        """Render the environment (not implemented yet).

        Returns:
            None for now (will return RGB array in Phase 4)
        """
        # Visualization will be implemented in Phase 4
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    def state(self) -> np.ndarray:
        """Return global state (optional for PettingZoo).

        Returns:
            Global state array (not implemented yet)
        """
        raise NotImplementedError("Global state not implemented")
