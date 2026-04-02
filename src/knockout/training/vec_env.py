"""Vectorized environment wrapper for parallel RL training.

Runs N PenguinEnv instances simultaneously. The learning team's (Team A)
observations and actions are batched.  The opponent team (Team B) uses a
configurable fixed-policy agent.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from knockout.agents.base import Agent
from knockout.agents.random_agent import RandomAgent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.env.penguin_env import PenguinEnv
from knockout.env.observations import ObservationBuilder


class SingleTeamVecEnv:
    """Vectorized env that exposes one team's perspective.

    Runs *num_envs* PenguinEnv instances.  Team A (penguin_0/1/2) actions
    come from the learner; Team B (penguin_3/4/5) actions come from a
    fixed opponent agent supplied via *opponent_factory*.

    All public arrays use a fixed shape so the caller never has to deal
    with variable-length agent lists:

        obs   : (num_envs, 3, 89)   float32
        masks : (num_envs, 3)        bool   -- True = agent alive
        rewards: (num_envs, 3)       float32
        dones : (num_envs,)          bool   -- True = episode ended

    Dead agents receive zero observations and zero reward; their mask
    entry is False.
    """

    # Fixed agent layout --------------------------------------------------
    TEAM_A = ["penguin_0", "penguin_1", "penguin_2"]
    TEAM_B = ["penguin_3", "penguin_4", "penguin_5"]
    OBS_DIM = ObservationBuilder.TOTAL_DIM  # 89
    ACTION_DIM = 2

    def __init__(
        self,
        num_envs: int,
        opponent_factory: Callable[[], Agent] | None = None,
        config: GameConfig = DEFAULTS,
        seeds: list[int] | None = None,
    ):
        """Create *num_envs* PenguinEnv instances.

        Args:
            num_envs: Number of parallel environments.
            opponent_factory: Callable returning a fresh Agent for Team B.
                              Defaults to ``lambda: RandomAgent("opp")``.
            config: Game configuration shared across all envs.
            seeds: Per-env seeds.  If None, uses ``[0, 1, ..., N-1]``.
        """
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        self.num_envs = num_envs
        self.config = config

        if seeds is None:
            seeds = list(range(num_envs))
        if len(seeds) != num_envs:
            raise ValueError(
                f"len(seeds)={len(seeds)} must equal num_envs={num_envs}"
            )

        if opponent_factory is None:
            opponent_factory = lambda: RandomAgent("opp")  # noqa: E731

        self.envs = [
            PenguinEnv(config=config, seed=s) for s in seeds
        ]
        self.opponents = [opponent_factory() for _ in range(num_envs)]

        # Mutable per-env state: last observations for ALL agents so the
        # opponent can always act on real data.
        self._last_obs: list[dict[str, np.ndarray]] = [{} for _ in range(num_envs)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset every environment.

        Returns:
            obs   : ndarray (num_envs, 3, 89)  float32
            masks : ndarray (num_envs, 3)       bool
        """
        obs_batch = np.zeros(
            (self.num_envs, 3, self.OBS_DIM), dtype=np.float32
        )
        mask_batch = np.zeros((self.num_envs, 3), dtype=bool)

        for i, env in enumerate(self.envs):
            obs_dict, _ = env.reset()
            self._last_obs[i] = obs_dict
            self.opponents[i].reset()
            self._fill_team_a(env, obs_dict, obs_batch[i], mask_batch[i])

        return obs_batch, mask_batch

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step all environments.

        Args:
            actions: ndarray (num_envs, 3, 2) -- Team A actions.
                     Dead/masked agent actions are ignored.

        Returns:
            obs     : ndarray (num_envs, 3, 89)  float32
            rewards : ndarray (num_envs, 3)       float32
            dones   : ndarray (num_envs,)         bool
            masks   : ndarray (num_envs, 3)       bool
            infos   : list[dict]                  per-env info dicts
        """
        if actions.shape != (self.num_envs, 3, self.ACTION_DIM):
            raise ValueError(
                f"Expected actions shape ({self.num_envs}, 3, 2), "
                f"got {actions.shape}"
            )

        obs_batch = np.zeros(
            (self.num_envs, 3, self.OBS_DIM), dtype=np.float32
        )
        reward_batch = np.zeros((self.num_envs, 3), dtype=np.float32)
        done_batch = np.zeros(self.num_envs, dtype=bool)
        mask_batch = np.zeros((self.num_envs, 3), dtype=bool)
        info_list: list[dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            # Build full action dict for this env --------------------------
            full_actions: dict[str, np.ndarray] = {}

            # Team A actions from the learner
            for j, agent_id in enumerate(self.TEAM_A):
                if agent_id in env.agents:
                    full_actions[agent_id] = actions[i, j]

            # Team B actions from the opponent (using real observations)
            for agent_id in self.TEAM_B:
                if agent_id in env.agents:
                    opp_obs = self._last_obs[i].get(
                        agent_id, np.zeros(self.OBS_DIM, dtype=np.float32)
                    )
                    full_actions[agent_id] = self.opponents[i].get_action(opp_obs)

            # Step the environment -----------------------------------------
            next_obs, rewards, terms, truncs, infos = env.step(full_actions)

            # Detect episode end
            episode_done = any(terms.values()) or any(truncs.values())
            done_batch[i] = episode_done

            # Collect per-env info
            env_info: dict[str, Any] = {}
            if episode_done:
                # Grab terminal info from any agent that has it
                for v in infos.values():
                    env_info.update(v)

            # Extract Team A rewards
            for j, agent_id in enumerate(self.TEAM_A):
                reward_batch[i, j] = rewards.get(agent_id, 0.0)

            # Auto-reset if done, otherwise keep next_obs
            if episode_done:
                reset_obs, _ = env.reset()
                self._last_obs[i] = reset_obs
                self.opponents[i].reset()
                self._fill_team_a(env, reset_obs, obs_batch[i], mask_batch[i])
                env_info["terminal_observation"] = True
            else:
                self._last_obs[i] = next_obs
                self._fill_team_a(env, next_obs, obs_batch[i], mask_batch[i])

            info_list.append(env_info)

        return obs_batch, reward_batch, done_batch, mask_batch, info_list

    def close(self) -> None:
        """Release all environment resources."""
        for env in self.envs:
            env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fill_team_a(
        self,
        env: PenguinEnv,
        obs_dict: dict[str, np.ndarray],
        obs_out: np.ndarray,
        mask_out: np.ndarray,
    ) -> None:
        """Write Team A observations into pre-allocated arrays.

        An agent is masked as alive only when it is present in the
        observation dict **and** its underlying penguin is still alive in
        the physics engine.  PettingZoo keeps eliminated penguins in
        ``env.agents`` until game-over, but the observation builder
        returns all-zeros for them; the mask tells the learner to ignore
        those slots.

        Args:
            env: The PenguinEnv that produced *obs_dict*.
            obs_dict: Full observation dict from env.
            obs_out:  (3, 89) slice to write into.
            mask_out: (3,) slice to write into.
        """
        for j, agent_id in enumerate(self.TEAM_A):
            if agent_id in obs_dict:
                obs_out[j] = obs_dict[agent_id]
                # Check actual alive status via the physics engine
                penguin = env.physics_engine.penguins.get(agent_id)
                mask_out[j] = penguin is not None and penguin.alive
            else:
                obs_out[j] = 0.0
                mask_out[j] = False
