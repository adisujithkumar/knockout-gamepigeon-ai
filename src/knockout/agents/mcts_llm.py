"""MCTS + LLM combined agent for the knockout game.

This module implements five concrete approaches for combining Monte Carlo
Tree Search with LLM reasoning in a simultaneous-move physics game.
The recommended implementation path starts with Approach 1 (simplest,
most informative) and builds toward Approach 5 (most novel).

Architecture overview
---------------------

The key insight is that the knockout game has three properties that
make standard MCTS difficult but LLMs useful:

1. **Simultaneous moves**: all 6 penguins act at once, so we cannot
   build a standard alternating-move game tree.
2. **Continuous actions**: (angle, power) is continuous, so the
   branching factor is infinite without discretisation.
3. **Fast simulation**: TensorPhysicsEngine can simulate 4096 games
   in parallel, giving us ~19K env-steps/sec on GPU.

The LLM bridges the gap between the infinite action space and the
fast simulator by proposing a small set of plausible strategies,
which the simulator then evaluates at scale.

Approach comparison
-------------------

+------+---------------------------------+----------+----------+-----------+
| #    | Name                            | LLM/turn | Latency  | Impl days |
+------+---------------------------------+----------+----------+-----------+
| 1    | Strategy Proposer + Sim Eval    |   1-2    |  3-5s    |   3-5     |
| 2    | Discretised MCTS + LLM Prior   |   1      |  5-10s   |   5-8     |
| 3    | LLM Analyst + RL Executor      |   0.2    |  <1s avg |   8-12    |
| 4    | LLM Rollout Evaluator           |   5-20   | 10-40s   |   5-8     |
| 5    | Strategic MCTS (novel)          |   1-3    |  3-8s    |   8-12    |
+------+---------------------------------+----------+----------+-----------+
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from knockout.agents.base import Agent
from knockout.agents.game_state_text import obs_to_text
from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.tensor_physics import TensorPhysicsEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Shared infrastructure used by all approaches
# ============================================================================


@dataclass
class GameSnapshot:
    """Immutable snapshot of a game state for MCTS rollouts.

    All tensors are on the same device and have batch dimension squeezed
    (i.e., shape (6, 2) not (1, 6, 2)).  The ``expand(n)`` method
    replicates the state for parallel rollouts.
    """

    positions: torch.Tensor       # (6, 2)
    velocities: torch.Tensor      # (6, 2)
    alive: torch.Tensor           # (6,) bool
    arena_half_width: float
    round_number: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def expand(self, n: int) -> "BatchedSnapshot":
        """Replicate this state n times for parallel simulation."""
        return BatchedSnapshot(
            positions=self.positions.unsqueeze(0).expand(n, -1, -1).clone(),
            velocities=self.velocities.unsqueeze(0).expand(n, -1, -1).clone(),
            alive=self.alive.unsqueeze(0).expand(n, -1).clone(),
            arena_half_width=self.arena_half_width,
            round_number=self.round_number,
            device=self.device,
        )

    @classmethod
    def from_observations(
        cls,
        obs: np.ndarray,
        agent_id: str,
        config: GameConfig = DEFAULTS,
    ) -> "GameSnapshot":
        """Reconstruct game state from a single agent's 89-dim observation.

        This is an approximate reconstruction: the observation contains
        normalised values and the ally/enemy sort order may not map back
        to exact penguin indices.  For MCTS rollouts, the approximate
        state is sufficient since we simulate forward from it.

        Args:
            obs: 89-dim observation vector for one agent.
            agent_id: e.g. "penguin_0"
            config: Game configuration.

        Returns:
            GameSnapshot with reconstructed positions/velocities/alive.
        """
        hw = config.ARENA_HALF_WIDTH
        max_speed = 200.0
        idx = int(agent_id.split("_")[1])
        is_team_a = idx < 3

        positions = torch.zeros(6, 2)
        velocities = torch.zeros(6, 2)
        alive = torch.zeros(6, dtype=torch.bool)

        def _read_block(start: int) -> tuple[float, float, float, float, bool]:
            """Read position, velocity, alive from a 14-feature block."""
            px = float(obs[start + 0] * hw)
            py = float(obs[start + 1] * hw)
            vx = float(obs[start + 2] * max_speed)
            vy = float(obs[start + 3] * max_speed)
            is_alive = bool(obs[start + 8] > 0.5)
            return px, py, vx, vy, is_alive

        # Ego penguin (block 0, indices 0-13)
        px, py, vx, vy, is_alive = _read_block(0)
        positions[idx] = torch.tensor([px, py])
        velocities[idx] = torch.tensor([vx, vy])
        alive[idx] = is_alive

        # Allies (blocks 1-2, indices 14-41)
        # Observation sorts allies by distance; we assign to the other
        # two same-team indices in order.
        if is_team_a:
            ally_indices = [i for i in range(3) if i != idx]
        else:
            ally_indices = [i for i in range(3, 6) if i != idx]

        for i, ally_idx in enumerate(ally_indices):
            block_start = 14 + i * 14
            # Use absolute position from the block (not relative)
            px, py, vx, vy, is_alive = _read_block(block_start)
            positions[ally_idx] = torch.tensor([px, py])
            velocities[ally_idx] = torch.tensor([vx, vy])
            alive[ally_idx] = is_alive

        # Enemies (blocks 3-5, indices 42-83)
        if is_team_a:
            enemy_indices = [3, 4, 5]
        else:
            enemy_indices = [0, 1, 2]

        for i, enemy_idx in enumerate(enemy_indices):
            block_start = 42 + i * 14
            px, py, vx, vy, is_alive = _read_block(block_start)
            positions[enemy_idx] = torch.tensor([px, py])
            velocities[enemy_idx] = torch.tensor([vx, vy])
            alive[enemy_idx] = is_alive

        # Global features (indices 84-88)
        # g[4] = timestep / 1000, clamped to 1.0
        timestep_norm = float(obs[84 + 4])
        round_est = int(timestep_norm * 1000)

        return cls(
            positions=positions,
            velocities=velocities,
            alive=alive,
            arena_half_width=hw,
            round_number=round_est,
            device=torch.device("cpu"),
        )


@dataclass
class BatchedSnapshot:
    """Batched game state for parallel simulation."""

    positions: torch.Tensor       # (B, 6, 2)
    velocities: torch.Tensor      # (B, 6, 2)
    alive: torch.Tensor           # (B, 6) bool
    arena_half_width: float
    round_number: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


class RolloutSimulator:
    """Fast forward simulation of game rounds using TensorPhysicsEngine.

    Given a batched game state and batched actions, simulates one round
    (apply impulses, step physics) and returns the resulting state.

    Uses a chunked stepping approach: on each round, applies impulses
    then runs ``ticks_per_round`` physics steps in small chunks
    (chunk_size steps at a time via ``step_n``).  This balances speed
    and memory: ``step_n`` unrolls the loop at trace time, so chunk
    sizes of 10-20 are optimal (keeps the compiled graph small while
    reducing Python loop overhead).

    Performance (measured on this codebase):
      CPU, B=64, 10 ticks, 1 round:  ~100ms
      CPU, B=64, 30 ticks, 1 round:  ~180ms
      GPU, B=4096, 60 ticks, 1 round: ~1ms  (estimated from training benchmarks)

    For MCTS on CPU, use ticks_per_round=30 and batch_size=64.
    For MCTS on GPU, use ticks_per_round=60 and batch_size=512-4096.
    """

    # Default physics ticks per round.  At 60 Hz:
    #   30 ticks = 0.5s game time (velocity retention ~0.70^0.5 = 0.84)
    #   60 ticks = 1.0s game time (retention ~0.70)
    #  120 ticks = 2.0s game time (retention ~0.49)
    #
    # 30 ticks is sufficient for MCTS because we care about the
    # *direction* of the outcome, not exact final positions.  Collisions
    # and boundary exits happen in the first 0.5s of sliding.
    DEFAULT_TICKS_PER_ROUND = 30

    # chunk_size for step_n calls.  Smaller = less memory, more Python
    # loop iterations.  10 is a good balance.
    CHUNK_SIZE = 10

    def __init__(
        self,
        config: GameConfig = DEFAULTS,
        device: str = "cpu",
        ticks_per_round: int | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.physics = TensorPhysicsEngine(config, device)
        self.ticks_per_round = ticks_per_round or self.DEFAULT_TICKS_PER_ROUND

    def simulate_one_round(
        self,
        snapshot: BatchedSnapshot,
        actions: torch.Tensor,
    ) -> BatchedSnapshot:
        """Simulate one round: apply impulses, step physics in chunks.

        Args:
            snapshot: Current batched game state.
            actions: (B, 6, 2) tensor of [angle_deg, power] per penguin.

        Returns:
            New BatchedSnapshot after physics resolution.
        """
        velocities = self.physics.apply_impulses(
            snapshot.velocities, actions, snapshot.alive,
        )
        positions = snapshot.positions.clone()
        alive = snapshot.alive.clone()

        remaining = self.ticks_per_round
        while remaining > 0:
            chunk = min(remaining, self.CHUNK_SIZE)
            positions, velocities, alive = self.physics.step_n(
                positions, velocities, alive,
                snapshot.arena_half_width, chunk,
            )
            remaining -= chunk

        return BatchedSnapshot(
            positions=positions,
            velocities=velocities,
            alive=alive,
            arena_half_width=snapshot.arena_half_width,
            round_number=snapshot.round_number + 1,
            device=snapshot.device,
        )

    def simulate_rounds(
        self,
        snapshot: BatchedSnapshot,
        team_a_actions: torch.Tensor,
        opponent_policy: Callable[[BatchedSnapshot], torch.Tensor] | None = None,
        n_rounds: int = 1,
    ) -> BatchedSnapshot:
        """Simulate n_rounds forward from the given state.

        For the first round, uses the provided team_a_actions.
        For subsequent rounds, uses a simple heuristic (aim at nearest
        enemy) for both teams, or the provided opponent_policy.

        Args:
            snapshot: Starting batched game state.
            team_a_actions: (B, 3, 2) actions for team A in round 1.
            opponent_policy: Callable that maps BatchedSnapshot to
                (B, 3, 2) team B actions.  If None, random actions.
            n_rounds: Number of rounds to simulate forward.

        Returns:
            Final BatchedSnapshot after n_rounds.
        """
        B = snapshot.positions.shape[0]
        state = snapshot

        for r in range(n_rounds):
            if r == 0:
                ta = team_a_actions
            else:
                # After round 1, use a simple heuristic for team A:
                # each penguin aims at the nearest alive enemy.
                ta = self._simple_policy(state, team_indices=[0, 1, 2],
                                         enemy_indices=[3, 4, 5])

            if opponent_policy is not None:
                tb = opponent_policy(state)
            else:
                # Random opponent actions
                tb = torch.stack([
                    torch.rand(B, 3, device=self.device) * 360.0,
                    torch.rand(B, 3, device=self.device) * self.config.MAX_LAUNCH_FORCE,
                ], dim=-1)

            actions = torch.cat([ta, tb], dim=1)  # (B, 6, 2)
            state = self.simulate_one_round(state, actions)

            # Check for arena shrink
            if (state.round_number > 0 and
                    state.round_number % self.config.SHRINK_INTERVAL == 0):
                new_hw = max(
                    state.arena_half_width * self.config.SHRINK_FACTOR,
                    self.config.MIN_ARENA_HALF_WIDTH,
                )
                scale = new_hw / state.arena_half_width
                state.positions, state.velocities = self.physics.rescale_penguins(
                    state.positions, state.velocities, state.alive,
                    scale, new_hw,
                )
                state.arena_half_width = new_hw

        return state

    def _simple_policy(
        self,
        state: BatchedSnapshot,
        team_indices: list[int],
        enemy_indices: list[int],
    ) -> torch.Tensor:
        """Simple heuristic: each team member aims at nearest alive enemy.

        Returns: (B, len(team_indices), 2) actions.
        """
        B = state.positions.shape[0]
        n_team = len(team_indices)
        actions = torch.zeros(B, n_team, 2, device=self.device)

        team_pos = state.positions[:, team_indices]      # (B, 3, 2)
        enemy_pos = state.positions[:, enemy_indices]    # (B, 3, 2)
        enemy_alive = state.alive[:, enemy_indices]      # (B, 3)

        for i in range(n_team):
            ego = team_pos[:, i:i+1]  # (B, 1, 2)
            delta = enemy_pos - ego   # (B, 3, 2)
            dist = torch.sqrt((delta * delta).sum(dim=-1) + 1e-8)  # (B, 3)

            # Mask dead enemies with large distance
            dist = torch.where(enemy_alive, dist, torch.full_like(dist, 1e6))
            nearest = dist.argmin(dim=-1)  # (B,)

            # Get direction to nearest enemy
            target_delta = delta[torch.arange(B), nearest]  # (B, 2)
            angle_rad = torch.atan2(target_delta[:, 1], target_delta[:, 0])
            angle_deg = (angle_rad * 180.0 / math.pi) % 360.0

            # Power proportional to distance (clamped)
            target_dist = dist[torch.arange(B), nearest]
            power = (target_dist / state.arena_half_width * self.config.MAX_LAUNCH_FORCE
                     ).clamp(100.0, self.config.MAX_LAUNCH_FORCE)

            actions[:, i, 0] = angle_deg
            actions[:, i, 1] = power

        return actions


def evaluate_outcome(state: BatchedSnapshot) -> torch.Tensor:
    """Score a batched game state from Team A's perspective.

    Returns a (B,) tensor of scores in [-1, 1]:
      +1.0 = Team A has won (all of B eliminated)
      -1.0 = Team A has lost (all of A eliminated)
       0.0 = ongoing or draw

    For non-terminal states, uses a heuristic value estimate based on:
      - alive count advantage
      - edge proximity of enemies (closer to edge = better for A)
      - edge proximity of team A (closer to edge = worse for A)

    Returns:
        (B,) float tensor of position evaluations.
    """
    a_alive = state.alive[:, :3].float().sum(dim=-1)   # (B,)
    b_alive = state.alive[:, 3:].float().sum(dim=-1)   # (B,)

    # Terminal states
    a_wins = (b_alive == 0) & (a_alive > 0)
    b_wins = (a_alive == 0) & (b_alive > 0)

    # Heuristic value for non-terminal states
    # Component 1: alive count advantage, normalised to [-1, 1]
    count_advantage = (a_alive - b_alive) / 3.0  # [-1, 1]

    # Component 2: enemy edge proximity (good for A)
    hw = state.arena_half_width
    enemy_pos = state.positions[:, 3:]  # (B, 3, 2)
    enemy_cheb = torch.max(enemy_pos[..., 0].abs(), enemy_pos[..., 1].abs())
    enemy_edge_score = (enemy_cheb / hw).clamp(0, 1)  # (B, 3)
    # Only count alive enemies
    enemy_edge_score = (enemy_edge_score * state.alive[:, 3:].float()).sum(dim=-1)
    enemy_edge_score = enemy_edge_score / b_alive.clamp(min=1)  # avg, (B,)

    # Component 3: own edge proximity (bad for A)
    team_pos = state.positions[:, :3]  # (B, 3, 2)
    team_cheb = torch.max(team_pos[..., 0].abs(), team_pos[..., 1].abs())
    team_edge_score = (team_cheb / hw).clamp(0, 1)  # (B, 3)
    team_edge_score = (team_edge_score * state.alive[:, :3].float()).sum(dim=-1)
    team_edge_score = team_edge_score / a_alive.clamp(min=1)

    # Combine: count matters most, edge proximity is a tiebreaker
    value = 0.6 * count_advantage + 0.25 * enemy_edge_score - 0.15 * team_edge_score

    # Override with terminal values
    value = torch.where(a_wins, torch.ones_like(value), value)
    value = torch.where(b_wins, -torch.ones_like(value), value)

    return value


# ============================================================================
# State-to-text formatting for LLM consumption
# ============================================================================

def snapshot_to_text(snap: GameSnapshot, config: GameConfig = DEFAULTS) -> str:
    """Convert a GameSnapshot to structured text for LLM consumption.

    Uses a zero-knowledge framing (no mention of "penguin" or "knockout")
    when zero_knowledge=True.  The text includes:
    - Absolute positions and velocities of all agents
    - Distance-to-edge for each agent
    - Alive/eliminated status
    - Arena dimensions and round number

    Returns:
        Multi-line string describing the game state.
    """
    hw = snap.arena_half_width
    lines = []
    lines.append(f"GAME STATE (round {snap.round_number}):")
    lines.append(f"Arena: square, half-width={hw:.0f} "
                 f"(agents beyond +/-{hw:.0f} are eliminated)")
    lines.append("")

    for team_name, indices in [("Team A (yours)", [0, 1, 2]),
                                ("Team B (opponent)", [3, 4, 5])]:
        lines.append(f"{team_name}:")
        for i in indices:
            pos = snap.positions[i].tolist()
            vel = snap.velocities[i].tolist()
            is_alive = bool(snap.alive[i])
            if not is_alive:
                lines.append(f"  agent_{i}: ELIMINATED")
                continue
            cheb = max(abs(pos[0]), abs(pos[1]))
            dist_edge = hw - cheb
            speed = math.sqrt(vel[0]**2 + vel[1]**2)
            lines.append(
                f"  agent_{i}: pos=({pos[0]:.1f}, {pos[1]:.1f}), "
                f"speed={speed:.1f}, "
                f"dist_to_edge={dist_edge:.1f}"
            )
        lines.append("")

    a_alive = int(snap.alive[:3].sum())
    b_alive = int(snap.alive[3:].sum())
    lines.append(f"Alive: {a_alive} vs {b_alive}")

    return "\n".join(lines)


def snapshot_to_compact_json(snap: GameSnapshot) -> str:
    """Convert a GameSnapshot to compact JSON for structured LLM output."""
    agents = []
    for i in range(6):
        pos = snap.positions[i].tolist()
        vel = snap.velocities[i].tolist()
        agents.append({
            "id": i,
            "team": "A" if i < 3 else "B",
            "alive": bool(snap.alive[i]),
            "x": round(pos[0], 1),
            "y": round(pos[1], 1),
            "vx": round(vel[0], 1),
            "vy": round(vel[1], 1),
        })
    return json.dumps({
        "round": snap.round_number,
        "arena_half_width": snap.arena_half_width,
        "agents": agents,
    })


# ============================================================================
# LLM interface (uses `claude --print` subprocess)
# ============================================================================

def call_llm(
    prompt: str,
    system: str = "",
    model: str = "claude-sonnet-4-20250514",
    timeout: int = 30,
) -> str:
    """Call the Claude CLI and return the response text.

    Uses ``claude --print`` which is already authenticated.

    Args:
        prompt: User message to send.
        system: Optional system prompt.
        model: Model identifier.
        timeout: Timeout in seconds for the subprocess.

    Returns:
        Response text from the LLM.

    Raises:
        RuntimeError: If the CLI call fails.
    """
    cmd = ["claude", "--print", "--model", model]
    if system:
        cmd.extend(["--system-prompt", system])

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}): {result.stderr[:300]}"
        )
    return result.stdout.strip()


def parse_strategy_actions(
    llm_response: str,
    n_agents: int = 3,
) -> list[tuple[float, float]] | None:
    """Parse LLM response into a list of (angle, power) tuples.

    Expected format in the response (inside a ```json block or raw):
    [
      {"agent": 0, "angle": 45.0, "power": 300.0},
      {"agent": 1, "angle": 180.0, "power": 200.0},
      {"agent": 2, "angle": 90.0, "power": 150.0}
    ]

    Falls back to regex extraction if JSON parsing fails.

    Returns:
        List of (angle, power) for each team agent, or None on failure.
    """
    import re

    # Try to find JSON array in response
    json_pattern = r'\[[\s\S]*?\]'
    matches = re.findall(json_pattern, llm_response)
    for match in reversed(matches):  # prefer last match
        try:
            data = json.loads(match)
            if isinstance(data, list) and len(data) >= n_agents:
                actions = []
                for entry in data[:n_agents]:
                    angle = float(entry.get("angle", entry.get("a", 0)))
                    power = float(entry.get("power", entry.get("p", 0)))
                    actions.append((angle, power))
                return actions
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            continue

    # Try individual JSON objects
    obj_pattern = r'\{[^{}]*"angle"\s*:\s*[\d.]+[^{}]*\}'
    obj_matches = re.findall(obj_pattern, llm_response)
    if len(obj_matches) >= n_agents:
        actions = []
        for match in obj_matches[:n_agents]:
            try:
                data = json.loads(match)
                actions.append((float(data["angle"]), float(data["power"])))
            except (json.JSONDecodeError, ValueError, TypeError):
                return None
        return actions

    return None


# ============================================================================
# APPROACH 1: LLM Strategy Proposer + Simulator Evaluator
# ============================================================================
#
# ARCHITECTURE:
#
#   Game State ──> snapshot_to_text() ──> LLM Call (~2s)
#        │                                    │
#        │                            5-10 strategies
#        │                            (natural language +
#        │                             concrete actions)
#        │                                    │
#        │                                    v
#        │         ┌──────────────────────────────────────┐
#        │         │ For each strategy:                   │
#        │         │   expand state to 512 copies          │
#        └────────>│   apply strategy's actions            │
#                  │   simulate 5-10 rounds forward        │
#                  │   evaluate_outcome() on final state   │
#                  │   compute mean score across 512 sims  │
#                  └──────────────────────────────────────┘
#                                    │
#                                    v
#                           Pick strategy with
#                           highest mean score
#                                    │
#                                    v
#                         Return that strategy's
#                         (angle, power) per agent
#
# LATENCY: 1 LLM call (2s) + simulation (10ms for 5K games) = ~2.1s/turn
#
# WHAT IS NOVEL:
#   - Using fast physics simulation to evaluate LLM-proposed natural
#     language strategies is new.  Prior work (e.g. SayCan, Inner Monologue)
#     uses LLMs for planning but relies on pre-trained low-level policies,
#     not direct simulation.
#   - The simultaneous-move structure means the opponent's actions are
#     unknown, so we marginalise over opponent responses by sampling
#     random/heuristic opponent actions across the 512 rollouts.
#
# IMPLEMENTATION COMPLEXITY: 3-5 days
#   - Day 1: Prompt engineering + action parsing
#   - Day 2: Rollout simulator integration
#   - Day 3: Agent wrapper + testing
#   - Day 4-5: Opponent modelling, prompt iteration


_STRATEGY_PROPOSER_SYSTEM = """\
You are playing a team strategy game. You control 3 agents (Team A: \
agent_0, agent_1, agent_2) on a square arena. Each round, every agent \
simultaneously chooses a direction (angle in degrees, 0=right, 90=up, \
180=left, 270=down) and a force (0-400). Agents slide on a low-friction \
surface. An agent that crosses the arena boundary is eliminated. The \
goal is to eliminate all opposing agents by pushing them off the edge \
through collisions.

IMPORTANT PHYSICS:
- Higher force = farther travel. Force 400 with mass 10 gives initial \
speed of 40 units/sec.
- Friction slows agents quickly (retain 70% velocity per second). Full \
power launch travels about 113 units.
- Collision elasticity is 0.8 (bouncy). Hitting an enemy near the edge \
can knock them off.
- Arena shrinks every 5 rounds (to 67% of current size).

You must propose EXACTLY 5 distinct strategies. For each strategy:
1. Name it briefly.
2. Explain the tactical idea in one sentence.
3. Give concrete actions for each alive Team A agent.

Respond in this JSON format:
```json
{
  "strategies": [
    {
      "name": "strategy name",
      "reasoning": "one sentence explanation",
      "actions": [
        {"agent": 0, "angle": 45.0, "power": 350.0},
        {"agent": 1, "angle": 90.0, "power": 300.0},
        {"agent": 2, "angle": 180.0, "power": 200.0}
      ]
    }
  ]
}
```
"""


@dataclass
class StrategyCandidate:
    """A single strategy proposed by the LLM with evaluation results."""

    name: str
    reasoning: str
    actions: list[tuple[float, float]]  # (angle, power) per team-A agent
    mean_score: float = 0.0
    std_score: float = 0.0
    win_rate: float = 0.0


class StrategyProposerAgent(Agent):
    """Approach 1: LLM proposes strategies, simulator evaluates them.

    Each turn:
    1. Convert game state to text.
    2. LLM proposes 5 strategies with concrete actions.
    3. For each strategy, simulate 512 games forward 5 rounds.
    4. Pick the strategy with highest mean evaluation score.
    5. Return that strategy's actions.

    Attributes:
        n_rollouts_per_strategy: Number of parallel rollouts per strategy
            for Monte Carlo evaluation.  Default: 64 (CPU) or 512 (GPU).
            64 rollouts at 30 ticks/round gives ~0.4s total sim time on CPU.
        n_rollout_rounds: Number of rounds to simulate forward.
            Default: 3 (captures most collision/elimination dynamics).
        last_strategies: The strategies from the most recent turn, for
            inspection/logging.
    """

    def __init__(
        self,
        agent_id: str,
        config: GameConfig = DEFAULTS,
        device: str = "cpu",
        model: str = "claude-sonnet-4-20250514",
        n_rollouts_per_strategy: int = 64,
        n_rollout_rounds: int = 3,
        seed: int | None = None,
    ):
        super().__init__(agent_id)
        self.config = config
        self.device = device
        self.model = model
        self.n_rollouts = n_rollouts_per_strategy
        self.n_rollout_rounds = n_rollout_rounds
        self.simulator = RolloutSimulator(config, device)
        self.last_strategies: list[StrategyCandidate] = []

        # Fallback agent for when LLM fails
        from knockout.agents.heuristic_agent import HeuristicAgent
        self._fallback = HeuristicAgent(agent_id, seed=seed, config=config)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """LLM proposes strategies, simulator evaluates, return best."""
        # Dead agent: no-op
        if np.allclose(observation, 0.0):
            return np.array([0.0, 0.0], dtype=np.float32)

        # Reconstruct game state from observation
        snap = GameSnapshot.from_observations(
            observation, self.agent_id, self.config
        )
        state_text = snapshot_to_text(snap, self.config)

        # Call LLM to propose strategies
        try:
            strategies = self._propose_strategies(state_text, snap)
        except (RuntimeError, TimeoutError) as e:
            logger.warning("LLM call failed: %s. Using fallback.", e)
            return self._fallback.get_action(observation)

        if not strategies:
            logger.warning("No valid strategies parsed. Using fallback.")
            return self._fallback.get_action(observation)

        # Evaluate each strategy via parallel simulation
        best = self._evaluate_strategies(strategies, snap)
        self.last_strategies = strategies

        # Return the best strategy's action for THIS agent
        ego_idx_in_team = int(self.agent_id.split("_")[1])
        if ego_idx_in_team >= len(best.actions):
            return self._fallback.get_action(observation)

        angle, power = best.actions[ego_idx_in_team]
        angle = float(np.clip(angle, 0, 360))
        power = float(np.clip(power, 0, self.config.MAX_LAUNCH_FORCE))
        return np.array([angle, power], dtype=np.float32)

    def _propose_strategies(
        self, state_text: str, snap: GameSnapshot,
    ) -> list[StrategyCandidate]:
        """Call LLM to propose 5 strategies."""
        prompt = f"Current game state:\n{state_text}\n\nPropose 5 strategies."
        response = call_llm(
            prompt=prompt,
            system=_STRATEGY_PROPOSER_SYSTEM,
            model=self.model,
            timeout=15,
        )

        return self._parse_strategies(response, snap)

    def _parse_strategies(
        self, response: str, snap: GameSnapshot,
    ) -> list[StrategyCandidate]:
        """Parse LLM response into StrategyCandidate list."""
        import re

        candidates = []

        # Try JSON parsing first
        json_match = re.search(r'\{[\s\S]*"strategies"[\s\S]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for s in data.get("strategies", []):
                    actions = []
                    for a in s.get("actions", []):
                        angle = float(a.get("angle", 0))
                        power = float(a.get("power", 0))
                        actions.append((
                            np.clip(angle, 0, 360),
                            np.clip(power, 0, self.config.MAX_LAUNCH_FORCE),
                        ))
                    # Pad if not enough actions
                    while len(actions) < 3:
                        actions.append((0.0, 0.0))
                    candidates.append(StrategyCandidate(
                        name=s.get("name", "unnamed"),
                        reasoning=s.get("reasoning", ""),
                        actions=actions[:3],
                    ))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return candidates

    def _evaluate_strategies(
        self,
        strategies: list[StrategyCandidate],
        snap: GameSnapshot,
    ) -> StrategyCandidate:
        """Evaluate all strategies via parallel simulation.

        For each strategy, creates n_rollouts copies of the game state,
        applies the strategy's actions for Team A, uses random/heuristic
        opponent actions, simulates forward n_rollout_rounds, and
        computes the mean outcome score.
        """
        best_strategy = strategies[0]
        best_score = -float("inf")

        for strategy in strategies:
            # Expand state for parallel rollouts
            batched = snap.expand(self.n_rollouts)
            batched = BatchedSnapshot(
                positions=batched.positions.to(self.device),
                velocities=batched.velocities.to(self.device),
                alive=batched.alive.to(self.device),
                arena_half_width=batched.arena_half_width,
                round_number=batched.round_number,
                device=torch.device(self.device),
            )

            # Build Team A actions tensor from strategy
            ta_actions = torch.zeros(
                self.n_rollouts, 3, 2, device=self.device
            )
            for i, (angle, power) in enumerate(strategy.actions[:3]):
                ta_actions[:, i, 0] = angle
                ta_actions[:, i, 1] = power

            # Simulate forward
            final_state = self.simulator.simulate_rounds(
                batched, ta_actions, n_rounds=self.n_rollout_rounds,
            )

            # Evaluate outcomes
            scores = evaluate_outcome(final_state)
            strategy.mean_score = float(scores.mean())
            strategy.std_score = float(scores.std())
            strategy.win_rate = float((scores > 0.5).float().mean())

            if strategy.mean_score > best_score:
                best_score = strategy.mean_score
                best_strategy = strategy

        return best_strategy


# ============================================================================
# APPROACH 2: Discretised MCTS with LLM Prior
# ============================================================================
#
# ARCHITECTURE:
#
#  ┌─────────────────────────────────────────────────────────┐
#  │                    MCTS Tree                            │
#  │                                                         │
#  │  Root (current state)                                   │
#  │   ├── Action A1 (angle=30, power=200)                   │
#  │   │    ├── Opponent response (sampled)                   │
#  │   │    │    ├── ... deeper search ...                    │
#  │   │    │    └──                                          │
#  │   │    └──                                               │
#  │   ├── Action A2 (angle=120, power=400)                  │
#  │   │    └── ...                                           │
#  │   └── ...                                                │
#  │                                                         │
#  │  Selection: UCT with LLM prior bias                     │
#  │  Expansion: top-K actions from LLM prior                │
#  │  Simulation: TensorPhysicsEngine (batched)              │
#  │  Backprop: mean outcome score                           │
#  └─────────────────────────────────────────────────────────┘
#
#  LLM Prior (called once at root):
#    LLM sees state ──> outputs per-agent action distribution
#    "agent_0 should aim ~45 degrees at power ~300 (high confidence)"
#    "agent_0 could also try 180 degrees at power 150 (medium)"
#    ...
#    This becomes the prior probability for PUCT selection.
#
# DISCRETISATION:
#   Angles: 16 directions (0, 22.5, 45, ..., 337.5)
#   Powers: 4 levels (100, 200, 300, 400)
#   Per-agent: 64 discrete actions
#   Per-team (factored): 3 agents x 64 actions = 192 action slots
#                         (NOT 64^3 = 262K joint actions)
#
#   We factorise the joint action by treating each agent's choice
#   independently, using the LLM prior to focus on the top-K actions
#   per agent (K=8), then forming joint actions by combining the
#   top per-agent choices.
#
# LATENCY: 1 LLM call (2s) + 200 MCTS iterations (~3s on GPU) = ~5s/turn
#
# WHAT IS NOVEL:
#   - Using an LLM as the prior policy (replacing the neural network
#     in AlphaZero) for a physics simulation game.
#   - Factored action selection for simultaneous-move MCTS with LLM
#     guidance on the factorisation.
#   - Prior work on LLM + MCTS (e.g., "Reasoning with LLMs via MCTS",
#     Hao et al. 2023) focuses on text reasoning tasks, not physical
#     simulation games.
#
# IMPLEMENTATION COMPLEXITY: 5-8 days
#   - Day 1-2: Discretisation scheme + MCTS tree data structure
#   - Day 3-4: LLM prior extraction + PUCT integration
#   - Day 5-6: Batched rollout integration with tree search
#   - Day 7-8: Testing + tuning exploration constants

N_ANGLES = 16
N_POWERS = 4
ANGLE_VALUES = [i * (360.0 / N_ANGLES) for i in range(N_ANGLES)]
POWER_VALUES = [100.0, 200.0, 300.0, 400.0]
N_DISCRETE_ACTIONS = N_ANGLES * N_POWERS  # 64

def discrete_action_to_continuous(action_idx: int) -> tuple[float, float]:
    """Convert a discrete action index to (angle, power)."""
    angle_idx = action_idx // N_POWERS
    power_idx = action_idx % N_POWERS
    return (ANGLE_VALUES[angle_idx], POWER_VALUES[power_idx])


def continuous_to_nearest_discrete(angle: float, power: float) -> int:
    """Map continuous (angle, power) to nearest discrete action index."""
    angle = angle % 360.0
    angle_idx = round(angle / (360.0 / N_ANGLES)) % N_ANGLES
    power_idx = min(range(N_POWERS),
                    key=lambda i: abs(POWER_VALUES[i] - power))
    return angle_idx * N_POWERS + power_idx


_LLM_PRIOR_SYSTEM = """\
You are analysing a team game to suggest good moves.

You control agents 0, 1, 2 (Team A) on a square arena with \
half-width {hw}. Agents choose a direction (0-360 degrees) and \
force (0-400) each round. Agents that leave the arena are eliminated. \
Goal: push opponents off through collisions.

For each alive Team A agent, rank the top 5 most promising actions.
Actions use these discrete choices:
  Angles: {angles}
  Powers: {powers}

Respond in JSON:
```json
{{
  "priors": [
    {{
      "agent": 0,
      "actions": [
        {{"angle": 45.0, "power": 300.0, "confidence": 0.35}},
        {{"angle": 22.5, "power": 400.0, "confidence": 0.25}},
        ...
      ]
    }},
    ...
  ]
}}
```
Confidences should sum to ~1.0 per agent (they are a probability distribution).
"""


@dataclass
class MCTSNode:
    """Node in the MCTS tree for discretised action search.

    Each node represents a state after Team A has chosen actions.
    Children correspond to different Team A action combinations.
    Opponent actions are sampled during rollout (simultaneous-move
    handling via self-play or random sampling).
    """

    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0          # LLM-derived prior probability
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    # Key is a combined action index encoding all 3 agents' choices

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, parent_visits: int, c_puct: float = 2.0) -> float:
        """PUCT selection score (as in AlphaZero)."""
        exploit = self.mean_value
        explore = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return exploit + explore


class DiscreteMCTSAgent(Agent):
    """Approach 2: MCTS with discretised actions and LLM prior.

    The search factorises the 3-agent joint action by:
    1. Getting per-agent top-K action priors from the LLM.
    2. Forming joint actions by combining per-agent top choices.
    3. Running PUCT-guided tree search over these joint actions.
    4. Using batched simulation for leaf evaluation.

    Attributes:
        n_simulations: Number of MCTS iterations per turn.
        top_k_per_agent: Number of top actions per agent from LLM prior.
        c_puct: Exploration constant for PUCT formula.
    """

    def __init__(
        self,
        agent_id: str,
        config: GameConfig = DEFAULTS,
        device: str = "cpu",
        model: str = "claude-sonnet-4-20250514",
        n_simulations: int = 200,
        top_k_per_agent: int = 8,
        c_puct: float = 2.0,
        seed: int | None = None,
    ):
        super().__init__(agent_id)
        self.config = config
        self.device = device
        self.model = model
        self.n_simulations = n_simulations
        self.top_k = top_k_per_agent
        self.c_puct = c_puct
        self.simulator = RolloutSimulator(config, device)

        from knockout.agents.heuristic_agent import HeuristicAgent
        self._fallback = HeuristicAgent(agent_id, seed=seed, config=config)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Run MCTS with LLM prior and return best action."""
        if np.allclose(observation, 0.0):
            return np.array([0.0, 0.0], dtype=np.float32)

        snap = GameSnapshot.from_observations(
            observation, self.agent_id, self.config
        )

        # Get LLM prior (one call per turn)
        try:
            priors = self._get_llm_prior(snap)
        except (RuntimeError, TimeoutError):
            return self._fallback.get_action(observation)

        # Run MCTS
        root = MCTSNode()
        for _ in range(self.n_simulations):
            self._mcts_iteration(root, snap, priors)

        # Select best action (highest visit count, as in AlphaZero)
        if not root.children:
            return self._fallback.get_action(observation)

        best_key = max(root.children, key=lambda k: root.children[k].visit_count)
        best_actions = self._decode_joint_action(best_key, priors)

        ego_idx = int(self.agent_id.split("_")[1])
        if ego_idx >= len(best_actions):
            return self._fallback.get_action(observation)

        angle, power = best_actions[ego_idx]
        return np.array([angle, power], dtype=np.float32)

    def _get_llm_prior(
        self, snap: GameSnapshot,
    ) -> dict[int, list[tuple[int, float]]]:
        """Get per-agent action priors from LLM.

        Returns:
            Dict mapping agent index (0, 1, 2) to list of
            (discrete_action_idx, probability) sorted by probability.
        """
        state_text = snapshot_to_text(snap, self.config)
        system = _LLM_PRIOR_SYSTEM.format(
            hw=snap.arena_half_width,
            angles=ANGLE_VALUES,
            powers=POWER_VALUES,
        )
        prompt = f"{state_text}\n\nSuggest the top 5 actions per agent."
        response = call_llm(prompt, system=system, model=self.model, timeout=15)

        # Parse response
        priors: dict[int, list[tuple[int, float]]] = {}
        try:
            import re
            json_match = re.search(r'\{[\s\S]*"priors"[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                for entry in data.get("priors", []):
                    agent_id = int(entry["agent"])
                    actions = []
                    for a in entry.get("actions", []):
                        disc_idx = continuous_to_nearest_discrete(
                            float(a["angle"]), float(a["power"])
                        )
                        conf = float(a.get("confidence", 0.2))
                        actions.append((disc_idx, conf))
                    # Normalise probabilities
                    total = sum(c for _, c in actions)
                    if total > 0:
                        actions = [(idx, c / total) for idx, c in actions]
                    priors[agent_id] = actions
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            pass

        # Default: uniform over 8 random actions for missing agents
        for i in range(3):
            if i not in priors or not priors[i]:
                rng = np.random.default_rng(42 + i)
                indices = rng.choice(N_DISCRETE_ACTIONS, size=self.top_k,
                                     replace=False).tolist()
                priors[i] = [(idx, 1.0 / self.top_k) for idx in indices]

        return priors

    def _encode_joint_action(
        self, per_agent_indices: list[int],
    ) -> int:
        """Encode 3 per-agent discrete action indices into one key."""
        # Simple: treat as base-64 number
        return (per_agent_indices[0] * N_DISCRETE_ACTIONS * N_DISCRETE_ACTIONS
                + per_agent_indices[1] * N_DISCRETE_ACTIONS
                + per_agent_indices[2])

    def _decode_joint_action(
        self, key: int, priors: dict[int, list[tuple[int, float]]],
    ) -> list[tuple[float, float]]:
        """Decode a joint action key back to (angle, power) per agent."""
        a2 = key % N_DISCRETE_ACTIONS
        key //= N_DISCRETE_ACTIONS
        a1 = key % N_DISCRETE_ACTIONS
        a0 = key // N_DISCRETE_ACTIONS
        return [
            discrete_action_to_continuous(a0),
            discrete_action_to_continuous(a1),
            discrete_action_to_continuous(a2),
        ]

    def _mcts_iteration(
        self,
        root: MCTSNode,
        snap: GameSnapshot,
        priors: dict[int, list[tuple[int, float]]],
    ) -> None:
        """One iteration of MCTS: select, expand, simulate, backprop."""
        # Selection + Expansion
        node = root
        action_key = self._select_action(node, priors)

        if action_key not in node.children:
            # Compute prior for this joint action
            per_agent = self._decode_joint_action_indices(action_key)
            joint_prior = 1.0
            for i, idx in enumerate(per_agent):
                agent_priors = dict(priors.get(i, []))
                joint_prior *= agent_priors.get(idx, 1e-3)
            node.children[action_key] = MCTSNode(prior=joint_prior)

        child = node.children[action_key]

        # Simulation: expand state, apply actions, simulate, evaluate
        actions_continuous = self._decode_joint_action(action_key, priors)
        batched = snap.expand(1)
        batched = BatchedSnapshot(
            positions=batched.positions.to(self.device),
            velocities=batched.velocities.to(self.device),
            alive=batched.alive.to(self.device),
            arena_half_width=batched.arena_half_width,
            round_number=batched.round_number,
            device=torch.device(self.device),
        )

        ta_actions = torch.zeros(1, 3, 2, device=self.device)
        for i, (angle, power) in enumerate(actions_continuous[:3]):
            ta_actions[0, i, 0] = angle
            ta_actions[0, i, 1] = power

        final = self.simulator.simulate_rounds(
            batched, ta_actions, n_rounds=3,
        )
        value = float(evaluate_outcome(final).item())

        # Backprop
        child.visit_count += 1
        child.total_value += value
        root.visit_count += 1

    def _select_action(
        self,
        node: MCTSNode,
        priors: dict[int, list[tuple[int, float]]],
    ) -> int:
        """Select action using PUCT over existing children + one new."""
        if not node.children:
            # First visit: sample from prior
            return self._sample_from_prior(priors)

        # PUCT over existing children
        best_score = -float("inf")
        best_key = -1
        for key, child in node.children.items():
            score = child.puct_score(node.visit_count, self.c_puct)
            if score > best_score:
                best_score = score
                best_key = key

        # With some probability, expand a new action from the prior
        explore_new = (len(node.children) < self.top_k ** 2 and
                       np.random.random() < 0.3)
        if explore_new:
            new_key = self._sample_from_prior(priors)
            if new_key not in node.children:
                return new_key

        return best_key

    def _sample_from_prior(
        self, priors: dict[int, list[tuple[int, float]]],
    ) -> int:
        """Sample a joint action from the factored LLM prior."""
        per_agent = []
        for i in range(3):
            agent_actions = priors.get(i, [])
            if agent_actions:
                indices = [a[0] for a in agent_actions]
                probs = np.array([a[1] for a in agent_actions])
                probs = probs / probs.sum()
                choice = np.random.choice(len(indices), p=probs)
                per_agent.append(indices[choice])
            else:
                per_agent.append(np.random.randint(0, N_DISCRETE_ACTIONS))
        return self._encode_joint_action(per_agent)

    def _decode_joint_action_indices(self, key: int) -> list[int]:
        """Decode joint action key to per-agent discrete indices."""
        a2 = key % N_DISCRETE_ACTIONS
        key //= N_DISCRETE_ACTIONS
        a1 = key % N_DISCRETE_ACTIONS
        a0 = key // N_DISCRETE_ACTIONS
        return [a0, a1, a2]


# ============================================================================
# APPROACH 3: LLM Game Analyst + RL Executor
# ============================================================================
#
# ARCHITECTURE:
#
#  ┌─────────────────────────────────────────────────────────┐
#  │                   LLM Analyst                           │
#  │  (called every K rounds, not every round)               │
#  │                                                         │
#  │  Input: game state summary                              │
#  │  Output: strategic objective embedding                  │
#  │                                                         │
#  │  Example objectives:                                    │
#  │   - "Focus fire on agent_4 (nearest edge, 12 units)"   │
#  │   - "Defensive: all agents move toward center"          │
#  │   - "Pincer: agents 0+1 converge on agent_5"           │
#  └──────────────┬──────────────────────────────────────────┘
#                 │
#                 │  strategic objective (one-hot + params)
#                 │  = 16-dim conditioning vector
#                 │
#                 v
#  ┌─────────────────────────────────────────────────────────┐
#  │              Conditioned RL Policy                      │
#  │                                                         │
#  │  Input: obs (89) + strategy_embedding (16) = 105 dim    │
#  │  Output: (angle, power) continuous action               │
#  │                                                         │
#  │  Trained via PPO with self-play.                        │
#  │  The strategy embedding tells the policy WHAT to do.    │
#  │  The policy learns HOW to execute it optimally.         │
#  └─────────────────────────────────────────────────────────┘
#
# STRATEGY EMBEDDING FORMAT (16 dims):
#   [0:5]   one-hot strategy type:
#           0 = focus_fire, 1 = defensive, 2 = pincer,
#           3 = scatter, 4 = rush
#   [5:7]   target position (normalised x, y) -- who/where to attack
#   [7:9]   target velocity (normalised vx, vy)
#   [9]     urgency (0-1, based on arena size / round number)
#   [10:12] self position hint (normalised x, y of ego)
#   [12:14] ally centroid (normalised x, y)
#   [14:16] reserved (padding)
#
# LLM CALL FREQUENCY:
#   Every K=3 rounds by default.  Between calls, the RL policy
#   executes the last-received strategy autonomously.
#   K can be adapted: call more often when the game state changes
#   dramatically (e.g., an elimination occurs).
#
# LATENCY:
#   LLM call rounds: ~2s (dominated by LLM)
#   Non-LLM rounds: <1ms (pure neural network inference)
#   Average over 20-round game with K=3: ~7 LLM calls = 14s total
#   Amortised per round: ~0.7s
#
# WHAT IS NOVEL:
#   - Prior work on LLM + RL (e.g., ELLM, SPRING) uses the LLM to
#     generate reward functions or curricula.  Here the LLM acts as
#     a real-time strategic commander that the RL policy takes orders
#     from.  This is closer to how human teams operate: a coach calls
#     plays, players execute.
#   - The strategy embedding is a fixed-size vector, so the RL policy
#     does not need to process text -- only the LLM analyst does.
#   - The RL policy can be trained with PPO against itself, where the
#     strategy comes from a random strategy sampler during training.
#     At inference time, the LLM replaces the random sampler.
#
# IMPLEMENTATION COMPLEXITY: 8-12 days
#   - Day 1-2: Strategy taxonomy + embedding format
#   - Day 3-4: Modified ActorCritic network (105-dim input)
#   - Day 5-7: PPO training with random strategy conditioning
#   - Day 8-9: LLM analyst prompt + parsing
#   - Day 10-12: Integration, testing, strategy-conditioned evaluation

N_STRATEGY_TYPES = 5
STRATEGY_NAMES = ["focus_fire", "defensive", "pincer", "scatter", "rush"]
STRATEGY_EMBEDDING_DIM = 16


@dataclass
class StrategicObjective:
    """High-level strategic objective from the LLM analyst."""

    strategy_type: int   # 0-4, index into STRATEGY_NAMES
    target_pos: tuple[float, float]  # normalised target position
    target_vel: tuple[float, float]  # normalised target velocity
    urgency: float       # 0-1
    reasoning: str       # LLM's reasoning (for logging)

    def to_embedding(
        self,
        ego_pos: tuple[float, float] = (0.0, 0.0),
        ally_centroid: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Convert to a 16-dim numpy embedding vector."""
        emb = np.zeros(STRATEGY_EMBEDDING_DIM, dtype=np.float32)
        # One-hot strategy type
        emb[self.strategy_type] = 1.0
        # Target position
        emb[5] = np.clip(self.target_pos[0], -1, 1)
        emb[6] = np.clip(self.target_pos[1], -1, 1)
        # Target velocity
        emb[7] = np.clip(self.target_vel[0], -1, 1)
        emb[8] = np.clip(self.target_vel[1], -1, 1)
        # Urgency
        emb[9] = np.clip(self.urgency, 0, 1)
        # Self position hint
        emb[10] = np.clip(ego_pos[0], -1, 1)
        emb[11] = np.clip(ego_pos[1], -1, 1)
        # Ally centroid
        emb[12] = np.clip(ally_centroid[0], -1, 1)
        emb[13] = np.clip(ally_centroid[1], -1, 1)
        return emb

    @staticmethod
    def random(rng: np.random.Generator | None = None) -> "StrategicObjective":
        """Generate a random strategic objective (for RL training)."""
        if rng is None:
            rng = np.random.default_rng()
        return StrategicObjective(
            strategy_type=int(rng.integers(0, N_STRATEGY_TYPES)),
            target_pos=(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))),
            target_vel=(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))),
            urgency=float(rng.uniform(0, 1)),
            reasoning="random",
        )


_LLM_ANALYST_SYSTEM = """\
You are a strategic analyst for a team game. Your team (agents 0, 1, 2) \
is playing against opponents (agents 3, 4, 5) on a shrinking square arena.

Analyse the current game state and choose ONE strategic objective:

1. FOCUS_FIRE: All agents target the weakest/nearest-edge opponent.
   Specify which opponent to target.

2. DEFENSIVE: All agents retreat toward center to avoid elimination.

3. PINCER: Two agents converge on a target from different angles, \
   third agent guards.

4. SCATTER: Agents spread out to control space and avoid being grouped.

5. RUSH: All agents charge the nearest opponent at full power.

Respond in JSON:
```json
{{
  "strategy": "focus_fire",
  "target_agent": 4,
  "reasoning": "agent_4 is only 15 units from the edge and moving outward",
  "urgency": 0.7
}}
```
"""


# ============================================================================
# APPROACH 4: LLM Rollout Evaluator (LLM as value function)
# ============================================================================
#
# ARCHITECTURE:
#
#  ┌─────────────────────────────────────────────────────────┐
#  │               Standard MCTS Loop                        │
#  │                                                         │
#  │  Selection ──> Expansion ──> Leaf Node                  │
#  │                                 │                       │
#  │                                 v                       │
#  │                    ┌─────────────────┐                  │
#  │                    │   LLM Evaluator │                  │
#  │                    │                 │                  │
#  │                    │  State ──> Text │                  │
#  │                    │  LLM: "Who is  │                  │
#  │                    │  likely to win  │                  │
#  │                    │  and why?"      │                  │
#  │                    │                 │                  │
#  │                    │  Output:        │                  │
#  │                    │  win_prob=0.75  │                  │
#  │                    │  reasoning=...  │                  │
#  │                    └────────┬────────┘                  │
#  │                             │                           │
#  │                    value = 2*win_prob - 1               │
#  │                             │                           │
#  │  Backprop <─────────────────┘                           │
#  └─────────────────────────────────────────────────────────┘
#
# The key difference from standard MCTS: instead of random rollouts
# or a trained value network, leaf nodes are evaluated by asking the
# LLM "who is likely to win from this position?"
#
# BATCHING LLM CALLS:
#   Naive implementation: 1 LLM call per leaf = too slow.
#   Optimization 1: Cache evaluations for similar states.
#   Optimization 2: Batch multiple leaf descriptions into one prompt.
#   Optimization 3: Use fast model (Haiku) for leaf evaluation,
#     full model (Sonnet) only for the prior at the root.
#
#   With batching: ~5 LLM calls for 50 leaf evaluations (10 per batch)
#   Latency: 5 calls x 2s = 10s per turn (acceptable for analysis,
#   too slow for real-time play).
#
# WHAT IS NOVEL:
#   - LLMs as game-state value functions is relatively unexplored.
#     Most LLM+MCTS work (e.g., "Language Agent Tree Search") uses
#     the LLM for both action proposal and value estimation in text
#     domains.  Applying this to a physics game with spatial reasoning
#     is new.
#   - The LLM can provide reasoning with its evaluation, creating
#     interpretable value estimates (unlike neural value networks).
#
# IMPLEMENTATION COMPLEXITY: 5-8 days
#   - Day 1-2: LLM evaluation prompt + response parsing
#   - Day 3-4: Batched evaluation with caching
#   - Day 5-6: MCTS integration
#   - Day 7-8: Comparison with heuristic value function

_LLM_EVALUATOR_SYSTEM = """\
You are evaluating a game position. Two teams of agents are competing \
on a square arena. Agents that cross the boundary are eliminated. \
The team that eliminates all opponents wins.

Evaluate the position from Team A's perspective.

Consider:
1. How many agents each team has alive
2. How close agents are to the edge (closer = more danger)
3. Which agents are positioned to make attacks
4. The arena size relative to agent positions

Respond in JSON:
```json
{{
  "win_probability": 0.65,
  "reasoning": "Team A has 3 agents vs 2, with agent_4 dangerously close to the edge. However, agent_1 is also near the boundary.",
  "key_factors": ["numerical advantage", "agent_4 edge proximity"]
}}
```
"""


class LLMValueEstimator:
    """Uses LLM to evaluate game positions (replacing a value network).

    Caches evaluations for similar states to reduce LLM calls.
    States are quantised to a grid for cache lookup.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        cache_resolution: float = 5.0,
    ):
        self.model = model
        self.cache_resolution = cache_resolution
        self._cache: dict[str, float] = {}
        self.call_count: int = 0
        self.cache_hits: int = 0

    def evaluate(self, snap: GameSnapshot) -> float:
        """Evaluate a game state from Team A's perspective.

        Returns a value in [-1, 1] where +1 = certain A wins.
        """
        # Check cache
        cache_key = self._state_to_cache_key(snap)
        if cache_key in self._cache:
            self.cache_hits += 1
            return self._cache[cache_key]

        # Call LLM
        state_text = snapshot_to_text(snap)
        prompt = f"{state_text}\n\nEvaluate this position for Team A."

        self.call_count += 1
        try:
            response = call_llm(
                prompt, system=_LLM_EVALUATOR_SYSTEM,
                model=self.model, timeout=10,
            )
            value = self._parse_value(response)
        except (RuntimeError, TimeoutError):
            # Fallback to heuristic
            value = self._heuristic_value(snap)

        self._cache[cache_key] = value
        return value

    def evaluate_batch(
        self, snaps: list[GameSnapshot],
    ) -> list[float]:
        """Evaluate multiple states, batching LLM calls where possible.

        Groups uncached states into batches of up to 5 and sends them
        in a single prompt (numbered positions).  This reduces LLM call
        count from N to ceil(N/5).
        """
        values = [0.0] * len(snaps)
        uncached_indices = []

        # Check cache first
        for i, snap in enumerate(snaps):
            key = self._state_to_cache_key(snap)
            if key in self._cache:
                values[i] = self._cache[key]
                self.cache_hits += 1
            else:
                uncached_indices.append(i)

        # Batch LLM calls for uncached states
        batch_size = 5
        for batch_start in range(0, len(uncached_indices), batch_size):
            batch_indices = uncached_indices[batch_start:batch_start + batch_size]
            batch_snaps = [snaps[i] for i in batch_indices]
            batch_values = self._evaluate_batch_llm(batch_snaps)
            for i, idx in enumerate(batch_indices):
                values[idx] = batch_values[i]
                key = self._state_to_cache_key(snaps[idx])
                self._cache[key] = batch_values[i]

        return values

    def _evaluate_batch_llm(
        self, snaps: list[GameSnapshot],
    ) -> list[float]:
        """Send multiple positions in one LLM call."""
        parts = []
        for i, snap in enumerate(snaps):
            parts.append(f"=== Position {i+1} ===")
            parts.append(snapshot_to_text(snap))
        prompt = "\n\n".join(parts)
        prompt += (
            "\n\nEvaluate each position for Team A. "
            "Respond with a JSON array of win probabilities:\n"
            '[{"position": 1, "win_probability": 0.65}, ...]'
        )

        self.call_count += 1
        try:
            response = call_llm(
                prompt, system=_LLM_EVALUATOR_SYSTEM,
                model=self.model, timeout=20,
            )
            return self._parse_batch_values(response, len(snaps))
        except (RuntimeError, TimeoutError):
            return [self._heuristic_value(s) for s in snaps]

    def _parse_value(self, response: str) -> float:
        """Parse win probability from LLM response."""
        import re
        match = re.search(r'"win_probability"\s*:\s*([\d.]+)', response)
        if match:
            prob = float(match.group(1))
            return 2.0 * np.clip(prob, 0, 1) - 1.0  # Map [0,1] to [-1,1]
        return 0.0

    def _parse_batch_values(
        self, response: str, n: int,
    ) -> list[float]:
        """Parse multiple win probabilities from batched response."""
        import re
        probs = re.findall(r'"win_probability"\s*:\s*([\d.]+)', response)
        values = []
        for i in range(n):
            if i < len(probs):
                prob = float(probs[i])
                values.append(2.0 * np.clip(prob, 0, 1) - 1.0)
            else:
                values.append(0.0)
        return values

    def _state_to_cache_key(self, snap: GameSnapshot) -> str:
        """Quantise state to a cache key.

        Positions are rounded to the nearest cache_resolution units.
        """
        r = self.cache_resolution
        parts = []
        for i in range(6):
            if not snap.alive[i]:
                parts.append(f"{i}:dead")
            else:
                qx = round(float(snap.positions[i][0]) / r) * r
                qy = round(float(snap.positions[i][1]) / r) * r
                parts.append(f"{i}:({qx:.0f},{qy:.0f})")
        parts.append(f"hw:{snap.arena_half_width:.0f}")
        return "|".join(parts)

    def _heuristic_value(self, snap: GameSnapshot) -> float:
        """Fallback heuristic value when LLM is unavailable."""
        value_t = evaluate_outcome(BatchedSnapshot(
            positions=snap.positions.unsqueeze(0),
            velocities=snap.velocities.unsqueeze(0),
            alive=snap.alive.unsqueeze(0),
            arena_half_width=snap.arena_half_width,
            round_number=snap.round_number,
            device=snap.device,
        ))
        return float(value_t.item())


# ============================================================================
# APPROACH 5: Strategic MCTS (Novel -- recommended for research)
# ============================================================================
#
# ARCHITECTURE:
#
#  ┌─────────────────────────────────────────────────────────────────┐
#  │                    Strategic MCTS                               │
#  │                                                                 │
#  │  Instead of searching over individual actions (huge space),     │
#  │  we search over STRATEGIES (small, meaningful space).           │
#  │                                                                 │
#  │  Level 0: LLM generates strategy set (called once per turn)     │
#  │    "pincer_attack_on_4"                                         │
#  │    "defensive_retreat"                                          │
#  │    "sacrifice_2_to_kill_3"                                      │
#  │    "split_2v1_and_1v2"                                          │
#  │    "wait_for_shrink"                                            │
#  │                                                                 │
#  │  Level 1: Strategy -> parameterised action template             │
#  │    pincer_attack_on_4:                                          │
#  │      agent_0: aim at target from angle offset -45               │
#  │      agent_1: aim at target from angle offset +45               │
#  │      agent_2: guard center                                      │
#  │                                                                 │
#  │  Level 2: Template -> concrete (angle, power) via geometry      │
#  │    Compute actual angles based on current positions.             │
#  │    Add random perturbations for MCTS exploration.               │
#  │                                                                 │
#  │  Level 3: MCTS over strategy tree                               │
#  │    Root                                                          │
#  │    ├── pincer_attack_on_4 (visit: 150, value: 0.62)             │
#  │    │    ├── (opponent responds) ──> child strategies for R2      │
#  │    │    └── ...                                                  │
#  │    ├── defensive_retreat (visit: 80, value: 0.45)               │
#  │    └── sacrifice_2_to_kill_3 (visit: 70, value: 0.38)          │
#  │                                                                 │
#  │  Branching factor: 5-10 strategies (vs 110K raw actions)        │
#  │  Depth: 2-3 strategy steps (each = 1-3 game rounds)            │
#  │  Rollout: GPU simulation of 512+ games per strategy             │
#  └─────────────────────────────────────────────────────────────────┘
#
# KEY INSIGHT: In this game, most of the 110K joint actions are
# "meaningless" (e.g., all agents shooting in the same direction at
# the wall).  By having the LLM generate semantically meaningful
# strategies, we search a MUCH smaller space where every option is
# plausible.  The simulator then tells us which plausible option is
# actually best.
#
# STRATEGY TEMPLATES:
#   Each strategy type has a parameterised template that converts
#   the high-level description into concrete actions given the
#   current game state.  The LLM does NOT need to output exact
#   numbers -- only the strategy type and its parameters (target,
#   formation, etc.).
#
# OPPONENT MODELLING:
#   At each strategy node, we model the opponent's response by:
#   1. Running 512 simulations with random opponent actions (baseline)
#   2. Running 512 simulations with heuristic opponent (pessimistic)
#   3. Weighting the results 50/50 (balanced estimate)
#
#   Advanced: ask the LLM "what would the opponent do in response?"
#   and simulate that as well.
#
# LATENCY:
#   1 LLM call for strategy generation: ~2s
#   1 LLM call for opponent modelling (optional): ~2s
#   Simulation of 5 strategies x 1024 rollouts: ~50ms on GPU
#   Total: ~2-4s per turn
#
# WHAT IS NOVEL:
#   - Searching over semantic strategy space instead of action space
#     is, to our knowledge, new.  This is NOT hierarchical RL (no
#     learned option/subgoal abstraction).  It is NOT standard macro
#     actions (which are fixed action sequences).  The strategies are
#     generated dynamically by the LLM based on the current state.
#   - The LLM provides both the branching factor (strategy set) and
#     the tree structure (what strategies to consider at each depth).
#   - The simulator provides ground-truth evaluation, avoiding the
#     LLM's well-known weakness at precise spatial/numerical reasoning.
#
# COMPARISON TO EXISTING WORK:
#   - AlphaZero: searches actions with learned prior.
#     Strategic MCTS: searches strategies with LLM prior.
#   - SayCan (Ahn et al.): LLM proposes tasks, learned policies execute.
#     Strategic MCTS: LLM proposes strategies, physics simulator evaluates.
#   - LATS (Zhou et al.): LLM generates thoughts, tree search explores.
#     Strategic MCTS: LLM generates game strategies, simulation evaluates.
#
# IMPLEMENTATION COMPLEXITY: 8-12 days
#   - Day 1-2: Strategy template system + parameterised action generation
#   - Day 3-4: Strategic MCTS tree search
#   - Day 5-6: LLM strategy generation + opponent modelling
#   - Day 7-8: GPU rollout integration
#   - Day 9-10: Multi-depth search
#   - Day 11-12: Testing, analysis, comparison with approaches 1-2


@dataclass
class Strategy:
    """A named strategy with a parameterised action template.

    The template function takes the current game state and returns
    concrete (angle, power) actions for each Team A agent.
    """

    name: str
    description: str
    target_agent: int | None = None   # which enemy to focus on, if any
    params: dict[str, Any] = field(default_factory=dict)


def strategy_to_actions(
    strategy: Strategy,
    snap: GameSnapshot,
    config: GameConfig = DEFAULTS,
    noise_std: float = 0.0,
) -> list[tuple[float, float]]:
    """Convert a strategy + game state into concrete actions.

    Each strategy type has a template that computes actions
    geometrically from the current positions.

    Args:
        strategy: The strategy to convert.
        snap: Current game state.
        config: Game configuration.
        noise_std: Standard deviation of Gaussian noise added to
            angles (degrees) for exploration during MCTS rollouts.

    Returns:
        List of (angle, power) for team A agents [0, 1, 2].
    """
    hw = snap.arena_half_width
    actions: list[tuple[float, float]] = []
    rng = np.random.default_rng()

    name = strategy.name.lower()

    if "focus" in name or "attack" in name:
        # FOCUS FIRE: all alive agents aim at the target enemy
        target_idx = strategy.target_agent
        if target_idx is None:
            # Default: aim at enemy closest to edge
            target_idx = _find_weakest_enemy(snap)

        target_pos = snap.positions[target_idx].numpy()

        for i in range(3):
            if not snap.alive[i]:
                actions.append((0.0, 0.0))
                continue
            ego_pos = snap.positions[i].numpy()
            angle = _angle_between(ego_pos, target_pos)
            dist = np.linalg.norm(target_pos - ego_pos)
            power = np.clip(dist / hw * config.MAX_LAUNCH_FORCE * 1.2,
                            150.0, config.MAX_LAUNCH_FORCE)
            if noise_std > 0:
                angle += rng.normal(0, noise_std)
            actions.append((float(angle % 360), float(power)))

    elif "pincer" in name:
        # PINCER: two agents converge on target from opposite sides,
        # third agent guards center.
        target_idx = strategy.target_agent
        if target_idx is None:
            target_idx = _find_weakest_enemy(snap)

        target_pos = snap.positions[target_idx].numpy()

        alive_team = [i for i in range(3) if snap.alive[i]]
        if len(alive_team) >= 2:
            # First two agents do the pincer
            for j, i in enumerate(alive_team[:2]):
                ego_pos = snap.positions[i].numpy()
                base_angle = _angle_between(ego_pos, target_pos)
                offset = -30.0 if j == 0 else 30.0  # converging angles
                angle = base_angle + offset
                dist = np.linalg.norm(target_pos - ego_pos)
                power = np.clip(dist / hw * config.MAX_LAUNCH_FORCE * 1.1,
                                200.0, config.MAX_LAUNCH_FORCE)
                if noise_std > 0:
                    angle += rng.normal(0, noise_std)
                actions.append((float(angle % 360), float(power)))

            # Third agent (if alive) guards center
            if len(alive_team) >= 3:
                i = alive_team[2]
                ego_pos = snap.positions[i].numpy()
                center = np.array([0.0, 0.0])
                angle = _angle_between(ego_pos, center)
                power = 150.0
                actions.append((float(angle % 360), float(power)))

        # Pad for dead agents
        while len(actions) < 3:
            actions.append((0.0, 0.0))

    elif "defensive" in name or "retreat" in name:
        # DEFENSIVE: all agents move toward center
        center = np.array([0.0, 0.0])
        for i in range(3):
            if not snap.alive[i]:
                actions.append((0.0, 0.0))
                continue
            ego_pos = snap.positions[i].numpy()
            dist_from_center = np.linalg.norm(ego_pos)
            if dist_from_center < 15.0:
                # Already near center, low power
                angle = rng.uniform(0, 360)
                power = 100.0
            else:
                angle = _angle_between(ego_pos, center)
                power = np.clip(dist_from_center / hw * config.MAX_LAUNCH_FORCE * 0.6,
                                100.0, 300.0)
            if noise_std > 0:
                angle += rng.normal(0, noise_std)
            actions.append((float(angle % 360), float(power)))

    elif "scatter" in name:
        # SCATTER: agents move away from each other
        alive_team = [i for i in range(3) if snap.alive[i]]
        if len(alive_team) >= 2:
            centroid = snap.positions[alive_team].mean(dim=0).numpy()
            for i in range(3):
                if not snap.alive[i]:
                    actions.append((0.0, 0.0))
                    continue
                ego_pos = snap.positions[i].numpy()
                # Move away from centroid, but stay on arena
                away_angle = _angle_between(centroid, ego_pos)
                # Bias toward center if near edge
                dist_to_edge = hw - max(abs(ego_pos[0]), abs(ego_pos[1]))
                if dist_to_edge < 20:
                    center_angle = _angle_between(ego_pos, np.array([0.0, 0.0]))
                    away_angle = 0.5 * away_angle + 0.5 * center_angle
                power = 200.0
                if noise_std > 0:
                    away_angle += rng.normal(0, noise_std)
                actions.append((float(away_angle % 360), float(power)))
        while len(actions) < 3:
            actions.append((0.0, 0.0))

    elif "rush" in name or "charge" in name:
        # RUSH: all agents charge nearest enemy at full power
        for i in range(3):
            if not snap.alive[i]:
                actions.append((0.0, 0.0))
                continue
            ego_pos = snap.positions[i].numpy()
            nearest_enemy = _find_nearest_enemy(snap, i)
            target_pos = snap.positions[nearest_enemy].numpy()
            angle = _angle_between(ego_pos, target_pos)
            power = config.MAX_LAUNCH_FORCE
            if noise_std > 0:
                angle += rng.normal(0, noise_std)
            actions.append((float(angle % 360), float(power)))

    elif "sacrifice" in name:
        # SACRIFICE: one agent charges at full power toward a target,
        # other two play safe
        target_idx = strategy.target_agent
        if target_idx is None:
            target_idx = _find_weakest_enemy(snap)

        alive_team = [i for i in range(3) if snap.alive[i]]
        if alive_team:
            # Sacrifice: the agent closest to the target
            target_pos = snap.positions[target_idx].numpy()
            dists = [(np.linalg.norm(snap.positions[i].numpy() - target_pos), i)
                     for i in alive_team]
            dists.sort()
            sacrifice_agent = dists[0][1]

            for i in range(3):
                if not snap.alive[i]:
                    actions.append((0.0, 0.0))
                elif i == sacrifice_agent:
                    ego_pos = snap.positions[i].numpy()
                    angle = _angle_between(ego_pos, target_pos)
                    actions.append((float(angle % 360), config.MAX_LAUNCH_FORCE))
                else:
                    # Play safe: move toward center
                    ego_pos = snap.positions[i].numpy()
                    center = np.array([0.0, 0.0])
                    angle = _angle_between(ego_pos, center)
                    actions.append((float(angle % 360), 150.0))
        while len(actions) < 3:
            actions.append((0.0, 0.0))

    else:
        # Unknown strategy: fall back to rush
        return strategy_to_actions(
            Strategy(name="rush", description="fallback"),
            snap, config, noise_std,
        )

    return actions[:3]


def _angle_between(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """Compute angle in degrees from from_pos to to_pos."""
    delta = to_pos - from_pos
    angle_rad = math.atan2(delta[1], delta[0])
    return math.degrees(angle_rad) % 360.0


def _find_weakest_enemy(snap: GameSnapshot) -> int:
    """Find the enemy closest to the arena edge."""
    hw = snap.arena_half_width
    worst_idx = 3
    worst_dist = float("inf")
    for i in range(3, 6):
        if not snap.alive[i]:
            continue
        pos = snap.positions[i]
        dist_to_edge = hw - max(abs(float(pos[0])), abs(float(pos[1])))
        if dist_to_edge < worst_dist:
            worst_dist = dist_to_edge
            worst_idx = i
    return worst_idx


def _find_nearest_enemy(snap: GameSnapshot, ego_idx: int) -> int:
    """Find the nearest alive enemy to the given agent."""
    ego_pos = snap.positions[ego_idx]
    best_idx = 3
    best_dist = float("inf")
    for i in range(3, 6):
        if not snap.alive[i]:
            continue
        dist = float(torch.sqrt(((snap.positions[i] - ego_pos) ** 2).sum()))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


@dataclass
class StrategyNode:
    """Node in the Strategic MCTS tree.

    Each node represents a strategy choice at a particular game state.
    Children are strategy choices for the next decision point (typically
    1-3 rounds later).
    """

    strategy: Strategy
    visit_count: int = 0
    total_value: float = 0.0
    children: list["StrategyNode"] = field(default_factory=list)
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def uct_score(self, parent_visits: int, c: float = 1.4) -> float:
        if self.visit_count == 0:
            return float("inf")
        exploit = self.mean_value
        explore = c * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploit + explore


_STRATEGY_GEN_SYSTEM = """\
You are a tactical planner for a team game. You control Team A \
(agents 0, 1, 2) against Team B (agents 3, 4, 5) on a square arena.

Each round, agents choose a direction and force to launch. Agents \
slide on a low-friction surface. Crossing the boundary = eliminated. \
Arena shrinks every 5 rounds. Goal: eliminate all opponents.

Given the current game state, generate {n} distinct tactical strategies.
Each strategy should be meaningfully different (not just angle variations).

For each strategy, specify:
- name: short identifier (snake_case)
- description: one-sentence tactical explanation
- target_agent: which opponent to focus on (3, 4, or 5), or null
- strategy_type: one of [focus_fire, pincer, defensive, scatter, \
  rush, sacrifice]

Respond in JSON:
```json
{{
  "strategies": [
    {{
      "name": "pincer_on_4",
      "description": "Converge agents 0 and 1 on agent_4 from two sides",
      "target_agent": 4,
      "strategy_type": "pincer"
    }},
    ...
  ]
}}
```
"""


class StrategicMCTSAgent(Agent):
    """Approach 5: MCTS over strategy space with LLM-generated strategies.

    This is the recommended approach for research novelty.  It combines:
    - LLM creativity (generating diverse, meaningful strategies)
    - Physics simulation accuracy (evaluating strategies at scale)
    - MCTS exploration (searching the strategy tree systematically)

    The search tree has strategies as nodes, not individual actions.
    This reduces the branching factor from ~110K (joint action space)
    to 5-10 (strategy space), making deep search feasible.

    Attributes:
        n_strategies: Number of strategies to generate per turn.
        n_rollouts_per_strategy: Parallel rollouts per strategy evaluation.
        n_rollout_rounds: Rounds to simulate per rollout.
        mcts_iterations: Number of MCTS iterations (strategy-level).
        noise_std: Gaussian noise added to action angles for exploration.
    """

    def __init__(
        self,
        agent_id: str,
        config: GameConfig = DEFAULTS,
        device: str = "cpu",
        model: str = "claude-sonnet-4-20250514",
        n_strategies: int = 6,
        n_rollouts_per_strategy: int = 64,
        n_rollout_rounds: int = 3,
        mcts_iterations: int = 50,
        noise_std: float = 10.0,
        seed: int | None = None,
    ):
        super().__init__(agent_id)
        self.config = config
        self.device = device
        self.model = model
        self.n_strategies = n_strategies
        self.n_rollouts = n_rollouts_per_strategy
        self.n_rollout_rounds = n_rollout_rounds
        self.mcts_iterations = mcts_iterations
        self.noise_std = noise_std
        self.simulator = RolloutSimulator(config, device)
        self.last_tree: StrategyNode | None = None

        from knockout.agents.heuristic_agent import HeuristicAgent
        self._fallback = HeuristicAgent(agent_id, seed=seed, config=config)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Generate strategies via LLM, search via MCTS, return best."""
        if np.allclose(observation, 0.0):
            return np.array([0.0, 0.0], dtype=np.float32)

        snap = GameSnapshot.from_observations(
            observation, self.agent_id, self.config
        )

        # Generate strategies via LLM
        try:
            strategies = self._generate_strategies(snap)
        except (RuntimeError, TimeoutError):
            logger.warning("LLM strategy generation failed. Using fallback.")
            return self._fallback.get_action(observation)

        if len(strategies) < 2:
            return self._fallback.get_action(observation)

        # Build strategy tree root
        root = StrategyNode(strategy=Strategy("root", "root"))

        # Create child nodes for each strategy
        for s in strategies:
            root.children.append(StrategyNode(strategy=s))

        # Run MCTS iterations
        for _ in range(self.mcts_iterations):
            self._mcts_iteration(root, snap)

        self.last_tree = root

        # Select best strategy (highest visit count)
        if not root.children:
            return self._fallback.get_action(observation)

        best_child = max(root.children, key=lambda c: c.visit_count)
        best_actions = strategy_to_actions(
            best_child.strategy, snap, self.config,
        )

        # Return action for this specific agent
        ego_idx = int(self.agent_id.split("_")[1])
        if ego_idx >= len(best_actions):
            return self._fallback.get_action(observation)

        angle, power = best_actions[ego_idx]
        angle = float(np.clip(angle, 0, 360))
        power = float(np.clip(power, 0, self.config.MAX_LAUNCH_FORCE))

        logger.info(
            "[%s] Strategic MCTS chose '%s' (visits=%d, value=%.3f)",
            self.agent_id, best_child.strategy.name,
            best_child.visit_count, best_child.mean_value,
        )

        return np.array([angle, power], dtype=np.float32)

    def _generate_strategies(
        self, snap: GameSnapshot,
    ) -> list[Strategy]:
        """Call LLM to generate diverse strategies for the current state."""
        state_text = snapshot_to_text(snap, self.config)
        system = _STRATEGY_GEN_SYSTEM.format(n=self.n_strategies)
        prompt = f"{state_text}\n\nGenerate {self.n_strategies} strategies."

        response = call_llm(
            prompt, system=system, model=self.model, timeout=15,
        )

        return self._parse_strategies(response)

    def _parse_strategies(self, response: str) -> list[Strategy]:
        """Parse LLM response into Strategy objects."""
        import re
        strategies = []

        json_match = re.search(r'\{[\s\S]*"strategies"[\s\S]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for s in data.get("strategies", []):
                    strategies.append(Strategy(
                        name=s.get("name", s.get("strategy_type", "unknown")),
                        description=s.get("description", ""),
                        target_agent=s.get("target_agent"),
                        params={"type": s.get("strategy_type", "rush")},
                    ))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Always include some default strategies as fallback
        if len(strategies) < 3:
            defaults = [
                Strategy("focus_weakest", "Attack enemy nearest to edge",
                         _find_weakest_enemy(snap=None) if False else None),
                Strategy("defensive_retreat", "All agents move to center"),
                Strategy("rush_nearest", "Each agent charges nearest enemy"),
            ]
            strategies.extend(defaults[:3 - len(strategies)])

        return strategies

    def _mcts_iteration(
        self, root: StrategyNode, snap: GameSnapshot,
    ) -> None:
        """One iteration: select child, simulate, backprop."""
        # Selection: UCT over children
        root.visit_count += 1
        best_child = max(
            root.children,
            key=lambda c: c.uct_score(root.visit_count),
        )

        # Simulation: convert strategy to actions, simulate, evaluate
        actions = strategy_to_actions(
            best_child.strategy, snap, self.config,
            noise_std=self.noise_std,
        )

        # Expand state for parallel rollouts
        batched = snap.expand(self.n_rollouts)
        batched = BatchedSnapshot(
            positions=batched.positions.to(self.device),
            velocities=batched.velocities.to(self.device),
            alive=batched.alive.to(self.device),
            arena_half_width=batched.arena_half_width,
            round_number=batched.round_number,
            device=torch.device(self.device),
        )

        # Build team A actions tensor with per-rollout noise
        ta_actions = torch.zeros(
            self.n_rollouts, 3, 2, device=self.device
        )
        for i, (angle, power) in enumerate(actions[:3]):
            ta_actions[:, i, 0] = angle + torch.randn(
                self.n_rollouts, device=self.device,
            ) * self.noise_std
            ta_actions[:, i, 1] = power

        # Simulate forward
        final_state = self.simulator.simulate_rounds(
            batched, ta_actions, n_rounds=self.n_rollout_rounds,
        )

        # Evaluate outcomes
        scores = evaluate_outcome(final_state)
        value = float(scores.mean())

        # Backprop
        best_child.visit_count += 1
        best_child.total_value += value


# ============================================================================
# Summary of implementation priorities
# ============================================================================
#
# RECOMMENDED ORDER:
#
# 1. APPROACH 1 (Strategy Proposer + Sim Eval) — 3-5 days
#    Why first: simplest to implement, immediately testable, establishes
#    the LLM-simulator pipeline.  Even if the LLM strategies are naive,
#    the simulator evaluation is rigorous.
#    Build: StrategyProposerAgent + RolloutSimulator.
#    Test: play vs HeuristicAgent, measure win rate.
#
# 2. APPROACH 5 (Strategic MCTS) — 8-12 days
#    Why second: this is the most novel and publishable approach.  It
#    builds on Approach 1's infrastructure (LLM calling, simulation,
#    strategy parsing) but adds tree search.
#    Build: StrategicMCTSAgent + strategy templates.
#    Test: play vs Approach 1 and HeuristicAgent.
#
# 3. APPROACH 3 (LLM Analyst + RL Executor) — 8-12 days
#    Why third: requires RL training infrastructure changes (wider
#    observation space).  The most practical for fast play (amortised
#    LLM cost), but needs PPO modifications.
#    Build: conditioned ActorCritic + LLM analyst.
#    Test: train with random strategy conditioning, then plug in LLM.
#
# 4. APPROACH 2 (Discretised MCTS + LLM Prior) — 5-8 days
#    Why fourth: standard MCTS infrastructure is useful but the
#    discretisation introduces approximation.  Better suited to games
#    with discrete actions.
#
# 5. APPROACH 4 (LLM Rollout Evaluator) — 5-8 days
#    Why last: most LLM calls per turn (expensive, slow).  Interesting
#    for studying LLM spatial reasoning quality, but not practical
#    for competitive play.
#
# TOTAL: ~30-45 days for all five, but Approaches 1 and 5 together
# give 80% of the research value in ~15 days.
#
# ============================================================================
# Zero-knowledge vs game-context framing
# ============================================================================
#
# The system prompts above give the LLM knowledge of the game mechanics
# (square arena, friction, collisions) but NOT the game name ("knockout")
# or the agents ("penguins").  This is a middle ground:
#
# - FULL ZERO-KNOWLEDGE: only describe observations as numbered features.
#   The LLM cannot reason about spatial strategy.  This is what we use
#   for reward discovery but it would severely hamper strategic planning.
#
# - GAME-AWARE: describe it as a team physics game with collisions and
#   boundaries.  The LLM can reason about angles, distances, and tactics.
#   This is what the prompts above use.
#
# - FULL CONTEXT: tell the LLM this is "penguin knockout".  Risk: the
#   LLM may have training data about similar games and produce memorised
#   rather than reasoned strategies.
#
# RECOMMENDATION: use game-aware (current approach) for competitive play,
# run zero-knowledge as a control condition for research papers.
