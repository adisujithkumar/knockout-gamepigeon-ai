"""Single source of truth for the bot lineup.

Every harness script (smoke.py, mini_tournament.py, full_tournament.py),
`scripts/play.py`, and `scripts/watch.py` reads from this module. Adding
or removing a bot is a one-place edit.

Each `BotSpec` declares: name, display name, description, type, checkpoint
path, and runtime requirements (env vars). `build_team()` constructs a
3-agent team for either side from a spec.

Usage:
    from eval.lineup import LINEUP, LINEUP_BY_NAME, build_team, get_available_bots

    available = get_available_bots()  # filters out missing-checkpoint / missing-env-var bots
    team_a = build_team(LINEUP_BY_NAME["heuristic"], team=0, seed=42)
    team_b = build_team(LINEUP_BY_NAME["ppo"], team=1, seed=42)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from knockout.agents.base import Agent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.agents.random_agent import RandomAgent
from knockout.agents.rl_agent import RLAgent

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BotSpec:
    """Declares one bot in the lineup.

    Attributes:
        name: Short identifier used in CLIs (e.g. "ppo", "self_play").
        display_name: Human-friendly label for tables/leaderboards.
        description: One-paragraph description for the README lineup table.
        bot_type: One of "scripted", "ppo", "mappo", "llm". Drives `build_team`.
        checkpoint: Path to a .pt file, relative paths resolved against repo root.
            None for scripted/llm bots that don't load weights.
        requires: Runtime requirements; entries of form "env:VAR_NAME" must be set.
    """

    name: str
    display_name: str
    description: str
    bot_type: str
    checkpoint: Path | None = None
    requires: tuple[str, ...] = field(default_factory=tuple)

    def is_available(self) -> tuple[bool, str]:
        """Return (available, reason). reason is empty if available."""
        if self.checkpoint is not None and not self.checkpoint.exists():
            return False, f"checkpoint missing: {self.checkpoint}"
        for req in self.requires:
            if req.startswith("env:"):
                var = req[4:]
                if not os.environ.get(var):
                    return False, f"env var unset: {var}"
            else:
                return False, f"unknown requirement type: {req}"
        return True, ""


def _resolve(rel: str) -> Path:
    return REPO_ROOT / rel


LINEUP: tuple[BotSpec, ...] = (
    BotSpec(
        name="random",
        display_name="Random",
        description="Uniform random angle and power. Floor of the capability ladder.",
        bot_type="scripted",
    ),
    BotSpec(
        name="heuristic",
        display_name="Heuristic",
        description="Hand-crafted target-utility scoring with predictive aim and edge awareness. Strong scripted baseline.",
        bot_type="scripted",
    ),
    BotSpec(
        name="ppo",
        display_name="PPO",
        description="Single-agent PPO trained vs random+heuristic curriculum. ~3M steps.",
        bot_type="ppo",
        checkpoint=_resolve("checkpoints/ppo_overnight/best.pt"),
    ),
    BotSpec(
        name="mappo",
        display_name="MAPPO",
        description="Multi-Agent PPO with shared centralized critic, three independent policies. ~1.7M steps.",
        bot_type="mappo",
        checkpoint=_resolve("checkpoints/mappo_overnight/best.pt"),
    ),
    BotSpec(
        name="self_play",
        display_name="Self-Play (v4)",
        description="PPO with self-play vs an evolving pool of past snapshots. 5.9M steps; final-iteration in-pool ELO 1443. May exhibit policy collapse on the final checkpoint; harness re-anchors against random/heuristic.",
        bot_type="ppo",
        checkpoint=_resolve("runs/self_play_v4/final_agent.pt"),
    ),
    BotSpec(
        name="contrastive",
        display_name="Contrastive Reward",
        description="PPO with reward shaping derived from Cohen's d on win/loss trajectories. Trained vs random.",
        bot_type="ppo",
        checkpoint=_resolve("runs/contrastive/final_agent.pt"),
    ),
    BotSpec(
        name="attention_reward",
        display_name="Attention-Reward",
        description="PPO with reward shaping derived from gradient attribution on the value head. Architecture is a plain MLP; the name refers to the reward-discovery method, not the network.",
        bot_type="ppo",
        checkpoint=_resolve("runs/attention/final_agent.pt"),
    ),
    BotSpec(
        name="llm_cli_v2",
        display_name="LLM-Reward (Claude)",
        description="PPO with reward function authored by Claude (CLI), iterated on training stats. Recovered from a 0%-win catastrophe to 84.8% across three iterations.",
        bot_type="ppo",
        checkpoint=_resolve("runs/llm_cli_v2/iteration_002/agent_checkpoint.pt"),
    ),
    BotSpec(
        name="llm_anthropic",
        display_name="LLM (Anthropic)",
        description="Claude reasons over a text-serialized game state and returns angle/power per turn. Slow (network round-trip per turn) and requires ANTHROPIC_API_KEY.",
        bot_type="llm",
        requires=("env:ANTHROPIC_API_KEY",),
    ),
)

LINEUP_BY_NAME: dict[str, BotSpec] = {bot.name: bot for bot in LINEUP}


def get_available_bots() -> list[BotSpec]:
    """Return bots that meet their runtime requirements (file + env)."""
    return [bot for bot in LINEUP if bot.is_available()[0]]


def get_unavailable_bots() -> list[tuple[BotSpec, str]]:
    """Return (bot, reason) for each bot that fails its requirements."""
    out: list[tuple[BotSpec, str]] = []
    for bot in LINEUP:
        ok, reason = bot.is_available()
        if not ok:
            out.append((bot, reason))
    return out


def build_team(
    spec: BotSpec,
    team: int,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Agent]:
    """Build a 3-agent dict for one team.

    Team A occupies penguin_0..2 (team=0); Team B occupies penguin_3..5 (team=1).
    Heuristic and Random bots get distinct seeds per slot to avoid collisions.

    Args:
        spec: Which bot to build.
        team: 0 for Team A, 1 for Team B.
        seed: Base seed; per-slot offsets are added.
        device: Torch device string for RL/MAPPO bots.

    Returns:
        Dict mapping agent_id to Agent instance.
    """
    if team not in (0, 1):
        raise ValueError(f"team must be 0 or 1, got {team}")

    base_idx = team * 3
    seed_offset = 100 if team == 0 else 200
    agent_ids = [f"penguin_{base_idx + i}" for i in range(3)]

    if spec.bot_type == "scripted":
        if spec.name == "random":
            return {
                aid: RandomAgent(aid, seed=seed + seed_offset + i)
                for i, aid in enumerate(agent_ids)
            }
        if spec.name == "heuristic":
            return {
                aid: HeuristicAgent(aid, seed=seed + seed_offset + i)
                for i, aid in enumerate(agent_ids)
            }
        raise ValueError(f"unknown scripted bot: {spec.name}")

    if spec.bot_type == "ppo":
        if spec.checkpoint is None:
            raise ValueError(f"PPO bot {spec.name} has no checkpoint")
        agent = RLAgent(
            agent_id=spec.name, obs_dim=89, action_dim=2, device=device
        )
        agent.load(spec.checkpoint)
        agent.network.eval()
        return {aid: agent for aid in agent_ids}

    if spec.bot_type == "mappo":
        if spec.checkpoint is None:
            raise ValueError(f"MAPPO bot {spec.name} has no checkpoint")
        from knockout.agents.mappo_agent import MAPPOAgent, MAPPOEvalAgent

        mappo = MAPPOAgent(obs_dim=89, action_dim=2, num_agents=3, device=device)
        mappo.load(spec.checkpoint)
        mappo.eval()
        return {
            aid: MAPPOEvalAgent(aid, mappo_agent=mappo, agent_index=i)
            for i, aid in enumerate(agent_ids)
        }

    if spec.bot_type == "llm":
        from knockout.agents.llm_agent import LLMAgent

        return {
            aid: LLMAgent(aid, seed=seed + seed_offset + i)
            for i, aid in enumerate(agent_ids)
        }

    raise ValueError(f"unknown bot_type: {spec.bot_type}")


def lineup_summary() -> str:
    """Pretty-print the lineup status. Used by smoke.py and CLIs with --list."""
    lines = ["=== Bot Lineup ==="]
    for bot in LINEUP:
        ok, reason = bot.is_available()
        marker = "[available]" if ok else f"[skipped: {reason}]"
        lines.append(f"  {bot.name:20s} {marker}  {bot.display_name}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(lineup_summary())
