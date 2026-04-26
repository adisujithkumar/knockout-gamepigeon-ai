"""Tier 1 + Tier 1.5 smoke probe for every available bot in the lineup.

For each available bot we build a 3-agent team via ``build_team`` and run it
inside :class:`PenguinEnv` for a target number of env steps (100 by default).
The opposing team is ``RandomAgent`` so the env keeps producing observations
even after eliminations. If a round terminates we reset and continue until we
have collected the requested number of steps.

The collected actions (an ``(N, 2)`` array of ``[angle, power]``) are then run
through six assertions:

  - finite               numpy ``isfinite`` on every entry
  - angle_variance       angle std > 5 degrees
  - power_variance       power std > 5 N
  - angle_in_range       0 <= angle <= 360
  - power_in_range       0 <= power <= 400
  - non_degenerate_mag   mean ``|action|_2`` > 5

The LLM bot (``llm_anthropic``) is special-cased: only 5 steps are run, the
variance check is skipped, and we report whether the underlying client's
``messages.create`` was invoked at least once. LLM probes are always treated
as informational only.

Outputs:
  - A pass/fail table on stdout in fixed-width format.
  - ``eval/results/smoke_report.json`` with full per-bot metrics.

Exit code is 0 if every non-LLM bot passes, 1 otherwise (so CI can use it).

Usage:
    python -m eval.smoke
    python -m eval.smoke --bot ppo
    python -m eval.smoke --steps 200
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval.lineup import (
    LINEUP_BY_NAME,
    BotSpec,
    build_team,
    get_available_bots,
)
from knockout.agents.base import Agent
from knockout.agents.random_agent import RandomAgent
from knockout.core.config import DEFAULTS
from knockout.env.penguin_env import PenguinEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "eval" / "results"

ASSERTION_LABELS: tuple[str, ...] = (
    "finite",
    "angle_variance",
    "power_variance",
    "angle_in_range",
    "power_in_range",
    "non_degenerate_magnitude",
)

# Short header labels for the table.
TABLE_HEADERS: tuple[str, ...] = (
    "finite",
    "ang_var",
    "pow_var",
    "ang_rng",
    "pow_rng",
    "magnitude",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the smoke probe."""
    parser = argparse.ArgumentParser(
        description="Tier 1 + 1.5 smoke probe for the lineup.",
    )
    parser.add_argument(
        "--bot",
        type=str,
        default=None,
        help="Probe just this bot (default: every available bot).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of env steps to collect actions over (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed (default: 42).",
    )
    return parser.parse_args(argv)


def select_bots(arg: str | None) -> list[BotSpec]:
    """Resolve which bots to probe.

    Args:
        arg: Optional bot name; if None probe every available bot.

    Returns:
        Ordered list of :class:`BotSpec` to probe.

    Raises:
        SystemExit: If a requested bot is unknown or unavailable.
    """
    if arg is None:
        return list(get_available_bots())

    if arg not in LINEUP_BY_NAME:
        print(f"ERROR: unknown bot: {arg}", file=sys.stderr)
        sys.exit(2)
    spec = LINEUP_BY_NAME[arg]
    ok, reason = spec.is_available()
    if not ok:
        print(f"ERROR: bot not available ({arg}): {reason}", file=sys.stderr)
        sys.exit(2)
    return [spec]


def _build_random_opponent(team: int, seed: int) -> dict[str, Agent]:
    """Build a RandomAgent team for the given team slot."""
    base_idx = team * 3
    seed_offset = 100 if team == 0 else 200
    return {
        f"penguin_{base_idx + i}": RandomAgent(
            f"penguin_{base_idx + i}", seed=seed + seed_offset + i
        )
        for i in range(3)
    }


def collect_actions(
    spec: BotSpec,
    steps: int,
    seed: int,
) -> tuple[np.ndarray | None, str | None]:
    """Build the bot, drive it through the env, and collect its actions.

    The probed bot occupies Team A (penguin_0..2). Team B is RandomAgent so
    the env keeps producing observations. If a round ends before ``steps``
    actions have been collected we reset with a derived seed and continue.

    Args:
        spec: Bot to probe.
        steps: Target number of probed-bot actions to collect.
        seed: Base seed for both bot and env.

    Returns:
        Tuple of (actions array of shape (N, 2), error_traceback). On
        instantiation/runtime failure the array is ``None`` and the
        traceback string is populated.
    """
    try:
        team_a = build_team(spec, team=0, seed=seed)
        team_b = _build_random_opponent(team=1, seed=seed)
    except Exception:  # noqa: BLE001 -- we want any error here
        return None, traceback.format_exc()

    collected: list[np.ndarray] = []
    env: PenguinEnv | None = None
    reset_count = 0
    try:
        env = PenguinEnv(seed=seed, max_steps=200, config=DEFAULTS)
        obs_dict, _ = env.reset()

        while len(collected) < steps:
            if not env.agents:
                # Round terminated; restart with a derived seed.
                reset_count += 1
                env.close()
                env = PenguinEnv(
                    seed=seed + 1_000 * reset_count,
                    max_steps=200,
                    config=DEFAULTS,
                )
                obs_dict, _ = env.reset()
                continue

            actions: dict[str, np.ndarray] = {}
            for agent_id in env.agents:
                idx = int(agent_id.split("_")[1])
                obs = obs_dict[agent_id]
                if idx < 3 and agent_id in team_a:
                    action = team_a[agent_id].get_action(obs)
                    actions[agent_id] = action
                    if len(collected) < steps:
                        collected.append(np.asarray(action, dtype=np.float32))
                elif idx >= 3 and agent_id in team_b:
                    actions[agent_id] = team_b[agent_id].get_action(obs)
                else:
                    actions[agent_id] = np.array([0.0, 0.0], dtype=np.float32)

            obs_dict, _, _, _, _ = env.step(actions)
    except Exception:  # noqa: BLE001 -- runtime failure is also a smoke fail
        return None, traceback.format_exc()
    finally:
        if env is not None:
            env.close()

    if not collected:
        return None, "collected zero actions"

    return np.stack(collected[:steps], axis=0), None


def evaluate_assertions(
    actions: np.ndarray,
    skip_variance: bool = False,
) -> dict[str, bool]:
    """Run the six smoke assertions on a collected action array.

    Args:
        actions: ``(N, 2)`` array of ``[angle_degrees, power_newtons]``.
        skip_variance: If True the variance assertions report ``True``
            unconditionally (used for LLM bot with very small N).

    Returns:
        Mapping from assertion label to pass/fail boolean.
    """
    angles = actions[:, 0]
    powers = actions[:, 1]
    magnitudes = np.linalg.norm(actions, axis=1)

    results: dict[str, bool] = {}
    results["finite"] = bool(np.isfinite(actions).all())

    if skip_variance or actions.shape[0] < 2:
        results["angle_variance"] = True
        results["power_variance"] = True
    else:
        results["angle_variance"] = bool(angles.std() > 5.0)
        results["power_variance"] = bool(powers.std() > 5.0)

    results["angle_in_range"] = bool((angles >= 0).all() and (angles <= 360).all())
    results["power_in_range"] = bool((powers >= 0).all() and (powers <= 400).all())
    results["non_degenerate_magnitude"] = bool(magnitudes.mean() > 5.0)
    return results


def metrics_for(actions: np.ndarray) -> dict[str, dict[str, float]]:
    """Return per-dimension summary statistics for the report JSON."""
    angles = actions[:, 0]
    powers = actions[:, 1]
    magnitudes = np.linalg.norm(actions, axis=1)
    return {
        "angle": {
            "mean": float(angles.mean()),
            "std": float(angles.std()),
            "min": float(angles.min()),
            "max": float(angles.max()),
        },
        "power": {
            "mean": float(powers.mean()),
            "std": float(powers.std()),
            "min": float(powers.min()),
            "max": float(powers.max()),
        },
        "magnitude": {
            "mean": float(magnitudes.mean()),
            "std": float(magnitudes.std()),
            "min": float(magnitudes.min()),
            "max": float(magnitudes.max()),
        },
    }


def probe_llm_bot(
    spec: BotSpec, seed: int
) -> tuple[np.ndarray | None, str | None, int | None]:
    """Special-case probe for the LLM bot.

    Runs 5 env steps, monkey-patches the underlying client's
    ``messages.create`` to count invocations, and returns
    (actions, error_traceback, llm_call_count).
    """
    try:
        team_a = build_team(spec, team=0, seed=seed)
        team_b = _build_random_opponent(team=1, seed=seed)
    except Exception:  # noqa: BLE001
        return None, traceback.format_exc(), None

    # Monkey-patch the client to count API calls. Each agent gets its own
    # LLMAgent instance with its own client; we wrap them all.
    counter = {"n": 0}

    def make_counting_create(orig: Any) -> Any:
        def counting_create(*args: Any, **kwargs: Any) -> Any:
            counter["n"] += 1
            return orig(*args, **kwargs)

        return counting_create

    for agent in team_a.values():
        client = getattr(agent, "client", None)
        if client is None:
            continue
        try:
            messages = client.messages
            messages.create = make_counting_create(messages.create)  # type: ignore[assignment]
        except AttributeError:
            # If we cannot wrap the client we still let the probe run.
            continue

    collected: list[np.ndarray] = []
    env: PenguinEnv | None = None
    try:
        env = PenguinEnv(seed=seed, max_steps=200, config=DEFAULTS)
        obs_dict, _ = env.reset()
        target = 5

        while len(collected) < target:
            if not env.agents:
                env.close()
                env = PenguinEnv(seed=seed + 1, max_steps=200, config=DEFAULTS)
                obs_dict, _ = env.reset()
                continue

            actions: dict[str, np.ndarray] = {}
            for agent_id in env.agents:
                idx = int(agent_id.split("_")[1])
                obs = obs_dict[agent_id]
                if idx < 3 and agent_id in team_a:
                    action = team_a[agent_id].get_action(obs)
                    actions[agent_id] = action
                    if len(collected) < target:
                        collected.append(np.asarray(action, dtype=np.float32))
                elif idx >= 3 and agent_id in team_b:
                    actions[agent_id] = team_b[agent_id].get_action(obs)
                else:
                    actions[agent_id] = np.array([0.0, 0.0], dtype=np.float32)

            obs_dict, _, _, _, _ = env.step(actions)
    except Exception:  # noqa: BLE001
        return None, traceback.format_exc(), counter["n"]
    finally:
        if env is not None:
            env.close()

    if not collected:
        return None, "collected zero LLM actions", counter["n"]

    return np.stack(collected, axis=0), None, counter["n"]


def render_table(
    rows: list[tuple[str, dict[str, bool] | None, bool, str]],
) -> str:
    """Render the human-readable smoke table.

    Args:
        rows: List of ``(bot_name, assertion_results, verdict, note)``. If
            ``assertion_results`` is None the row is rendered as a single
            wide ERROR cell.

    Returns:
        Multi-line string ready for printing.
    """
    name_w = max(20, max((len(name) for name, _, _, _ in rows), default=20))
    cell_w = 9
    header = f"{'bot':<{name_w}}  " + "  ".join(
        f"{h:<{cell_w}}" for h in TABLE_HEADERS
    ) + "  VERDICT"
    lines = ["=== Tier 1 + 1.5 Smoke ===", header]

    for name, results, verdict, note in rows:
        if results is None:
            cells = "  ".join(f"{'ERR':<{cell_w}}" for _ in TABLE_HEADERS)
        else:
            cells = "  ".join(
                f"{('PASS' if results[label] else 'FAIL'):<{cell_w}}"
                for label in ASSERTION_LABELS
            )
        verdict_str = "PASS" if verdict else "FAIL"
        suffix = f"  <-- {note}" if note else ""
        lines.append(f"{name:<{name_w}}  {cells}  {verdict_str}{suffix}")

    return "\n".join(lines)


def write_report(report: dict[str, Any]) -> Path:
    """Write the JSON report and return its path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "smoke_report.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the smoke probe and write outputs. Returns a process exit code."""
    args = parse_args(argv)
    bots = select_bots(args.bot)

    rows: list[tuple[str, dict[str, bool] | None, bool, str]] = []
    report_per_bot: dict[str, Any] = {}
    overall_pass = True

    for spec in bots:
        is_llm = spec.bot_type == "llm"
        if is_llm:
            actions, err, llm_calls = probe_llm_bot(spec, seed=args.seed)
        else:
            actions, err = collect_actions(spec, steps=args.steps, seed=args.seed)
            llm_calls = None

        bot_entry: dict[str, Any] = {
            "bot": spec.name,
            "display_name": spec.display_name,
            "bot_type": spec.bot_type,
            "is_llm": is_llm,
            "steps_collected": int(actions.shape[0]) if actions is not None else 0,
        }

        if actions is None:
            bot_entry["error_traceback"] = err
            bot_entry["assertions"] = {label: False for label in ASSERTION_LABELS}
            bot_entry["verdict"] = False
            note = "INSTANTIATION_ERROR" if err and "build_team" in err else "RUNTIME_ERROR"
            rows.append((spec.name, None, False, note))
            report_per_bot[spec.name] = bot_entry
            overall_pass = False
            continue

        results = evaluate_assertions(actions, skip_variance=is_llm)
        bot_entry["assertions"] = results
        bot_entry["metrics"] = metrics_for(actions)

        # Verdict + note logic.
        if is_llm:
            # LLM probe is informational only.
            informational = (
                f"informational only (llm_calls={llm_calls})"
                if llm_calls is not None
                else "informational only"
            )
            verdict = bool(results["finite"])
            bot_entry["verdict"] = verdict
            bot_entry["llm_call_count"] = llm_calls
            note = informational
            if not verdict:
                note = f"non-finite actions ({informational})"
            rows.append((spec.name, results, verdict, note))
        else:
            verdict = all(results[label] for label in ASSERTION_LABELS)
            bot_entry["verdict"] = verdict
            note = ""
            if not verdict:
                if spec.name == "self_play":
                    note = "expected: known policy collapse"
                else:
                    note = "smoke failure"
                overall_pass = False
            rows.append((spec.name, results, verdict, note))

        report_per_bot[spec.name] = bot_entry

    table = render_table(rows)
    print(table)

    report = {
        "schema": "smoke_report.v1",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "steps": args.steps,
            "seed": args.seed,
            "bot_filter": args.bot,
        },
        "results": report_per_bot,
        "overall_pass": overall_pass,
    }
    path = write_report(report)
    print(f"\nWrote {path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
