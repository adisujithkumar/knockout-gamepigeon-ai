"""Full ELO round-robin that emits a markdown leaderboard.

Like ``eval.mini_tournament`` but with no assertions and a higher default
games-per-pair count. The output is a markdown report containing standings,
a row-vs-column win-rate matrix, and a list of bots that were skipped
because of missing checkpoints or env vars.

Usage:
    python -m eval.full_tournament [--games-per-pair 50] [--seed 42] \
                                   [--out eval/results/leaderboard.md]
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.lineup import (
    LINEUP_BY_NAME,
    build_team,
    get_available_bots,
    get_unavailable_bots,
)
from knockout.core.config import DEFAULTS
from knockout.training.elo_rating import ELOTracker
from knockout.training.evaluation import run_match

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "eval" / "results" / "leaderboard.md"


def per_game_seed(base_seed: int, bot_a_idx: int, bot_b_idx: int, game_idx: int) -> int:
    """Canonical per-game seed shared with :mod:`eval.mini_tournament`.

    Formula: ``seed * 1_000_000 + bot_a_idx * 10_000 + bot_b_idx * 100 + game_idx``.
    """
    return base_seed * 1_000_000 + bot_a_idx * 10_000 + bot_b_idx * 100 + game_idx


def _try_tqdm(iterable: Any, **kwargs: Any) -> Any:
    """Wrap iterable in tqdm if available, otherwise return iterable as-is."""
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the full tournament."""
    parser = argparse.ArgumentParser(
        description="Run a full round-robin ELO tournament and emit a markdown leaderboard.",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=50,
        help="Number of games per ordered pair (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; per-game seeds are derived deterministically (default: 42).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Markdown output path (default: {DEFAULT_OUT.relative_to(REPO_ROOT)}).",
    )
    return parser.parse_args(argv)


def run_tournament(
    bots: list[str],
    games_per_pair: int,
    base_seed: int,
) -> tuple[ELOTracker, dict[tuple[str, str], dict[str, int]], int]:
    """Run the full round-robin and return ELO + a head-to-head record matrix.

    The ``records`` mapping uses ``(bot_a, bot_b)`` keys with
    ``{"wins", "losses", "draws", "games"}`` counts representing the result
    of ``bot_a`` over ``bot_b``. Per-pair exceptions are captured and counted
    as draws so the tournament keeps running.

    Args:
        bots: Ordered list of bot names.
        games_per_pair: Number of games per ordered pair.
        base_seed: Base RNG seed for :func:`per_game_seed`.
    """
    tracker = ELOTracker(initial_rating=1000.0, k=32.0)
    for name in bots:
        tracker.register(name)

    pairs: list[tuple[int, int, str, str]] = []
    for i, name_a in enumerate(bots):
        for j, name_b in enumerate(bots):
            if i == j:
                continue
            pairs.append((i, j, name_a, name_b))

    records: dict[tuple[str, str], dict[str, int]] = {
        (a, b): {"wins": 0, "losses": 0, "draws": 0, "games": 0}
        for i, a in enumerate(bots)
        for j, b in enumerate(bots)
        if i != j
    }
    exceptions = 0

    iterator = _try_tqdm(
        list(enumerate(pairs)),
        desc="pairs",
        unit="pair",
        leave=False,
    )

    for pair_idx, (i, j, name_a, name_b) in iterator:
        prefix = f"[{pair_idx + 1}/{len(pairs)}]"
        print(f"{prefix} {name_a} vs {name_b}")
        sys.stdout.flush()

        spec_a = LINEUP_BY_NAME[name_a]
        spec_b = LINEUP_BY_NAME[name_b]

        for g in range(games_per_pair):
            game_seed = per_game_seed(base_seed, i, j, g)
            a_as_team_a = (g % 2) == 0

            try:
                if a_as_team_a:
                    team_a = build_team(spec_a, team=0, seed=game_seed)
                    team_b = build_team(spec_b, team=1, seed=game_seed)
                else:
                    team_a = build_team(spec_b, team=0, seed=game_seed)
                    team_b = build_team(spec_a, team=1, seed=game_seed)

                result = run_match(
                    team_a, team_b, config=DEFAULTS, seed=game_seed, max_steps=200
                )
            except Exception as exc:  # noqa: BLE001 -- log + draw
                exceptions += 1
                traceback.print_exc()
                print(f"  {name_a} vs {name_b} game {g} raised {exc!r} -> draw")
                result = {"winner": -1, "steps": 0, "team_a_alive": 0, "team_b_alive": 0}

            winner = result["winner"]
            if winner == -1:
                score_a = 0.5
            elif a_as_team_a:
                score_a = 1.0 if winner == 0 else 0.0
            else:
                score_a = 1.0 if winner == 1 else 0.0

            tracker.update(name_a, name_b, score_a)

            rec = records[(name_a, name_b)]
            rec["games"] += 1
            if score_a == 1.0:
                rec["wins"] += 1
            elif score_a == 0.0:
                rec["losses"] += 1
            else:
                rec["draws"] += 1

    return tracker, records, exceptions


def compute_per_bot_stats(
    bots: list[str],
    records: dict[tuple[str, str], dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate per-bot games and win percentages from the records matrix.

    Each draw counts as half a win for both sides when computing win%.
    """
    stats: dict[str, dict[str, float | int]] = {
        name: {"games": 0, "wins": 0, "losses": 0, "draws": 0, "win_pct": 0.0}
        for name in bots
    }
    for (a, b), rec in records.items():
        stats[a]["games"] += rec["games"]
        stats[a]["wins"] += rec["wins"]
        stats[a]["losses"] += rec["losses"]
        stats[a]["draws"] += rec["draws"]
    for name in bots:
        s = stats[name]
        if s["games"]:
            s["win_pct"] = (s["wins"] + 0.5 * s["draws"]) / s["games"] * 100.0
    return stats


def render_markdown(
    bots: list[str],
    tracker: ELOTracker,
    records: dict[tuple[str, str], dict[str, int]],
    games_per_pair: int,
    base_seed: int,
    elapsed_s: float,
    exceptions: int,
) -> str:
    """Build the markdown report body."""
    leaderboard = [(n, r) for n, r in tracker.get_leaderboard() if n in set(bots)]
    stats = compute_per_bot_stats(bots, records)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    lines.append("# Knockout-v2 ELO Leaderboard")
    lines.append("")
    lines.append(
        f"_Run on {timestamp}, {games_per_pair} games per pair, "
        f"alternating sides per game, per-game seeds (base seed {base_seed})._"
    )
    lines.append("")
    lines.append(
        f"_Wall time: {elapsed_s:.1f}s. Bots in tournament: {len(bots)}. "
        f"Exceptions caught: {exceptions}._"
    )
    lines.append("")

    # Standings
    lines.append("## Standings")
    lines.append("")
    lines.append("| Rank | Bot | Rating | Games | Win% |")
    lines.append("|---|---|---|---|---|")
    for rank, (name, rating) in enumerate(leaderboard, start=1):
        display = LINEUP_BY_NAME[name].display_name if name in LINEUP_BY_NAME else name
        s = stats[name]
        lines.append(
            f"| {rank} | {display} | {rating:.0f} | {s['games']} | {s['win_pct']:.0f}% |"
        )
    lines.append("")

    # Win-rate matrix
    lines.append("## Win-rate matrix (row vs column, percent)")
    lines.append("")
    headers = [LINEUP_BY_NAME[n].display_name if n in LINEUP_BY_NAME else n for n in bots]
    header_row = "|         | " + " | ".join(headers) + " |"
    sep_row = "|---|" + "|".join(["---"] * len(headers)) + "|"
    lines.append(header_row)
    lines.append(sep_row)
    for row_name in bots:
        row_label = (
            LINEUP_BY_NAME[row_name].display_name
            if row_name in LINEUP_BY_NAME
            else row_name
        )
        cells: list[str] = []
        for col_name in bots:
            if row_name == col_name:
                cells.append("-")
                continue
            rec = records.get((row_name, col_name))
            if rec is None or rec["games"] == 0:
                cells.append("n/a")
                continue
            pct = (rec["wins"] + 0.5 * rec["draws"]) / rec["games"] * 100.0
            cells.append(f"{pct:.0f}%")
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
    lines.append("")

    # Skipped bots
    lines.append("## Skipped bots")
    lines.append("")
    skipped = get_unavailable_bots()
    if not skipped:
        lines.append("- None")
    else:
        for spec, reason in skipped:
            lines.append(f"- {spec.display_name} ({spec.name}) - {reason}")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m eval.full_tournament``."""
    args = parse_args(argv)

    bots = [b.name for b in get_available_bots()]
    if len(bots) < 2:
        print("ERROR: need at least 2 available bots", file=sys.stderr)
        return 2

    print(
        f"Running full tournament: {len(bots)} bots, "
        f"{args.games_per_pair} games per ordered pair, base seed {args.seed}"
    )
    print(f"  bots: {', '.join(bots)}")

    t0 = time.perf_counter()
    tracker, records, exceptions = run_tournament(
        bots=bots,
        games_per_pair=args.games_per_pair,
        base_seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    md = render_markdown(
        bots=bots,
        tracker=tracker,
        records=records,
        games_per_pair=args.games_per_pair,
        base_seed=args.seed,
        elapsed_s=elapsed,
        exceptions=exceptions,
    )

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    # Echo a short standings snapshot to stdout.
    print()
    print("=== Final Standings ===")
    for rank, (name, rating) in enumerate(tracker.get_leaderboard(), start=1):
        if name not in set(bots):
            continue
        display = LINEUP_BY_NAME[name].display_name if name in LINEUP_BY_NAME else name
        print(f"  {rank:>2d}. {display:<24s} {rating:8.1f}")
    print()
    print(f"Total wall time: {elapsed:.1f}s (exceptions: {exceptions})")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
