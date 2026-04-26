"""Tier-2 mini tournament harness with ELO-sanity assertions.

Runs an ordered round-robin between every pair of available bots in
``eval.lineup`` and tracks ratings with :class:`ELOTracker`. After all games
finish, three sanity assertions are checked:

  (i)   distinct_ratings: at least three distinct integer ratings.
  (ii)  heuristic_beats_random: heuristic ELO is at least 100 above random.
  (iii) at_least_one_trained_above_random: at least one trained bot is
        at least 150 above random.

Exit code is 0 when all assertions pass and 1 otherwise.

Usage:
    python -m eval.mini_tournament [--games-per-pair 30] [--seed 42] \
                                   [--bots NAME,NAME,...]

Default is 30 games per ordered pair across every available bot in the
lineup. Each game uses a deterministic per-game seed; sides alternate per
game so each bot plays as Team A and Team B equally.
"""

from __future__ import annotations

import argparse
import json
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
)
from knockout.core.config import DEFAULTS
from knockout.training.elo_rating import ELOTracker
from knockout.training.evaluation import run_match

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "eval" / "results"

# Trained-bot names eligible for assertion (iii).
TRAINED_BOT_NAMES = (
    "ppo",
    "mappo",
    "contrastive",
    "attention_reward",
    "llm_cli_v2",
    "self_play",
)


def per_game_seed(base_seed: int, bot_a_idx: int, bot_b_idx: int, game_idx: int) -> int:
    """Canonical per-game seed used by every tournament harness.

    Formula: ``seed * 1_000_000 + bot_a_idx * 10_000 + bot_b_idx * 100 + game_idx``.
    Choosing constants well above the bot count and games-per-pair we ever
    expect avoids collisions across bots and games for the same base seed.
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
    """Parse CLI arguments for the mini tournament."""
    parser = argparse.ArgumentParser(
        description="Run a Tier-2 mini ELO tournament with sanity assertions.",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=30,
        help="Number of games per ordered pair (default: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; per-game seeds are derived deterministically (default: 42).",
    )
    parser.add_argument(
        "--bots",
        type=str,
        default=None,
        help="Comma-separated bot names to include. Defaults to all available bots.",
    )
    return parser.parse_args(argv)


def select_bots(arg: str | None) -> list[str]:
    """Resolve bot list from CLI argument, defaulting to all available bots.

    Args:
        arg: Comma-separated names or None.

    Returns:
        Ordered list of bot names that exist in the lineup and are available.

    Raises:
        SystemExit: If a requested bot is unknown or unavailable.
    """
    if arg is None:
        return [b.name for b in get_available_bots()]

    names = [n.strip() for n in arg.split(",") if n.strip()]
    available = {b.name for b in get_available_bots()}
    bad: list[str] = []
    for name in names:
        if name not in LINEUP_BY_NAME:
            bad.append(f"unknown bot: {name}")
        elif name not in available:
            ok, reason = LINEUP_BY_NAME[name].is_available()
            bad.append(f"bot not available ({name}): {reason}")
    if bad:
        for msg in bad:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(2)
    return names


def run_tournament(
    bots: list[str],
    games_per_pair: int,
    base_seed: int,
) -> tuple[ELOTracker, list[dict[str, Any]], int]:
    """Run a full ordered round-robin and return ELO + match log.

    For every ordered pair ``(a, b)`` with ``a != b`` we play
    ``games_per_pair`` games. On even ``g`` the first bot plays as Team A,
    on odd ``g`` they swap; ELO is updated with the appropriate score.
    Per-pair exceptions are caught: failing matches count as draws (0.5/0.5)
    so the tournament keeps running.

    Args:
        bots: Ordered list of bot names (drawn from :data:`LINEUP_BY_NAME`).
        games_per_pair: Number of games per ordered pair.
        base_seed: Base RNG seed for :func:`per_game_seed`.

    Returns:
        Tuple of (ELOTracker, list of per-match result dicts, total exception count).
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

    match_log: list[dict[str, Any]] = []
    exceptions = 0

    iterator = _try_tqdm(
        list(enumerate(pairs)),
        desc="pairs",
        unit="pair",
        leave=False,
    )

    for pair_idx, (i, j, name_a, name_b) in iterator:
        # tqdm absorbs the print, but bare iteration still gets a progress hint.
        prefix = f"[{pair_idx + 1}/{len(pairs)}]"
        print(f"{prefix} {name_a} vs {name_b}")
        sys.stdout.flush()

        spec_a = LINEUP_BY_NAME[name_a]
        spec_b = LINEUP_BY_NAME[name_b]

        for g in range(games_per_pair):
            game_seed = per_game_seed(base_seed, i, j, g)
            # Even games: a as Team A. Odd games: a as Team B (sides swap).
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
                error: str | None = None
            except Exception as exc:  # noqa: BLE001 -- we deliberately log + draw
                exceptions += 1
                traceback.print_exc()
                print(f"  {name_a} vs {name_b} game {g} raised {exc!r} -> draw")
                # Treat as a draw so the tournament keeps running.
                result = {"winner": -1, "steps": 0, "team_a_alive": 0, "team_b_alive": 0}
                error = repr(exc)

            winner = result["winner"]
            if winner == -1:
                score_a = 0.5
            elif a_as_team_a:
                score_a = 1.0 if winner == 0 else 0.0
            else:
                # bot a was Team B in this game.
                score_a = 1.0 if winner == 1 else 0.0

            tracker.update(name_a, name_b, score_a)

            match_log.append(
                {
                    "pair_index": pair_idx,
                    "game_index": g,
                    "bot_a": name_a,
                    "bot_b": name_b,
                    "a_as_team_a": a_as_team_a,
                    "seed": game_seed,
                    "winner": winner,
                    "steps": result.get("steps", 0),
                    "team_a_alive": result.get("team_a_alive", 0),
                    "team_b_alive": result.get("team_b_alive", 0),
                    "score_a": score_a,
                    "error": error,
                }
            )

    return tracker, match_log, exceptions


def evaluate_assertions(
    tracker: ELOTracker, bots: list[str]
) -> tuple[bool, list[tuple[str, bool, str]]]:
    """Check the three sanity assertions and return per-assertion outcomes.

    Args:
        tracker: ELO tracker after all matches.
        bots: Bots that participated in the tournament.

    Returns:
        Tuple of (all_passed, list of (label, passed, message)).
    """
    leaderboard = tracker.get_leaderboard()
    rounded = {round(r) for _, r in leaderboard}
    bot_set = set(bots)

    results: list[tuple[str, bool, str]] = []

    # (i) distinct_ratings
    passed_i = len(rounded) >= 3
    results.append(
        (
            "distinct_ratings",
            passed_i,
            f"{len(rounded)} distinct integer ratings (need >= 3)",
        )
    )

    # (ii) heuristic beats random by 100+ ELO (only when both present)
    if "heuristic" in bot_set and "random" in bot_set:
        h = tracker.get_rating("heuristic")
        r = tracker.get_rating("random")
        passed_ii = h > r + 100
        results.append(
            (
                "heuristic_beats_random",
                passed_ii,
                f"Heuristic {h:.0f} > Random {r:.0f} + 100 = {r + 100:.0f}",
            )
        )
    else:
        results.append(
            (
                "heuristic_beats_random",
                True,
                "skipped (heuristic and/or random not in lineup)",
            )
        )

    # (iii) at least one trained bot above random by 150+
    if "random" in bot_set:
        r = tracker.get_rating("random")
        threshold = r + 150
        present_trained = [n for n in TRAINED_BOT_NAMES if n in bot_set]
        if not present_trained:
            results.append(
                (
                    "at_least_one_trained_above_random",
                    True,
                    "skipped (no trained bots in lineup)",
                )
            )
        else:
            best_name = max(present_trained, key=tracker.get_rating)
            best_rating = tracker.get_rating(best_name)
            passed_iii = best_rating > threshold
            results.append(
                (
                    "at_least_one_trained_above_random",
                    passed_iii,
                    f"Best trained ({best_name}) {best_rating:.0f} > Random {r:.0f} + 150 = {threshold:.0f}",
                )
            )
    else:
        results.append(
            (
                "at_least_one_trained_above_random",
                True,
                "skipped (random not in lineup)",
            )
        )

    all_passed = all(p for _, p, _ in results)
    return all_passed, results


def print_leaderboard(tracker: ELOTracker, bots: list[str]) -> None:
    """Print a simple leaderboard table including display names."""
    leaderboard = tracker.get_leaderboard()
    leaderboard = [(n, r) for n, r in leaderboard if n in set(bots)]
    print()
    print("=== Leaderboard ===")
    print(f"  {'Rank':>4s}  {'Bot':<22s}  {'Display':<22s}  {'Rating':>8s}")
    print(f"  {'-' * 4}  {'-' * 22}  {'-' * 22}  {'-' * 8}")
    for rank, (name, rating) in enumerate(leaderboard, start=1):
        display = LINEUP_BY_NAME[name].display_name if name in LINEUP_BY_NAME else name
        print(f"  {rank:>4d}  {name:<22s}  {display:<22s}  {rating:>8.1f}")


def print_assertions(results: list[tuple[str, bool, str]]) -> None:
    """Render the three pass/fail assertions to stdout."""
    print()
    print("=== Assertions ===")
    for label, passed, msg in results:
        marker = "PASS" if passed else "FAIL"
        print(f"  ({label}) {msg}  [{marker}]")


def write_json(
    bots: list[str],
    games_per_pair: int,
    base_seed: int,
    tracker: ELOTracker,
    match_log: list[dict[str, Any]],
    assertions: list[tuple[str, bool, str]],
    elapsed_s: float,
    exceptions: int,
    out_path: Path,
) -> None:
    """Persist the full tournament record as JSON for later inspection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_seed": base_seed,
        "n_games_per_pair": games_per_pair,
        "bots": bots,
        "elapsed_seconds": elapsed_s,
        "exceptions": exceptions,
        "leaderboard": [
            {"name": name, "rating": rating}
            for name, rating in tracker.get_leaderboard()
            if name in set(bots)
        ],
        "assertions": [
            {"label": label, "passed": passed, "message": msg}
            for label, passed, msg in assertions
        ],
        "matches": match_log,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m eval.mini_tournament``."""
    args = parse_args(argv)
    bots = select_bots(args.bots)

    if len(bots) < 2:
        print("ERROR: need at least 2 bots in the tournament", file=sys.stderr)
        return 2

    print(f"Running mini tournament: {len(bots)} bots, "
          f"{args.games_per_pair} games per ordered pair, base seed {args.seed}")
    print(f"  bots: {', '.join(bots)}")

    t0 = time.perf_counter()
    tracker, match_log, exceptions = run_tournament(
        bots=bots,
        games_per_pair=args.games_per_pair,
        base_seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    print_leaderboard(tracker, bots)
    all_passed, assertion_results = evaluate_assertions(tracker, bots)
    print_assertions(assertion_results)

    failed = [label for label, passed, _ in assertion_results if not passed]
    print()
    if all_passed:
        print("ALL PASS")
    else:
        print(f"FAILED: {', '.join(failed)}")
    print(f"Total wall time: {elapsed:.1f}s "
          f"(matches: {len(match_log)}, exceptions: {exceptions})")

    out_path = RESULTS_DIR / "mini_tournament.json"
    write_json(
        bots=bots,
        games_per_pair=args.games_per_pair,
        base_seed=args.seed,
        tracker=tracker,
        match_log=match_log,
        assertions=assertion_results,
        elapsed_s=elapsed,
        exceptions=exceptions,
        out_path=out_path,
    )
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
