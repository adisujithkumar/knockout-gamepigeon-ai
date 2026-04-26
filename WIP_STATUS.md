# WIP Status

This repo is undergoing a polish-and-publish audit pass. This file tracks what's
done, what's queued, and what's known-broken so a reader doesn't have to spelunk
through the git log to find the seams.

## Done (committed, working)

- **Engine and env.** `src/knockout/core/`, `src/knockout/env/` — Pymunk physics,
  PettingZoo Parallel API, square arena with periodic shrink, 89-dim obs, 2-dim
  continuous actions. CPU `PhysicsEngine` and GPU `TensorPhysicsEngine` (the latter
  ~19K env-steps/s at 4096 envs).
- **Bot lineup.** `eval/lineup.py` — single source of truth for every bot
  (Random, Heuristic, PPO, MAPPO, Self-Play v4, Contrastive, Attention-Reward,
  LLM-CLI-v2, LLM-Anthropic). All eight non-LLM bots verified to load and produce
  finite actions.
- **Tier 1 + 1.5 smoke harness.** `eval/smoke.py` — every available bot is
  instantiated, run for 100 env steps, and the resulting `(N, 2)` action array
  is checked against six assertions (finiteness, angle/power variance, range,
  magnitude). Per-bot pass/fail table + JSON report at `eval/results/smoke_report.json`.
  Surfaced three real findings on first run; see "Known issues" below.
- **Tier 2 mini-tournament.** `eval/mini_tournament.py` — 30 games per ordered
  pair (configurable), alternating sides, per-game seeds. Three sanity assertions
  (heuristic > random + 100 ELO, ≥ 1 trained > random + 150, ≥ 3 distinct ratings).
  All three assertions PASS at default settings.
- **Full tournament.** `eval/full_tournament.py` — round-robin at higher
  games-per-pair, no assertions, generates `eval/results/leaderboard.md` with
  win-rate matrix and skipped-bot footer.
- **Lineup-driven play.** `scripts/play.py` — rewired to read from `eval/lineup.py`
  via `--opponent NAME`. Replaces the old two-flag `--opponent-type/--opponent-checkpoint`
  interface. `--list` prints the lineup.
- **Training pipelines.** `src/knockout/training/` — PPO, MAPPO with shared
  centralized critic, self-play with snapshot pool, three reward-discovery
  modules (`src/knockout/reward/`).
- **pytest suite.** ~472 tests under `tests/`. Last full-run: see git log.
- **Manual-check protocol.** `eval/manual_checks.md` — the one human-in-loop
  Pygame check, scoped narrowly to click-drag input handling.
- **Debug-log scaffold.** `eval/DEBUG_LOG.md` — format conventions for the audit.
- **Research notes.** `RESEARCH_NOTES.md` — historical findings (positive AND
  negative) from past experiments, including the LLM-CLI 0%→85% catastrophe
  recovery and the train-on-random-get-0%-vs-heuristic ceiling.
- **README.md.** Front door, lineup table, architecture, future-work section.
- **`.gitignore` + git baseline.** Pre-audit baseline tagged `pre-audit` on
  `master`; audit work proceeds on `audit/main`.

## Queued (not yet committed; next pass)

- `eval/pairwise_smoke.py` — 1-round headless match per ordered pair, traceback
  capture for any exception. (Some of this coverage is implicit in
  `mini_tournament.py` already, but the per-pair traceback dump is missing.)
- `eval/render_smoke.py` — automated render-to-PNG correctness (frame variance,
  no-crash, non-uniform pixel content).
- `scripts/watch.py` — bot-vs-bot Pygame visualization, lineup-driven. The
  preliminary `scripts/watch_game_visual.py` (random-vs-random) and
  `scripts/watch_heuristic_game.py` (heuristic-vs-random) cover the basic
  use case until the unified script lands.
- Deletion of superseded scripts (`scripts/watch_game.py`,
  `scripts/watch_game_visual.py`, `scripts/watch_heuristic_game.py`,
  `scripts/play_llm.py`) — once `scripts/watch.py` is in place.
- Top-level cleanup: delete `STATUS.md`, `STATUS2.md`, root `tournament.py`,
  empty `attention_discovery_log.json`, orphaned `runs/llm_cli_train.log`.
  All findings already preserved in `RESEARCH_NOTES.md`.
- Delete superseded run lineages (`runs/self_play/`, `runs/self_play_v2/`,
  `runs/self_play_v3/`, `runs/llm_cli/`, `runs/llm_architect/`) once we're
  confident the kept lineages still load post-audit.
- Run final tournament at higher games-per-pair (50+) and patch the README's
  leaderboard placeholder with the real numbers.

## Known issues surfaced by the harness

Three real findings from the first `eval/smoke.py` run, all genuine bugs/spec
mismatches the audit was designed to catch:

- **`random` bot violates the action-space spec.** `RandomAgent` samples power
  from `[0, 500]` but the documented action range is `[0, 400]` (and
  `MAX_LAUNCH_FORCE = 400` in `core/config.py`). The smoke probe's
  `power_in_range` assertion catches this. Fix: clamp the sampling distribution
  to `[0, MAX_LAUNCH_FORCE]` in `RandomAgent`.
- **`mappo` bot violates the action-space spec.** Same `power_in_range`
  failure. The MAPPO actor outputs values that, after sigmoid-scaling, exceed
  400. Likely a scaling-constant mismatch with the per-agent config used at
  training time. Fix: audit the MAPPO action-scaling pipeline against
  `MAX_LAUNCH_FORCE`.
- **`self_play` final-checkpoint policy collapse.**
  `runs/self_play_v4/final_agent.pt` has `actor_log_std=[3.71, 0.58]`; sampled
  actions peg to angle≈0 / power≈0. Loadable but degenerate. Fix in next pass:
  swap the lineup entry to a stable pool snapshot (likely
  `runs/self_play_v4/pool/pool_step_3735552.pt` or similar) and re-anchor ELO
  empirically against random/heuristic.

## Other known-broken / drop-from-default-lineup

- **LLM-Architect runs** (`runs/llm_architect/`). Three iterations of
  template-fallback masquerading as Claude-authored rewards — the LLM was
  never actually called. The shipped lineup uses `runs/llm_cli_v2/iteration_002/`
  (real Claude calls) instead. Re-running llm_architect with the real
  Anthropic SDK in the loop is in the README's Future Work.
- **"Attention" naming.** `runs/attention/` is reward-discovery via gradient
  attribution; the network is a plain MLP. The lineup label is
  `attention_reward` to make the distinction explicit.

## How to read the git history

- `master` (tag `pre-audit`): the state before this audit pass started, for
  rollback safety.
- `audit/main`: where the polish work is happening. Every meaningful change is
  a commit with a `Mega-Phase N / X:` prefix so the audit shape is visible in
  the log.
