# Knockout

> **Status: work in progress.** Public preview of an audit pass. The core engine, environment,
> agents, training pipelines, and 472-test pytest suite are in place and functional. The
> evaluation harness in `eval/` ships `lineup.py`, `smoke.py`, `mini_tournament.py`,
> `full_tournament.py`, `manual_checks.md`, and `DEBUG_LOG.md`. `scripts/play.py` is rewired
> to use the lineup via `--opponent NAME`. Still queued for the next pass:
> `eval/pairwise_smoke.py`, `eval/render_smoke.py`, `scripts/watch.py`, and the deletion of
> superseded scripts. The harness's first run already surfaced three real findings (action-space
> spec mismatches in `random` and `mappo`; policy collapse in the `self_play` final checkpoint).
> See [WIP_STATUS.md](WIP_STATUS.md) for the full done/queued/known-broken breakdown.

Knockout is a 3v3 penguin physics game (think GamePigeon Knockout): two teams launch penguins
across a square ice sheet that shrinks every five rounds, and the team with at least one penguin
remaining at round 25 wins. This repo is a self-contained ML/RL playground built around that
game — physics, environment, agents, training, and evaluation harness.

The interesting bits are research-flavoured: an LLM-in-loop reward designer (Claude authors a
shaped reward, training stats are fed back, and the reward is iterated), three zero-knowledge
reward-discovery methods (contrastive Cohen's d, gradient-attribution attention, LLM-architect),
PPO and MAPPO with self-play against an evolving snapshot pool, and a CUDA-tensor physics engine
that hits roughly 19K env-steps/second at 4096 parallel environments. Nine bots live in the
`eval/lineup.py` registry and play a round-robin tournament on demand.

---

## Quick start

### Install

```bash
git clone <repo>
cd knockout-v2
uv sync                          # or: pip install -e ".[viz,dev]"
```

`uv` is recommended; the `viz` extra pulls in Pygame for the watcher and play mode, `dev` pulls
pytest/ruff/mypy. The bundled virtualenv is `.venv/`.

### Run a tournament

```bash
.venv/bin/python -m eval.full_tournament
```

Round-robin between every available bot in `eval/lineup.py`. Bots whose checkpoints or env
vars are missing are skipped automatically. The leaderboard, win-rate matrix, and skip list
are written to `eval/results/leaderboard.md`.

### Watch a game

```bash
.venv/bin/python scripts/watch.py --team-a heuristic --team-b ppo
```

Renders a Pygame window showing the round-by-round physics. `--team-a` and `--team-b` accept
any short bot name from the lineup table below.

### Play yourself

```bash
.venv/bin/python scripts/play.py --opponent ppo
```

Click-and-drag aiming on cyan penguins (Team A); SPACE to launch. Right-click auto-aims via
the heuristic. `--opponent` accepts the same names as the watcher.

> Note: `eval/full_tournament.py`, `eval/mini_tournament.py`, `eval/smoke.py`,
> `eval/pairwise_smoke.py`, `eval/render_smoke.py`, `eval/manual_checks.md`, `scripts/watch.py`,
> and `scripts/play.py` (with the unified `--opponent` flag) are part of the audit polish pass
> — see `eval/lineup.py` and `scripts/play_llm.py` / `scripts/watch_game_visual.py` for the
> current preliminary entry points if a file above is missing in your checkout.

---

## Bot lineup

Source of truth: `eval/lineup.py`. Bots that fail their requirement check (missing checkpoint
or unset env var) are skipped at runtime.

| Name              | Display Name          | Type     | Source / Training                                                     | Notes                                                                 |
|-------------------|-----------------------|----------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
| `random`          | Random                | scripted | Uniform random angle and power                                        | Floor of the capability ladder                                        |
| `heuristic`       | Heuristic             | scripted | Hand-crafted target-utility scoring                                   | Predictive aim, edge awareness; strong scripted baseline              |
| `ppo`             | PPO                   | PPO      | Single-agent PPO vs random+heuristic curriculum, ~3M steps            | `checkpoints/ppo_overnight/best.pt`                                   |
| `mappo`           | MAPPO                 | MAPPO    | Multi-agent PPO, shared centralized critic, three policies, ~1.7M steps | `checkpoints/mappo_overnight/best.pt`                               |
| `self_play`       | Self-Play (v4)        | PPO      | Self-play vs evolving pool of snapshots, 5.9M steps                   | Final-iter in-pool ELO 1443; may show policy collapse at final ckpt   |
| `contrastive`     | Contrastive Reward    | PPO      | Reward shaping from Cohen's d on win/loss trajectories, vs random     | Zero-knowledge feature discovery                                      |
| `attention_reward`| Attention-Reward      | PPO      | Reward shaping from gradient attribution on the value head            | "Attention" refers to the discovery method, not the network           |
| `llm_cli_v2`      | LLM-Reward (Claude)   | PPO      | Reward function authored by Claude (CLI), iterated on training stats  | Recovered from 0%-win catastrophe to 84.8% across three iterations    |
| `llm_anthropic`   | LLM (Anthropic)       | LLM      | Claude reasons over a serialized game state, returns angle/power      | Slow (network round-trip per turn); requires `ANTHROPIC_API_KEY`      |

The `llm_anthropic` bot is the only entry that calls a remote model at decision time. It needs
`ANTHROPIC_API_KEY` exported in the environment; if unset, the bot is excluded from tournaments
and CLI commands gracefully. It is slower than every other bot by 2-3 orders of magnitude
because each turn does a network round-trip; expect single-game (not tournament) usage.

---

## Results — ELO leaderboard

> **Results placeholder.** ELO leaderboard generated by `eval/full_tournament.py` lands in
> `eval/results/leaderboard.md` after the validation harness runs. The numbers below will be
> regenerated at audit close.

| Rank | Bot     | Rating |
|------|---------|--------|
| 1    | _TBD_   | _TBD_  |
| 2    | _TBD_   | _TBD_  |
| 3    | _TBD_   | _TBD_  |
| 4    | _TBD_   | _TBD_  |
| 5    | _TBD_   | _TBD_  |

---

## Architecture

**Physics.** Pymunk-based, square arena (half-width 100), knockout-by-edge: a penguin whose
center leaves the ice sheet is removed. The sheet shrinks 0.67x at rounds 5/10/15/20 (four
shrink events), with penguin positions rescaled rather than eliminated. The CPU `PhysicsEngine`
(`src/knockout/core/physics_engine.py`) drives single-environment rollouts; the GPU
`TensorPhysicsEngine` (`src/knockout/core/tensor_physics.py`) is a fully vectorized rewrite that
runs penguin–penguin collisions, damping, and edge tests as torch ops. It clocks roughly
19K env-steps/sec at 4096 parallel environments — the difference between training overnight
and training over a long weekend.

**Environment.** PettingZoo Parallel API in `src/knockout/env/penguin_env.py` (single env) and
`src/knockout/env/tensor_env.py` (vectorized). Observations are 89-dim per agent: ego state
(position, velocity, alive flag), two allies, three enemies, and global game state (round
number, ice radius, etc.). Actions are continuous 2D per turn — `angle ∈ [0, 360]` and
`power ∈ [0, 400]` — applied as a single launch impulse on the penguin's center.

**Training.** Three trainers live in `src/knockout/training/`: PPO (`ppo.py`), MAPPO with a
shared centralized critic (`mappo.py`), and self-play (`self_play.py`) which maintains an
evolving pool of past snapshots and adapts the random-opponent ratio over time. The reward
side has three zero-knowledge methods in `src/knockout/reward/`: `contrastive.py` (Cohen's d
on win/loss trajectories), `attention_discovery.py` (gradient attribution on the value head),
and `llm_architect.py` (Claude authors a reward function, training stats feed the next
iteration). All checkpoints land in `checkpoints/` (production) and `runs/` (experiment
outputs, including snapshot pools).

---

## Validation harness

`eval/` is the audit-time test pyramid: each tier is faster, looser, and cheaper than the next.

- `eval/smoke.py` — Tier 1: every bot loads, takes finite actions over 100 steps,
  action variance is non-degenerate (catches a frozen / NaN / degenerate-distribution policy).
- `eval/pairwise_smoke.py` — Tier 1.5: a 1-round headless match between every pair of bots
  (catches obs-shape and action-shape mismatches across team boundaries).
- `eval/render_smoke.py` — automated rendering correctness: render-to-PNG for a fixed seed
  to catch silent visualization regressions.
- `eval/mini_tournament.py` — Tier 2: 30 games per ordered pair with three sanity assertions
  (heuristic must be ≥ 100 ELO above random; at least one trained bot ≥ 150 above random;
  ≥ 3 distinct ratings).
- `eval/full_tournament.py` — Tier 3: full round-robin at higher games-per-pair, no
  assertions, generates `eval/results/leaderboard.md`.
- `eval/manual_checks.md` — the one-time human-in-loop check for Pygame click-drag input
  (the only thing that can't be automated cheaply).

Everything except `manual_checks.md` is automated. The unit suite under `tests/` is
roughly 472 tests covering physics, environments, agents, training, and reward modules.

---

## Project layout

```
knockout-v2/
├── src/knockout/        # package: core/, env/, agents/, training/, reward/, visualization/
├── scripts/             # play.py, watch.py, train_*.py, dashboard.py
├── eval/                # smoke.py, *_tournament.py, lineup.py, results/
├── tests/               # pytest suite (~472 tests)
├── checkpoints/         # PPO, MAPPO trained weights
├── runs/                # self-play and reward-discovery experiment outputs
└── docs/                # design + research notes
```

---

## Future work

Known gaps and live research directions:

- **Self-Play v4 final-checkpoint policy collapse** — re-train from a stable pool snapshot,
  or train longer with stricter entropy regularization. The in-pool ELO is fine; the released
  final policy regresses against the hand-crafted heuristic.
- **LLM-Architect actual API loop** — the shipped checkpoint used a CLI subprocess wrapper
  with template fallback; re-run with the real Anthropic SDK in the loop so the reward
  authoring genuinely tracks training stats.
- **Local Qwen agent wrapper** — the gguf weights at `models/qwen2.5-3b-instruct-q4_k_m.gguf`
  (~74 ms/eval, 2.3 GB VRAM) are sitting unused; write a `QwenAgent` analog of `LLMAgent`
  for an offline reasoning baseline that doesn't require an API key.
- **ELO progression visualization** — `runs/self_play_v4/pool/pool_step_*.pt` stores every
  snapshot in the pool; plot ELO over time across the whole pool, not just the final number.
- **1v1 → 3v3 curriculum** — current PPO/MAPPO max out at ~7% win rate vs the heuristic.
  A staged curriculum (1v1 first, then 2v2, then 3v3) is the next thing to try for crossing
  the heuristic ceiling.
- **MCTS + LLM strategy proposer** — Qwen as the strategy generator and PPO as the executor
  (skeleton in `src/knockout/agents/mcts_llm.py`, not wired into training yet).
- **Reward co-evolution in self-play** — periodic reward re-discovery interleaved with the
  self-play loop (`src/knockout/reward/coevolution.py` is the starting point).
- **Cross-validation of CPU vs GPU physics** — make sure agents trained against the
  `TensorPhysicsEngine` transfer cleanly to the Pymunk reference engine; we've audited the
  physics by hand but not built a transfer-equivalence test.

---

## Reference

**Research notes.** `RESEARCH_NOTES.md` (audit deliverable) collects historical findings —
positive and negative — from past training experiments, including the LLM-CLI catastrophe-
recovery story and the self-play ELO progression.

**Reference implementations.** Two sibling folders are preserved as read-only references and
are not actively maintained:

- `../knockout-claude/` — the original Pymunk + PettingZoo architecture (had seven known
  bugs; `knockout-v2` is the consolidated rewrite).
- `../knockout-ai-codex/` — the Codex Pygame UI version with a simpler training loop.

---

## Roadmap

Longer-term direction in `../ROADMAP.md`.
