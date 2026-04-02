# Knockout v2 — Agent Context

## Project Overview
3v3 penguin physics knockout game. Pymunk physics, PettingZoo Parallel API, circular arena, simultaneous actions.
Consolidated from knockout-claude (best architecture) + knockout-ai-codex (training/UI).

## Current State
- Phase 0: COMPLETE (scaffold)
- Phase 1: COMPLETE (core physics) — 70 tests
- Phase 2: COMPLETE (environment) — 57 tests
- Phase 3: COMPLETE (agents) — 31 tests
- Phase 4: COMPLETE (visualization) — renderer + game_viewer
- Phase 5: COMPLETE (training) — 27 tests
- Phase 6: COMPLETE (integration + regression) — 20 tests
- **TOTAL: 205 tests passing**

## Import Pattern
All source lives under `src/knockout/`. Imports use: `from knockout.core.config import GameConfig, DEFAULTS`

## Key Constants
- Arena: circular, radius=100
- Teams: 3v3 (penguin_0..2 = Team A, penguin_3..5 = Team B)
- Physics: 60Hz fixed timestep, Pymunk, COLLISION_BIAS=pow(1-0.1,60)≈0.0018
- Obs: 89-dim vector per agent, all values clipped to [-1, 1]
- Actions: continuous [angle_degrees(0-360), power_newtons(0-500)]

## Bug Fixes Applied (all verified by regression tests)
1. `apply_impulse_at_world_point` instead of `apply_impulse_at_local_point`
2. Instance `np.random.default_rng(seed)` instead of global `np.random.seed()`
3. `COLLISION_BIAS = pow(1.0 - 0.1, 60.0)` instead of 0.2
4. `np.clip(value, -1.0, 1.0)` on all obs features
5. True relative velocity: `other_vel - ego_vel`
6. Geometric sum prediction: `vel * dt * (1 - d^N) / (1 - d)`
7. Utility weights sum to 1.0: edge(0.4) + distance(0.25) + vel_toward_edge(0.35)

## File Layout
```
src/knockout/
  core/config.py, ice_sheet.py, penguin.py, physics_engine.py
  env/observations.py, penguin_env.py, wrappers.py
  agents/base.py, random_agent.py, heuristic_agent.py, rl_agent.py
  reward/strategy_reward.py (stub)
  training/rollout_buffer.py, ppo.py, self_play.py, elo_rating.py, evaluation.py, plots.py
  visualization/renderer.py, game_viewer.py
scripts/test_physics.py, watch_game.py, watch_game_visual.py, train.py, benchmark_agents.py
tests/ mirrors src/ structure + test_integration.py + test_regression.py
```

## Running
- Tests: `uv run pytest tests/ -v`
- Train: `uv run python scripts/train.py --timesteps 10000`
- Benchmark: `uv run python scripts/benchmark_agents.py`
- Watch (headless): `uv run python scripts/watch_game.py`
- Watch (visual): `uv run python scripts/watch_game_visual.py`
