# Knockout Game AI — Session Status

## What Was Built

### Reward Discovery (zero-knowledge, no hand-designed features)

- **Contrastive Trajectory Mining** (`src/knockout/reward/contrastive.py`, 19 tests) — 99.7% vs random, discovered 10 features via Cohen's d
- **Feature Attention Discovery** (`src/knockout/reward/attention_discovery.py`, 11 tests) — ~90% vs random, 8 features via gradient attribution
- **LLM Reward Architect** (`src/knockout/reward/llm_architect.py`, 33 tests) — fallback + API mode
- **Reward Co-Evolution** (`src/knockout/reward/coevolution.py`, 17 tests) — EMA blending, feature lifetime tracking

### Self-Play Training

- **Self-play trainer** (`scripts/train_self_play.py`) — checkpoint pool, ELO, adaptive random ratio, contrastive re-discovery
- **Checkpoint manager** (`src/knockout/training/checkpoint_manager.py`, 18 tests) — pause/resume, signal handlers, PID file
- **Resume script** (`scripts/resume_training.py`)

### LLM + MCTS Integration

- **5 MCTS+LLM approaches** (`src/knockout/agents/mcts_llm.py`, 2358 lines) — strategy proposer, discrete MCTS, game analyst, LLM value estimator, strategic MCTS
- **LLM CLI reward discovery** (`scripts/run_llm_reward_discovery.py`) — Claude writes reward functions via `claude --print`
- **Local Qwen 2.5-3B** (`models/qwen2.5-3b-instruct-q4_k_m.gguf`) — 74ms/eval, 2.3GB VRAM

### Infrastructure

- **Web dashboard** (`scripts/dashboard.py`) — Flask app at localhost:5000
- **Visualizations** (`scripts/visualize_results.py`) — 5 PNG plots
- **Tournament runner** (`scripts/run_tournament.py`) — evaluate any checkpoint
- **Pluggable opponents** — `TensorVecEnv.step(team_a, team_b)` + `get_team_b_obs()`

### Research Docs

- Design doc: `docs/design/reward_discovery.md`
- Deep dive: `docs/design/approaches_deep_dive.md`
- Research notes: `docs/research/reward_discovery_notes.md` (3400 words)

---

## Training Results (latest)

| Run | ELO | vs Random | vs Heuristic | Status |
|---|---|---|---|---|
| Contrastive | — | 99.7% | 0% | Done |
| Attention | — | ~90% | 0% | Done |
| LLM CLI v2 | — | 85% | — | Done (3 iters) |
| Self-Play v3 | 1090 | 100% | 0% | Paused |
| **Self-Play v4** | **1355** | **100%** | **0-3%** | **Running (iter 16/30)** |

### Self-Play v4 Progression

| Iter | Steps | ELO | vs Random | vs Pool | Random% | Features |
|---|---|---|---|---|---|---|
| 0 | 197K | 1025 | 100% | 90% | 50% | 0 |
| 3 | 786K | 1128 | 98% | 90% | 42% | 0 |
| 5 | 1.2M | 1203 | 100% | 97% | 35% | 0 |
| 8 | 1.6M | 1232 | 98% | 85% | 28% | 0 |
| 10 | 2.2M | ~1250 | — | — | 24% | 10 |
| 12 | 2.6M | 1318 | 98% | 100% | 20% | 10 |
| 14 | 2.9M | 1355 | 100% | 100% | 16% | 10 |

- ELO: 1000 → 1355 (16 iters)
- Random ratio: 50% → 16% (adaptive schedule working)
- Contrastive re-discovery found 10 features at iter 10
- First heuristic win seen at step 2.16M (3%), sporadic since

---

## Key Findings

1. **Reward discovery works** — contrastive mining found game mechanics from pure statistics
2. **Opponent quality > reward quality** — all approaches 0% vs heuristic when trained only vs random
3. **Self-play produces real arms race** — ELO 1000→1355, pool of 16 diverse checkpoints
4. **Claude discovers game mechanics** — wrote a self-correcting reward (0%→85%) from stats alone
5. **Local LLM viable for MCTS** — Qwen 3B at 74ms/eval on RTX 3070 Ti

---

## Next Steps

### Immediate
- [ ] Continue self-play v4 (14 more iters, ~2 hrs)
- [ ] Test MCTS Strategy Proposer against heuristic using local Qwen
- [ ] Integrate reward co-evolution into self-play

### Short-term
- [ ] Strategic MCTS (Approach 5) — novel: search over strategies, not actions
- [ ] Fine-tune Qwen 1.5B on game data from Claude + MCTS rollouts
- [ ] 1v1 curriculum → 3v3 composition

### Research
- [ ] Do discovered features change as opponents get smarter?
- [ ] LLM-guided MCTS vs pure RL self-play comparison
- [ ] Distill Claude reasoning into local model

---

## Quick Commands

```bash
# Dashboard
.venv/bin/python scripts/dashboard.py

# Training progress
tail -10 runs/self_play_v4/train.log

# Play against a checkpoint
python scripts/play.py --opponent-type ppo --opponent-checkpoint runs/self_play_v4/pool/pool_step_2949120.pt

# Tournament
PYTHONPATH=src .venv/bin/python scripts/run_tournament.py

# Test local LLM
PYTHONPATH=src .venv/bin/python scripts/test_local_llm.py

# Resume training
PYTHONPATH=src .venv/bin/python scripts/train_self_play.py --iterations 30 --num-envs 32 --device cuda --output-dir runs/self_play_v4 --start-checkpoint runs/self_play_v4/final_agent.pt
```
