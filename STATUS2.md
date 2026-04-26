# Knockout Game AI — Session Status

## What Was Built

### Reward Discovery (zero-knowledge, no hand-designed features)
| Module | File | Tests | Result |
|---|---|---|---|
| Contrastive Trajectory Mining | `src/knockout/reward/contrastive.py` | 19 | 99.7% vs random, 10 features discovered |
| Feature Attention Discovery | `src/knockout/reward/attention_discovery.py` | 11 | ~90% vs random, 8 features crystallized |
| LLM Reward Architect | `src/knockout/reward/llm_architect.py` | 33 | Fallback + API mode |
| Reward Co-Evolution | `src/knockout/reward/coevolution.py` | 17 | EMA blending, feature lifetime tracking |

### Self-Play Training
| Component | File |
|---|---|
| Self-play trainer | `scripts/train_self_play.py` — checkpoint pool, ELO, adaptive random ratio |
| Checkpoint manager | `src/knockout/training/checkpoint_manager.py` — pause/resume/signal handlers |
| Resume script | `scripts/resume_training.py` |

### LLM + MCTS Integration
| Component | File |
|---|---|
| 5 MCTS+LLM approaches | `src/knockout/agents/mcts_llm.py` (2358 lines) |
| LLM CLI reward discovery | `scripts/run_llm_reward_discovery.py` |
| Local Qwen 2.5-3B | `models/qwen2.5-3b-instruct-q4_k_m.gguf` — 74ms/eval, 2.3GB VRAM |

### Infrastructure
| Component | File |
|---|---|
| Web dashboard | `scripts/dashboard.py` — localhost:5000 |
| Visualizations | `scripts/visualize_results.py` — 5 plots |
| Tournament runner | `scripts/run_tournament.py` |
| Pluggable opponents | `tensor_env.step(team_a, team_b)` |

## Training Results

| Run | ELO | vs Random | vs Heuristic | Status |
|---|---|---|---|---|
| Contrastive | — | 99.7% | 0% | Done |
| Attention | — | ~90% | 0% | Done |
| LLM CLI v2 | — | 85% | — | Done |
| Self-Play v4 (iter 9) | **1237** | 98% | **3% (1/30)** | **Running** |

**First heuristic win detected at step 2.16M (ELO 1237).** Still 3%, but nonzero for the first time.

## Key Findings

1. Reward discovery finds WHAT matters — but opponent quality determines HOW agents learn to exploit it
2. Self-play ELO climbing 1000→1237 with adaptive random ratio (50%→28%)
3. Claude discovered game mechanics from pure statistics and self-corrected a failed reward (0%→85%)
4. Local Qwen 3B runs at 74ms/eval — viable for real-time MCTS

## Next Steps

### Immediate
- Continue self-play v4 (21 more iters, ~3 hrs) — heuristic wins should increase
- Test MCTS Strategy Proposer against heuristic using local Qwen
- Integrate reward co-evolution into self-play

### Research
- Do discovered features change as opponents get smarter? (co-evolution)
- LLM-guided MCTS vs pure RL self-play comparison
- Fine-tune Qwen 1.5B on game data from Claude + MCTS rollouts
- 1v1 curriculum → 3v3 composition

## Quick Commands
```bash
.venv/bin/python scripts/dashboard.py                    # Web UI
tail -10 runs/self_play_v4/train.log                     # Training progress
python scripts/play.py --opponent-type ppo --opponent-checkpoint runs/self_play_v4/pool/pool_step_1572864.pt
PYTHONPATH=src .venv/bin/python scripts/run_tournament.py # Full tournament
PYTHONPATH=src .venv/bin/python scripts/test_local_llm.py # Test local LLM
```
