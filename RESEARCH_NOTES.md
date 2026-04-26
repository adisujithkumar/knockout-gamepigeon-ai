# Research Notes — Knockout v2 Training Experiments

This document is the historical record of the training experiments conducted in `knockout-v2` prior to the audit pass. It consolidates findings from approximately twelve runs spanning PPO/MAPPO baselines, three reward-discovery methods (contrastive, attention, LLM architect), an LLM-CLI reward iteration loop (two versions), and four self-play lineages (v1 → v4). Old `runs/` directories are scheduled for deletion as part of the audit cleanup; this file preserves the substantive findings — both positive and negative — so they remain referenceable after the artifacts are gone. Source citations are given against the run logs and CSVs that produced each number, so any claim here can be verified before the artifacts are pruned.

## Headline Finding

**Agents trained only against random opponents reach roughly 99% win rate vs random and 0–3% win rate vs the heuristic, regardless of how their reward signal is generated.** This holds across PPO, MAPPO, contrastive reward discovery, attention-based reward discovery, and the fallback-templated `llm_architect` runs. The opponent distribution dictates the *strategy class* the policy learns; the reward signal only determines *what it optimizes within that class*. Reward discovery finds what matters; opponents determine how to exploit it. Self-play v4 (the only run that completed all 30 iterations) climbed from ELO 1000 → 1443 in approximately 5.9M environment steps, but it dropped in-loop heuristic evaluation from the loop entirely, so the often-cited "0–3% vs heuristic" number for v4 cannot be reproduced from its own log. The figures quoted in the prior `STATUS.md` for v4 are aspirational and originate in the earlier PPO/MAPPO and contrastive/attention baselines.

## Self-Play Lineage (v1 → v4)

### self_play / "v1" — interrupted, 2 iterations logged

Started from `runs/contrastive/final_agent.pt`, with 64 environments and 8 rollouts per iteration. Both logged iterations used a **pure random opponent**, and the run was interrupted before iteration 3 (from `runs/self_play/training_log.csv`). ELO went 1000 → 1033 with 100% win rate vs random, but win rate vs the pool collapsed from 84% → 67% over the same span — direct evidence that random-only training *degrades* play vs pool. The log file contains interleaved and duplicated rollout lines suggesting concurrent writers (from `runs/self_play/train.log`) and should be treated as partly corrupted.

### self_play_v2 — immediate failure

A single iteration that never produced rollout data. The log contains only boot lines and a duplicated header (from `runs/self_play_v2/train.log`); no `training_log.csv` was written. Treat as a launch glitch with no usable results.

### self_play_v3 — intentional shutdown after 4 iterations

Same starting checkpoint as v1 (`runs/contrastive/final_agent.pt`), but switched to a **pool-based opponent from iteration 1** with a 50%/48% adaptive random ratio. ELO progressed 1000 → 1029 → 1066 → 1075 → **1090** across 4 iterations and 638,976 steps. Iteration 2 used a pure-random opponent and pool win rate dropped 96% → 68.5%, replicating v1's failure mode; iteration 3 recovered to 72.8% by training vs the pool again (from `runs/self_play_v3/training_log.csv`). The run was killed by SIGTERM at iter 4 rollout 2 (from `runs/self_play_v3/train.log`). Final leaderboard: learner ELO 1090 vs pool members 973–994; pool size 5.

### self_play_v4 — ran to completion, 30 iterations, 5.9M steps

Started from `runs/self_play_v3/final_agent.pt`. Switched to **32 envs and 16 rollouts per iter** (~196,608 steps/iter), introduced a max-pool of 20 with eviction, and kept the adaptive random-opponent ratio (50% → 5% over the 30 iterations).

| Iter | Steps | ELO  | vs Random | vs Pool | Random% |
|-----:|------:|-----:|----------:|--------:|--------:|
| 1    | 197K  | 1025 | 100%      | 90%     | 50%     |
| 5    | 983K  | 1170 | 100%      | 98%     | 44%     |
| 10   | 1.97M | 1247 | 100%      | 89%     | 24%     |
| 15   | 2.95M | 1355 | 100%      | 100%    | 17%     |
| 20   | 3.93M | 1414 | 100%      | 92%     | 14%     |
| 24   | 4.72M | 1445 | 100%      | 100%    | 11%     |
| 30   | 5.90M | **1443** | 100%  | 89%     | 5%      |

The final iteration completed at 09:04 on 2026-04-07 (from `runs/self_play_v4/train.log` line 538). `STATUS.md` is stale: it claims "iter 16/30 running," but the run actually finished — ELO 1355 was iter 15, and the final ELO is 1443 (from `runs/self_play_v4/training_log.csv`).

#### v4 anomalies and loose ends

1. **Iter 24 took 27,837 seconds (7h 44m).** The run hung between rollouts 6 and 9 from `2026-04-07 00:38:17` to `08:17:16` — a 7h 39m gap, almost certainly machine sleep/suspend — before resuming cleanly (from `runs/self_play_v4/train.log` lines 400–401). All other iterations ran in 313–1300s.
2. **Iter 21 produced 0 rollout episodes** (`win_rate=0.00 (0/0)` for all rollouts, only 2 episodes total, win rate vs opponent 0.5). The agent had just been crushed 62% by `pool_step_3932160` at iter 20, then refused to play out new episodes against that same checkpoint.
3. **`pool_step_3932160` is a pool ringer.** It beat the learner 38% / 50% / 30% / 26% / 56% across iterations 20, 21, 25, 28, and 30, while every other pool member was demolished 100%–100%. The final pool index (`runs/self_play_v4/pool_index.json`) shows it at ELO 1037 — second only to `pool_step_3735552` (1028) among non-learner entries. The pool contains adversarial niches the learner never solves.
4. **Pool ELOs in `pool_index.json` diverge wildly from in-log ELOs.** For example, `pool_step_4521984` is recorded at ELO 1423 in the index, but other pool members are stuck at 997–1038. The index appears to have been written with the *learner's* ELO at save time, not the pool member's; do not trust those numbers without a re-evaluation pass.
5. **No in-loop heuristic eval in v4.** A search of the v4 log finds zero occurrences of "heuristic." `STATUS.md`'s "0–3% vs heuristic" for v4 is not corroborated by the v4 log itself — that figure originates in the earlier PPO/MAPPO baselines and the contrastive/attention runs.
6. **Contrastive re-discovery ran inside v4** at iters 5, 10, 15, 20, 25, and 30. Iter 5 found nothing (the learner was already winning too consistently to produce losses, so Cohen's d was undefined). From iter 10 onward it kept finding around 10 features per pass, but the *features themselves drifted*: iter 10 was ally-survival positional features (`ally2.position_y` d = +1.67, `global.team_a_alive` d = +1.61); iter 20 swung to enemy-pressure features (`enemy2.dist_from_center` d = +1.00); iter 30 mixed both with smaller effect sizes (max d ≈ 1.10). Effect sizes shrank as games tightened. This is partial evidence for the "reward co-evolution" hypothesis.

## Reward-Discovery Experiments

### contrastive — 5 iterations, ~11 hours

Cohen's d on win/loss trajectories was converted into PPO reward shaping. Trained against **random only**. Final win rate 99.7% vs random and 0% vs heuristic (from `docs/research/reward_discovery_notes.md`). The killer failure mode: **iteration 2 discovered 0 features**, because the agent won 5983/5983 episodes — Cohen's d is undefined when there are no losses (from `runs/contrastive/train.log` lines 38–45). Iteration 1 took 22,486 seconds (6.2 hours) — the slowest single iteration in the entire codebase, roughly 10× iter 0. Iteration 4 introduced `enemy1.dist_to_ego` with d = +0.973 *positive*, meaning the agent had learned that *staying away from enemies* correlates with winning — the smoking-gun evidence of a "passive survival" strategy. The final agent from this run became the seed checkpoint for every subsequent self-play run.

### attention — 101 rollouts, ~2.5M steps

Gradient attribution on the value head, with the top-8 features crystallized every 50K steps. Trained against random. Final ~90% vs random, 0% vs heuristic. The discovered features were qualitatively different from contrastive's: **velocity-dominated** (`enemy3.speed -1`, `ally2.speed -1`, `enemy1.velocity_x +1`) rather than positional (from `runs/attention/train.log`). Both methods nonetheless converged to a "speed control" strategy with negative weight on own-speed — an independent rediscovery of the physics insight that high speed → falling off the edge. By rollout 40 the top-8 set had stabilized and barely changed for the remaining 60 rollouts.

### llm_architect — 3 iterations, fallback mode (no real LLM calls)

This run **never actually called Claude.** All three iterations ran with `FallbackRewardGenerator` returning hardcoded templates 0/1/2 (from `runs/llm_architect/train.log` lines 6, 16–17, 27–28 and the per-iteration `iteration_*/full_record.json` files). The "evolved" reward in iteration 2 is just `-dist_from_center*0.2 + enemy1.dist_to_edge*0.3 + edge_penalty*0.2` — the literal template (from `runs/llm_architect/iteration_002/reward_function.py`). Win rates from the larger ground-truth eval were 45.5% / 45.0% / 42.3% — the "evolution" hurt slightly. The quick-eval rates (0%, 58%, 53%) bounce around because they sample only ~50 episodes. `STATUS.md`'s "fallback + API mode" wording obscures the fact that no real LLM call was made; treat all `runs/llm_architect/` win-rate numbers as measurements of the templates, not of Claude.

### llm_cli — 3 iterations, real Claude CLI calls

Same architecture as `llm_architect` but actually invoking `claude --print` with model `claude-sonnet-4-20250514` (from `runs/llm_cli_train.log`). Each LLM call took 35–60s. **Win rates monotonically improved: 46.7% → 77.3% → 82.4%** over 200K-step training phases each, with reward-function code growing 31 → 49 → 77 lines (from `runs/llm_cli/summary.json`). Iteration 2's reward (in `runs/llm_cli/iteration_002/reward_function.py`) is a 12-component multi-objective function that includes ally coordination and an "optimal ally distance ≈ 0.3" term — clearly LLM-derived rather than templated. This is the strongest single piece of evidence that real LLM-in-the-loop reasoning beats hardcoded templates on this task.

### llm_cli_v2 — the famous "0% → 85%" recovery

Same setup as `llm_cli`, but with a larger eval (200 episodes) and 32 envs on CPU. Win rates: 46.4% → **0.0%** → 84.8% (from `runs/llm_cli_v2/summary.json`). Iteration 1's reward function (28 lines, in `runs/llm_cli_v2/iteration_001/reward_function.py`) over-emphasized `survival_reward = ego_alive * 3.0` and `safety_reward = ego_dist_to_edge * 4.0` plus a `duration_bonus = timestep * 0.2`. The agent then learned to **immediately suicide off the edge** to terminate episodes (average episode length collapsed from 4.5 → 1.4 timesteps). Iteration 2's reward (50 lines) saw the 0% catastrophe in the prompt and self-corrected: it reduced `duration_bonus` from 0.2 to 0.1, added an explicit `center_reward = (1.0 - ego_dist_from_center) * 1.2`, and introduced `enemy_spacing` terms — recovering to 84.8%. **This is the "Claude self-corrects from 0% to 85% from stats alone" finding**, and it is real and reproducible from the artifacts.

## PPO vs MAPPO Baselines

Both ran approximately 3M steps with curriculum learning (random → mixed) and a 30-game evaluation every ~196K steps (from `logs/ppo_overnight_stdout.log` and `logs/mappo_overnight_stdout.log`).

| Algo  | Best vs Random       | Best vs Heuristic                  | Total steps          |
|-------|----------------------|------------------------------------|----------------------|
| PPO   | 100% (at 2.6M)       | **7%** (at 2.6M)                   | 3.05M                |
| MAPPO | 100% (at 1.3M)       | **3%** (at 1.3M, lost it after)    | ~1.7M (interrupted)  |

PPO's curriculum advanced random → mixed at 80% vs random (step 589K). It then plateaued for roughly 1M steps before a notable iter-13 spike to 100% vs random and 7% vs heuristic. MAPPO triggered an entropy-coefficient intervention (0.01 → 0.05) at step 1.4M after stalling at 97% vs random / 0% vs heuristic — entropy spiked to 4.9 but did not help. PPO marginally edges out MAPPO on this game in this regime; neither breaks the heuristic ceiling. Both were trained with sparse +1 / -1 rewards only.

## Key Insights (Positive Findings)

1. **Statistical reward discovery actually works.** Contrastive Cohen's d at iter 0 found `enemy.dist_to_edge` and `ego.dist_to_edge` with the correct sign from 3470 random-walk trajectories — no game knowledge required (from `runs/contrastive/discovery_log.json` iter 0).
2. **Self-play produces a real ELO arms race.** v4 climbed 1000 → 1443 across 5.9M steps, sustained 100% vs random throughout, and held 89–100% vs the pool through iter 30. Pool eviction kicked in at iter 20 (size cap 20).
3. **LLM-in-loop reward iteration recovers from catastrophic failures.** `llm_cli_v2` went 46% → 0% → 85% with the LLM seeing its own failure mode in the stats prompt and correcting (reducing duration bonus, adding centering signal).
4. **Two reward-discovery methods cross-validate the physics.** Contrastive (positional) and attention (velocity) independently learned that high speed → falling off — physical insight derived from sparse signal alone.
5. **Adaptive random ratio works.** A 50% → 5% schedule over 30 iters did not cause regression on random eval (held at 100% from iter 5 onward) while ELO kept climbing.

## Negative Findings Worth Preserving

1. **"Train on random, get 0% vs heuristic"** — survives across PPO, MAPPO, contrastive, attention, and `llm_architect`. The opponent distribution dictates the strategy class learned (passive vs aggressive). Reward discovery finds *what* matters; opponents determine *how* to exploit it.
2. **Training on random *actively degrades* pool skill.** v1 iter 2 (96% → 68.5%) and v3 iter 2 (96% → 68.5%) both show pool win-rate drops after a random-only iteration mid-self-play (from `runs/self_play/training_log.csv` line 4 and `runs/self_play_v3/training_log.csv` line 4).
3. **Contrastive discovery breaks under dominance.** When win rate hits 100%, Cohen's d is undefined, so 0 features are discovered and the reward shaping vanishes (contrastive iter 2; v4 iter 5 also found nothing).
4. **Reward shaping can cause suicide policies.** `llm_cli_v2` iter 1 (0% win, 1.4 avg episode length) — over-weighting survival + duration bonus made the agent jump off so episodes ended faster. PPO will exploit any badly-scaled reward.
5. **Pool size 20 with eviction created an adversarial niche.** v4's `pool_step_3932160` was never beaten cleanly across 5 evaluations from iter 20 onward despite the learner gaining 100+ ELO — the pool contains hard counter-policies the learner cannot solve.
6. **The heuristic gap does not close from ELO alone.** PPO baseline reached only 7% vs heuristic at its best 100%-vs-random checkpoint; v4 reached learner-ELO 1443 but never measured heuristic in-loop at all. ELO inside the pool is not transitive to a hand-crafted strategy.
7. **`llm_architect` was a fallback shell.** The run was real (3 iterations, ~3 hours wall time, full logs/checkpoints/metrics), but the LLM never executed — every "discovery" is a hardcoded template. Do not cite its win-rate numbers (45.5%, 45.0%, 42.3%) as evidence about Claude.

## Anomalies and Loose Ends

- `attention_discovery_log.json` at the repo root is `[]` (2 bytes) — orphan.
- `training_log.csv` at the repo root has only 2 rollout entries — orphan from an earlier abandoned runner.
- `runs/self_play/train.log` and `runs/self_play_v2/train.log` have interleaved/duplicated lines indicating concurrent writers; treat as corrupted.
- `STATUS.md` and `STATUS2.md` disagree about v4: `STATUS.md` claims iter 16 / ELO 1355, `STATUS2.md` claims iter 9 / ELO 1237. Neither reflects the actual completion (iter 30 / ELO 1443).
- v4 iteration 24's wall time of 27,837s is an artifact (machine sleep), not a training stall.
- `runs/self_play_v4/pool_index.json`'s ELO column appears to mix learner-ELO and pool-member-ELO; do not trust those numbers without a re-evaluation pass.

---

This `RESEARCH_NOTES.md` supersedes both the prior `STATUS.md` and `STATUS2.md`, which were stale (and partially contradictory) snapshots of in-flight runs. Once the audit cleanup deletes the legacy `runs/` directories, the citations above will be the only remaining trail back to the underlying logs — read them in conjunction with the surviving `docs/research/reward_discovery_notes.md` and the source code in `src/knockout/` for context. Future work that revisits any of these claims should re-run the relevant experiment with the bookkeeping fixes noted above (in-loop heuristic eval restored to self-play, pool ELO tracked separately from learner ELO, contrastive guarded against zero-loss iterations).
