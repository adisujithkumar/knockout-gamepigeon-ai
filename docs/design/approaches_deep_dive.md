Now I have a complete picture of the design document and the existing codebase. Let me provide the comprehensive deep-dive.

---

# Deep-Dive: Six Zero-Knowledge Reward Discovery Approaches

Below is an exhaustive analysis of each approach from the design document. All are grounded in the actual knockout-v2 codebase: 89-dim observations, continuous `[angle_degrees, power]` actions, the existing `ActorCritic(89 -> 128 -> 64 -> 2)` network, and the `TensorVecEnv` GPU backend.

---

## Approach A: Feature Attention Discovery

### Detailed Mechanics

**Pseudocode:**

```
# Phase 1: Bootstrap
V = ValueNetwork(89 -> 128 -> 64 -> 1)  # Critic from ActorCritic
for rollout in 1..N_bootstrap:
    collect_rollout(env, policy, reward=terminal_only)
    ppo_update(policy, V)

# Phase 2+3: Attribution loop (repeats every K=50 rollouts)
for outer_iter in 1..N_outer:
    # Collect a buffer of ~6400 states from recent rollouts
    state_buffer = last_rollout_buffer.observations  # shape (128*64*3, 89)

    # Compute gradient attribution for each of 89 features
    for feature_i in 0..88:
        obs_tensor.requires_grad_(True)
        values = V(obs_tensor)
        values.sum().backward()
        attribution[i] = obs_tensor.grad[:, feature_i].abs().mean()

    # Crystallize: top-K features become reward components
    top_k = argsort(attribution)[-8:]  # take top 8
    for idx in top_k:
        sign = sign_of(mean_gradient_for_feature[idx])
        weight = attribution[idx] / sum(attribution[top_k]) * 0.5
        components.append(CrystallizedReward(idx, sign * weight))

    # Phase 4: Train with shaped reward
    for rollout in 1..K:
        shaped_r = terminal + sum(c.weight * obs[c.idx] for c in components)
        collect_rollout(env, policy, reward=shaped_r)
        ppo_update(policy, V)

    # Phase 5: Decay shaping weights by 0.95 each outer iteration
    for c in components:
        c.weight *= 0.95
```

**Iteration 1 (rollout 1-50).** The agent is random. V(s) outputs near-zero for everything. The only gradient signal comes from the few episodes that terminate -- states where a penguin was near the edge just before dying get slightly negative V, states in the center get slightly positive V. Attribution is noisy, dominated by `ego.dist_to_edge` (feature 5) and `ego.alive` (feature 8) because these have the clearest correlation with terminal reward. The agent "understands" nothing -- it is just noticing that position matters.

**Iteration 10 (rollout 500).** V(s) has learned a crude value landscape: center = safe = positive value, edge = danger = negative value. Gradient attribution now clearly highlights `ego.dist_to_edge` (feature 5), `ego.dist_from_center` (feature 4), and `ego.speed` (feature 6). The crystallized reward says "stay near center, move slowly." The agent has learned survival but not aggression. Its "understanding" is: "the middle is good, the edge is bad."

**Iteration 100 (rollout 5000).** The agent survives well. V(s) now has enough signal from games where the agent won by outlasting enemies. The gradient shifts: `enemy1.dist_to_edge` (feature 47) starts showing high attribution because games where enemies are near the edge tend to have higher V. The crystallized reward adds "enemies near edge is good." The agent's understanding: "survive and position myself so enemies are near edges."

### Concrete Example Walkthrough

Round 1: `penguin_0` at (-60, -10), `penguin_3` at (60, -10).

**What the agent sees (selected features from the 89-dim obs):**
- obs[0] = -0.60 (ego pos_x, normalized by arena radius 100)
- obs[1] = -0.10 (ego pos_y)
- obs[4] = 0.607 (ego dist_from_center = sqrt(3600+100)/100 = 0.607)
- obs[5] = 0.393 (ego dist_to_edge = 1.0 - 0.607)
- obs[42] = 0.60 (enemy1 pos_x)
- obs[47] = 0.393 (enemy1 dist_to_edge)
- obs[55] = 1.204 (enemy1 dist_to_ego = sqrt((120)^2)/100, capped at 1.0)

**At iteration 1 (untrained):** V(s) returns ~0.01. The policy outputs a random angle and power. Suppose it outputs [270, 300] -- move down with moderate force. Terminal reward is 0 (round continues). The agent gets r = 0 + 0.15 * 0.393 (survival shaping from dist_to_edge) = 0.059. This is tiny but positive -- slightly better than zero for the optimizer.

**At iteration 50 (survival learned):** V(s) returns ~0.25 (moderate -- it knows it's alive but not in a great position). The crystallized reward now includes -0.10 * obs[47] (enemy edge distance). The agent observes enemy1's dist_to_edge is only 0.393 -- enemy is also fairly far from center. The negative weight means the agent gets rewarded when this value is LOW (enemy near edge). But 0.393 is moderate, so the shaping reward is -0.10 * 0.393 = -0.039. Net shaped reward is small. The policy has not yet learned to approach enemies.

**At iteration 100 (aggression discovered):** The crystallized reward now includes a term for `enemy1.dist_to_ego` (feature 55) with a negative sign: reward for being CLOSE to enemies. The agent at (-60, -10) sees enemy at (60, -10) -- very far away (dist_to_ego near 1.0). The shaping reward penalizes this distance. The policy has learned to move toward the nearest enemy, so it outputs something like [0, 400] (move right toward enemy). Over the round, closing distance to the enemy generates positive shaping reward. When it reaches the enemy and applies an impulse that pushes the enemy toward the edge, `enemy1.dist_to_edge` drops, generating further reward.

### What Gets Discovered and When

- **1M steps (~120 rollouts):** Survival -- stay away from edge, don't move too fast. Win rate ~40% (above random's 33% because it doesn't suicide).
- **5M steps (~600 rollouts):** Passive aggression -- the agent positions itself centrally and enemies sometimes bounce off it toward the edge. `enemy.dist_to_edge` attribution appears. Win rate ~50%.
- **10M steps (~1200 rollouts):** Active targeting -- the agent approaches the nearest enemy and pushes toward the edge. `enemy.dist_to_ego` and `relative_velocity` features gain attribution. Win rate ~60%.
- **30M steps (~3600 rollouts):** Tactical play -- target the enemy closest to the edge first. Feature interactions between `enemy.dist_to_edge` and `enemy.dist_to_ego` become important. Win rate ~70%.

**The "aha moment"** for edge-pushing happens around rollout 300-500 (2.5-4M steps). The value network first registers that states where enemies are near the edge have higher expected return. This shows up in the attribution log as `enemy1.dist_to_edge` jumping from noise-level attribution to top-5.

**Credit assignment:** This approach partially solves credit assignment through the value function. V(s) bootstraps over time steps via GAE (lambda=0.95, gamma=0.99), propagating the terminal +1/-1 backward through ~30-50 steps per episode. The shaped reward components give per-step signal that dramatically shortens the credit chain -- instead of "I won 30 steps later," the agent gets "enemy moved closer to edge THIS step." However, very long-term strategies (e.g., "set up a teammate to make a play 15 rounds from now") are still difficult.

### Architecture Specifics

- **Policy network:** 89 -> 128 (ReLU) -> 64 (ReLU) -> 2 (actor mean) + 1 (critic). Parameters: 89*128 + 128 + 128*64 + 64 + 64*2 + 2 + 64*1 + 1 = 11,392 + 8,256 + 130 + 65 = **19,843 parameters** (~80KB).
- **Attribution computation:** One backward pass per feature per attribution step. 89 features * 6400 states = 89 backward passes on a (6400, 89) tensor. On GPU, this takes ~2 seconds per attribution cycle. Negligible overhead versus training.
- **Memory footprint:** The rollout buffer for 128 steps * 64 envs * 3 agents stores ~24,576 transitions. At 89 floats per obs + 2 action + 1 reward + 1 value + 1 logprob = 94 floats * 4 bytes = 376 bytes per transition. Total buffer: ~9.2 MB. Model: ~80KB. Total: **under 10MB**.
- **GPU utilization:** Compute-bound during PPO updates (matrix multiplications on the policy network). The attribution step is a brief spike of backward passes. The bottleneck is environment stepping, not gradient computation.

### What Could Go Wrong

1. **Noisy attribution in early training.** With a randomly initialized V(s), gradient attribution is essentially random. If the crystallization happens too early (K too small), the shaped reward is nonsense and can push the agent in harmful directions. **Diagnosis:** Check if the attribution scores are stable across successive batches. If they flip sign every iteration, K needs to be larger.

2. **Reward hacking on shaped components.** The agent could learn to maximize `ego.dist_to_edge` by standing completely still in the center, never engaging enemies. This "wins" the shaped reward but loses games when the arena shrinks. **Diagnosis:** Win rate stalls while shaped reward increases. **Fix:** Ensure terminal reward weight stays at 1.0 and decay the shaping components.

3. **Feature correlation vs. causation.** `ego.alive` (feature 8) will always have the highest attribution because dead agents always lose. But rewarding "be alive" adds no useful gradient -- the agent cannot choose not to die; it can only choose actions that reduce the probability. **Fix:** Exclude binary features (alive/dead) from crystallization; focus on continuous features that the agent can influence smoothly.

4. **Attribution collapses to a single feature.** If `dist_to_edge` dominates, the agent ignores all other strategic dimensions. **Diagnosis:** Inspect the attribution vector -- if one feature has 10x the score of the next, diversity is too low. **Fix:** Cap individual feature weights or use top-K with minimum diversity.

### What Makes It Novel for Research

**Unique data generated:** A timestamped log of `(rollout_number, feature_name, attribution_score, crystallized_weight)` across the full training run. This is a "feature importance timeline" that no other approach produces. It shows the agent's attention shifting from survival to aggression to tactics, analogous to how a human player's understanding evolves.

**Foundation model connection:** The narrative log ("at rollout 50, the agent discovered dist_to_edge matters") is directly convertible to natural language training data. A foundation model trained on many such logs across different games could learn to predict "what an RL agent should pay attention to next" given its current skill level -- a form of automated curriculum design.

**Paper angle:** "Interpretable Reward Discovery via Value Network Attribution" -- show that the order of feature discovery in a multi-agent physics game matches the pedagogical ordering a human coach would use, then demonstrate that crystallizing these features as shaping rewards accelerates training 3-5x over pure sparse reward.

---

## Approach B: LLM Reward Architect

### Detailed Mechanics

**Pseudocode:**

```
reward_code = "def reward(obs, prev_obs, terminal, info): return terminal"
history = []

for iteration in 1..50:
    # PLAY: Train PPO for 200 rollouts with current reward
    agent = PPOTrainer(reward_fn=compile(reward_code))
    for rollout in 1..200:
        collect_rollout(env, agent)
        ppo_update(agent)

    # COLLECT STATS
    stats = {
        "win_rate": evaluate(agent, n_games=200),
        "avg_game_length": mean(episode_lengths),
        "survival_rate": mean(ego_survived_per_game),
        "knockouts_per_game": mean(enemies_knocked_out),
        "feature_analysis": {
            # Per-feature means for winning vs losing games
            "ego_dist_to_edge": {"win_mean": 0.52, "loss_mean": 0.31},
            "enemy1_dist_to_edge": {"win_mean": 0.22, "loss_mean": 0.45},
            ...
        },
        "action_distribution": {"angle_mean": 180, "power_mean": 350},
        "sample_trajectory": [obs_0, obs_1, ..., obs_T]  # one game
    }

    # REFLECT: Ask LLM to improve reward function
    prompt = format_prompt(reward_code, stats, obs_layout)
    response = llm.generate(prompt)  # Claude or GPT-4
    new_code = extract_python(response)
    reasoning = extract_reasoning(response)

    # VALIDATE
    errors = validate_reward_fn(new_code, n_samples=1000)
    if errors:
        fix_prompt = f"Your code had errors: {errors}. Fix it."
        response = llm.generate(fix_prompt)
        new_code = extract_python(response)

    # UPDATE
    history.append(RewardProgram(reward_code, stats, reasoning))
    reward_code = new_code
```

**Iteration 1.** The LLM receives: "Win rate 33%, avg game length 28 rounds, survival rate 48%. Feature analysis shows no significant differences between wins and losses (agent is random)." The LLM, having general knowledge about games, likely writes something like: "Add a small reward for staying alive: `return terminal + 0.01 * obs[8]`." This is a reasonable first guess. But since obs[8] is binary (alive=1/dead=0), it's not very useful as a shaping signal. The LLM is fumbling -- it has game intuition but doesn't know the physics.

**Iteration 10.** After 2000 rollouts of training, stats now show: "Win rate 45%, survival rate improves to 62%, but knockouts per game is only 0.3." The LLM sees the feature analysis: "ego_dist_to_edge is 0.52 in wins vs 0.31 in losses." It writes: `return terminal + 0.1 * obs[5] - 0.05 * obs[4]` (reward for high dist_to_edge, penalize high dist_from_center). The LLM is starting to build a reasonable reward, guided by the statistics the system provides.

**Iteration 30.** Win rate 58%. The LLM now sees: "enemy_dist_to_edge averages 0.25 in wins vs 0.40 in losses. Knockouts per game rose to 1.2." It writes a multi-component reward: enemy proximity, enemy edge distance, closing speed. The reasoning trace says: "The agent needs to actively push enemies off. I'll add a reward for reducing enemy distance-to-edge and for being close to enemies." This is the LLM's "aha moment" -- it has inferred the game mechanic from the statistics.

**Iteration 50.** Win rate 72%. The LLM starts writing conditional rewards: "if the nearest enemy is within 20 units and their dist_to_edge is below 0.3, reward heavily for pushing toward them." It has essentially discovered targeting strategy from statistics alone.

### Concrete Example Walkthrough

Same scenario: `penguin_0` at (-60, -10), `penguin_3` at (60, -10).

**At iteration 5, the LLM's reward function is:**
```python
def reward(obs, prev_obs, terminal, info):
    r = terminal
    r += 0.05 * obs[5]   # stay away from edge
    r += 0.02 * obs[8]   # stay alive bonus
    return r
```

The agent at (-60, -10) has dist_to_edge = 0.393. Shaped reward: 0 + 0.05 * 0.393 + 0.02 * 1.0 = 0.0397. This is a gentle pull toward the center. The agent learns to drift centerward but has no reason to engage the enemy at (60, -10).

**At iteration 20, the LLM has iterated to:**
```python
def reward(obs, prev_obs, terminal, info):
    r = terminal
    r += 0.05 * obs[5]  # stay safe
    # Reward when enemies are near edge
    for enemy_edge_idx in [47, 61, 75]:
        if obs[enemy_edge_idx - 2]:  # enemy alive (idx-2 = alive flag)
            r -= 0.03 * obs[enemy_edge_idx]  # lower edge dist = more reward
    # Reward closing distance to nearest enemy
    if prev_obs is not None:
        r += 0.02 * (prev_obs[55] - obs[55])  # reward decreasing enemy distance
    return r
```

Now the agent at (-60, -10) sees enemy at (60, -10) with dist_to_ego ~1.0. The delta-based closing reward incentivizes moving toward the enemy. Over several rounds, the agent approaches, and once close enough, impulses push the enemy toward the edge, triggering the enemy-edge-distance component.

### What Gets Discovered and When

- **Iterations 1-5 (0-1M steps):** Basic survival. The LLM adds edge avoidance. Win rate ~40%.
- **Iterations 5-15 (1-3M steps):** Edge awareness refined. LLM discovers from statistics that speed matters (fast agents fly off). Adds speed penalty. Win rate ~50%.
- **Iterations 15-25 (3-5M steps):** Aggression discovered. LLM sees the enemy-edge-distance correlation in the stats and adds offensive components. Win rate ~60%.
- **Iterations 25-40 (5-8M steps):** Targeting and timing. LLM writes conditional logic: target weakened enemies, timing around arena shrinkage. Win rate ~68%.
- **Iterations 40-50 (8-10M steps):** Diminishing returns. The LLM tries complex multi-component rewards but starts to cause reward hacking. Some iterations regress. Win rate oscillates 68-73%.

**Credit assignment:** The LLM handles credit assignment in a fundamentally different way than RL -- it reasons about it in natural language. When the stats show "games with more early knockouts have higher win rates," the LLM infers that early aggression is important and writes rewards for it. This is a form of macro-level credit assignment that RL struggles with: "play aggressively in the first 10 rounds" is a long-horizon strategy that's hard to learn from per-step rewards but easy for the LLM to specify in code.

### Architecture Specifics

- **Policy network:** Same as Approach A: 89 -> 128 -> 64 -> 2/1. ~19,843 params, ~80KB.
- **LLM costs:** ~50 iterations * 1 API call each. With Claude, ~2K input tokens + 500 output tokens per call. Total: ~125K tokens. At current prices, ~$1-3 total. Negligible.
- **Memory:** The reward function is Python source code, typically 10-30 lines. The stats collector stores per-game aggregates, not raw trajectories. 200 games * ~20 statistics = 4000 numbers. Under 1MB total overhead.
- **GPU utilization:** Identical to standard PPO training. The LLM call is off-GPU (API call) and takes 2-5 seconds, versus minutes of PPO training. The bottleneck is PPO training, not LLM inference.
- **Total compute:** 50 iterations * 200 rollouts * 128 steps * 64 envs * 3 agents = ~245M timesteps. ~2-3 hours on a single GPU.

### What Could Go Wrong

1. **The LLM writes reward functions that cause reward hacking.** For example, it adds `r += 0.5 * obs[88]` (timestep), inadvertently rewarding the agent for surviving longer without caring about winning. The agent stalls and never engages. **Diagnosis:** Win rate drops while shaped reward increases. The validation suite should catch "average reward per step should not exceed X." **Fix:** Add a validation rule: reward must correlate positively with winning.

2. **The LLM oscillates between conflicting strategies.** Iteration 20 adds "be aggressive," iteration 21 sees the agent dying from aggression and removes it, iteration 22 adds it back. **Diagnosis:** Reward function code changes significantly every iteration. **Fix:** Keep an archive and let the LLM see the last 3-5 versions plus their results, enabling it to reason about trends rather than reacting to the latest snapshot.

3. **Compiled code is unsafe.** The LLM could write `import os; os.system(...)`. **Fix:** Execute the reward function in a sandboxed namespace with only `np` and `math` available. The `exec(source, {"np": np, "math": math})` pattern in the design prevents arbitrary imports.

4. **The LLM has prior knowledge about knockout-style games.** This weakens the "zero-knowledge" claim. **Fix:** Strip game-specific terminology from the prompt. Call it "Game X" with "entities" and "positions." The LLM still has general reasoning ability, but no game-specific priors.

5. **Prompt sensitivity.** Small changes in how statistics are formatted can dramatically change the LLM's output. **Diagnosis:** Run the same iteration 3 times with slightly different formatting; if outputs diverge wildly, the prompt is too fragile. **Fix:** Structure statistics as a consistent JSON/table format.

### What Makes It Novel for Research

**Unique data generated:** Each iteration produces a `(game_statistics, llm_reasoning_trace, python_reward_code, resulting_win_rate)` tuple. This is the richest data product of all six approaches -- it directly captures the reasoning process of reward design in natural language.

**Foundation model connection:** This is the most direct pipeline to the project's goal. The reasoning traces teach an LLM: "When you see these game statistics, here's how to reason about reward design, and here's what code to write." A foundation model fine-tuned on 50+ games' worth of these traces could generalize to designing rewards for novel games from statistics alone.

**Paper angle:** "LLM-Guided Reward Discovery: Program Synthesis for Reinforcement Learning" -- demonstrate that an LLM with no game-specific training can iteratively design reward functions that match or exceed hand-engineered baselines. Compare the LLM's discovery order to Approach A's attribution order. Analyze the failure modes (reward hacking, oscillation) as data about LLM reasoning limitations.

---

## Approach C: Evolutionary Reward Search

### Detailed Mechanics

**Pseudocode:**

```
# Define ~20 reward features extracted from 89-dim obs
features = [ego_dist_edge, ego_alive, ego_speed, enemy1_dist_edge, ...]

# Initialize population
population = []
for i in 1..20:
    if i <= 10:
        w = zeros(20)  # pure terminal reward
    else:
        w = normal(0, 0.1, size=20)  # small random weights
    population.append(EvolvableReward(weights=w, terminal_weight=1.0))

# Evolution loop
for generation in 1..30:
    # EVALUATE each individual
    for individual in population:
        agent = fresh_PPOTrainer()
        for rollout in 1..50:
            r = individual.terminal_weight * terminal + dot(individual.weights, features(obs))
            collect_rollout(env, agent, reward=r)
            ppo_update(agent)
        individual.fitness = evaluate(agent, n_games=100)  # win rate

    # SELECTION (tournament, size 3)
    next_gen = []
    for _ in 1..20:
        candidates = random_sample(population, 3)
        winner = max(candidates, key=lambda x: x.fitness)
        next_gen.append(copy(winner))

    # MUTATION
    for ind in next_gen:
        if random() < 0.3:
            idx = random_int(0, 19)
            ind.weights[idx] += normal(0, 0.05)
        if random() < 0.05:
            idx = random_int(0, 19)
            ind.weights[idx] = 0.0  # feature dropout
        if random() < 0.05:
            # Discover a new feature (turn on a zero weight)
            zero_idxs = where(ind.weights == 0)
            if len(zero_idxs) > 0:
                idx = random_choice(zero_idxs)
                ind.weights[idx] = normal(0, 0.1)

    # CROSSOVER
    for i in range(0, 20, 2):
        if random() < 0.3:
            mask = random_bool(20)
            next_gen[i].weights[mask], next_gen[i+1].weights[mask] = \
                next_gen[i+1].weights[mask], next_gen[i].weights[mask]

    # ELITISM: keep top 2
    elite = sorted(population, key=lambda x: x.fitness)[-2:]
    next_gen[-2:] = elite

    population = next_gen
```

**Generation 1.** All individuals have near-random weights. Half are pure terminal (zero weights). The pure-terminal individuals train for 50 rollouts and achieve ~35% win rate (barely above random). The random-weight individuals are chaotic -- some accidentally discover survival by rewarding `ego_dist_to_edge` and achieve 38-42%. Others penalize useful features and achieve 28%. Selection pressure keeps the 38-42% individuals.

**Generation 10.** The population has converged toward positive weight on `ego_dist_to_edge`, small negative weight on `ego_speed`, and near-zero weights on most enemy features. Average fitness ~45%. The agent has "discovered" survival but not aggression. The evolutionary search is slow because each fitness evaluation requires training a fresh PPO agent for 50 rollouts.

**Generation 30.** The best individuals have weights like: `ego_dist_to_edge = +0.15`, `enemy1_dist_to_edge = -0.08`, `enemy2_dist_to_edge = -0.06`, `nearest_enemy_dist = -0.04`, `team_alive_advantage = +0.12`. Fitness ~65%. The phylogenetic tree shows that the `enemy_dist_to_edge` feature was first introduced around generation 12 by a mutation in a survival-focused individual, then spread through the population via crossover.

### Concrete Example Walkthrough

Same scenario: `penguin_0` at (-60, -10), `penguin_3` at (60, -10).

**Generation 1, Individual #7 (random weights).** Weights: `ego_dist_edge=+0.12, ego_speed=-0.03, enemy1_dist_edge=+0.02, enemy1_dist_to_ego=-0.01`.

The agent computes reward features:
- ego_dist_to_edge = 0.393
- ego_speed = 0.0 (start of round)
- enemy1_dist_to_edge = 0.393
- enemy1_dist_to_ego = 1.0 (far away)

Shaped reward: 0.12 * 0.393 + (-0.03) * 0.0 + 0.02 * 0.393 + (-0.01) * 1.0 = 0.047 + 0 + 0.008 - 0.01 = 0.045.

This individual slightly rewards staying alive and slightly penalizes being far from enemies -- a reasonable starting point, but the weights are too small to significantly guide learning in 50 rollouts.

**Generation 15, the lineage leader (descendant of #7).** Weights evolved to: `ego_dist_edge=+0.18, ego_speed=-0.08, enemy1_dist_edge=-0.10, enemy1_dist_to_ego=-0.06`.

Same observation now gives: 0.18 * 0.393 + (-0.08) * 0.0 + (-0.10) * 0.393 + (-0.06) * 1.0 = 0.071 - 0.039 - 0.06 = -0.028.

Interesting: the agent is penalized in this state because enemy1 is far away (dist_to_ego = 1.0 contributes -0.06). This drives the agent to approach enemies. Once close enough to push them toward the edge, `enemy1_dist_edge` drops, and the reward becomes positive. This is a mechanically sound incentive structure, discovered entirely by evolution.

### What Gets Discovered and When

- **Generations 1-5 (0-5M total steps across pop):** Survival features dominate. `ego_dist_to_edge` is the first feature to consistently have positive weight across the population.
- **Generations 5-15 (5-15M steps):** `ego_speed` penalty emerges (don't overshoot), `team_alive` advantage appears. Average fitness reaches ~50%.
- **Generations 15-25 (15-25M steps):** Enemy-targeting features emerge. `enemy_dist_to_edge` with negative weight spreads through the population. Fitness ~60%.
- **Generations 25-30 (25-30M steps):** Feature interactions stabilize. The best individuals have 6-8 nonzero weights. Fitness plateaus at ~65%.

**The "aha moment"** is generation 12-15, when a mutation introduces negative weight on `enemy_dist_to_edge`. This individual's fitness jumps from ~50% to ~58%, causing it to dominate the next few generations via selection.

**Credit assignment:** Evolution sidesteps the per-step credit assignment problem entirely. It only cares about the final outcome: did the reward function produce a high win rate after training? This means it can discover reward components that are useful for learning even if they have complex temporal dynamics -- evolution does not need to propagate gradients through time.

### Architecture Specifics

- **Per-individual policy:** 89 -> 128 -> 64 -> 2/1. ~19,843 params. Created fresh for each fitness evaluation.
- **Reward function:** A weight vector of length ~20. ~80 bytes. Negligible.
- **Total compute:** 30 generations * 20 individuals * 50 rollouts * 128 steps * 64 envs * 3 agents = **~245M timesteps per generation, ~7.4B total**. This is the most compute-intensive approach by far. With a single GPU running TensorVecEnv at ~50K steps/sec, this takes ~41 hours. With parallelization across 4 GPUs (5 individuals per GPU), ~10 hours.
- **Memory:** 20 concurrent PPO agents if parallelized. Each needs ~10MB (model + buffer). Total ~200MB on GPU, well within capacity.
- **GPU utilization:** Embarrassingly parallel within a generation. Each individual's 50-rollout training is independent. Between generations, a brief CPU-bound selection/mutation step (<1 second). This approach is almost perfectly parallelizable across GPUs.

### What Could Go Wrong

1. **Premature convergence.** The population converges on a local optimum (e.g., pure survival) and cannot escape because mutations are too small to jump to aggression. **Diagnosis:** All individuals have similar weights, fitness plateau early. **Fix:** Increase mutation rate, add "random restart" individuals each generation, increase population size.

2. **Noisy fitness evaluation.** 50 rollouts is often not enough to reliably train a PPO agent, so fitness estimates are noisy. A bad individual might get lucky and propagate harmful genes. **Diagnosis:** Fitness variance within the same individual across evaluations. **Fix:** Increase evaluation rollouts (costs more compute) or use fitness averaging (evaluate each individual 3 times, take the median).

3. **Feature basis is too narrow.** The 20 predefined features miss an important reward component. **Diagnosis:** Best fitness plateaus below what you know is achievable (e.g., below what Approach A discovers). **Fix:** Add more features to the basis, or use a higher-dimensional basis with regularization.

4. **Compute budget is too expensive.** 7.4B timesteps is a lot. **Fix:** Reduce to P=10, E=30, G=20 for a cheaper pilot (900M timesteps, ~5 hours on one GPU). Or use a cheaper proxy: evaluate each individual's reward function by running it on pre-collected trajectories and measuring correlation with winning, rather than training a full PPO agent.

### What Makes It Novel for Research

**Unique data generated:** The phylogenetic tree of reward functions. You can trace every surviving feature back to its origin: which mutation introduced it, which crossover combined it with other features, which selection pressure kept it. This is analogous to a gene phylogeny in biology.

**Foundation model connection:** The (generation, weight_vector, fitness) data across 30 generations and 20 individuals (600 data points) can be used to train a model to predict "what reward function modifications will improve fitness." This is a form of meta-learning about reward design.

**Paper angle:** "Evolutionary Reward Discovery: A Phylogenetic Analysis of Learned Reward Functions" -- present the phylogenetic tree, show convergent evolution across multiple independent runs (does evolution always discover the same features in the same order?), compare to Approach A's attribution timeline.

---

## Approach D: Curiosity-Driven Discovery with Hindsight Reward Modeling

### Detailed Mechanics

**Pseudocode:**

```
# PHASE 1: Pure curiosity exploration
rnd = RNDModule(obs_dim=89, embed_dim=32)
policy = ActorCritic(89, 2)
for rollout in 1..500:
    for step in 1..128:
        obs = env.observe()
        action = policy.act(obs)
        next_obs, terminal = env.step(action)

        # Curiosity reward only -- no win/loss signal
        r_curiosity = rnd.intrinsic_reward(next_obs)
        store_transition(obs, action, r_curiosity, next_obs, terminal)

        # Train predictor to reduce error on visited states
        rnd.train_predictor(next_obs)

    ppo_update(policy, using=curiosity_rewards)

    # Also store full episodes with outcomes for Phase 2
    replay_buffer.store_episode(episode_obs, episode_actions, outcome=win_or_loss)

# PHASE 2: Build hindsight reward model
# Label every state in the buffer with eventual outcome
for episode in replay_buffer:
    for obs_t in episode:
        labels.append((obs_t, episode.outcome))

hindsight_model = HindsightRewardModel(89)
transition_model = TransitionRewardModel(89)

hindsight_model.train(states, outcomes, epochs=50)
transition_model.train(state_pairs, outcomes, epochs=50)

# PHASE 3: Blended training
for rollout in 1..1000:
    alpha = max(0.0, 0.8 - rollout * 0.0008)   # curiosity decays
    beta = min(0.7, 0.2 + rollout * 0.0005)     # hindsight grows
    gamma = 1.0                                   # terminal stays

    for step in 1..128:
        obs = env.observe()
        action = policy.act(obs)
        next_obs, terminal = env.step(action)

        r = alpha * rnd.intrinsic_reward(next_obs) \
          + beta * hindsight_model(next_obs) \
          + gamma * terminal

        rnd.train_predictor(next_obs)
        store_transition(obs, action, r, next_obs, terminal)

    ppo_update(policy, using=blended_rewards)

    # Retrain hindsight model every 100 rollouts
    if rollout % 100 == 0:
        hindsight_model.train(new_replay_data, epochs=20)
```

**Phase 1, rollout 1-50 (pure curiosity).** The agent is driven entirely by novelty. It has never seen most states, so everything is novel. Early on, RND prediction error is high everywhere, and the agent wanders randomly. Gradually, the predictor learns common states (center of arena, stationary), and their novelty drops. The agent starts seeking unusual states: high speeds, near edges, close to enemies. This is exploration without purpose -- the agent visits states it has never seen before, not states that are strategically useful.

**Phase 1, rollout 200-500.** The agent has explored most of the reachable state space. It has experienced collisions, edge proximity, knockouts (both getting knocked off and knocking enemies off), high-speed chases, and static standoffs. The curiosity reward is now low everywhere -- the predictor has learned to predict the target network for all common states. The replay buffer contains thousands of diverse episodes, some won, some lost.

**Phase 2 (after rollout 500).** The hindsight model trains on ~500 * 128 * 64 * 3 = ~12M state-outcome pairs. It learns that states with high `ego.dist_to_edge` and low `enemy.dist_to_edge` tend to have positive outcomes. The transition model learns that transitions where `enemy.dist_to_edge` decreases are associated with winning. Crucially, because curiosity forced exploration, the model has diverse training data -- it has seen states that a pure RL agent would never visit (e.g., all agents near the edge simultaneously, high-speed collisions).

**Phase 3, rollout 500-1000.** The blended reward initially gives 80% weight to curiosity and 20% to the hindsight model. The agent continues exploring but with a bias toward "good" states as identified by the hindsight model. Over 500 rollouts, the blend shifts to 70% hindsight, 0% curiosity. The agent transitions from explorer to strategist.

### Concrete Example Walkthrough

Same scenario: `penguin_0` at (-60, -10), `penguin_3` at (60, -10).

**Phase 1, rollout 50 (pure curiosity).** The agent at (-60, -10) has visited this general area many times. RND prediction error for obs is low (~0.02). But if the agent moves toward the enemy (a relatively rare event in random play), the resulting state -- being near an enemy at high speed -- is novel. RND error spikes to ~0.15. The agent gets intrinsic reward 0.15 for approaching the enemy. It does not "know" this is strategically useful -- it's just novel. But this data point goes into the replay buffer. When it collides with the enemy and both go flying near the edge, those states are highly novel (RND error ~0.5). The buffer now contains a trajectory: approach enemy -> collision -> both near edge -> one falls off.

**Phase 2, after 500 rollouts.** The hindsight model has seen this collision trajectory multiple times. In games where the enemy fell off after the collision, the outcome was "win." In games where the ego fell off, "loss." The model learns: `R_model(state_near_enemy_near_edge) > R_model(state_far_from_enemy_center)`. When probed with the starting state (ego at -60, -10), the model outputs ~0.1 (neutral -- could go either way). When probed with a state where ego is near enemy and enemy is near edge, it outputs ~0.5.

**Phase 3, rollout 700 (mostly hindsight-driven).** The agent at (-60, -10) evaluates: hindsight_model(current_state) = 0.1. If it moves toward the enemy, the predicted next-state has higher hindsight value. The agent approaches. When close enough to push the enemy toward the edge, the transition model gives high reward. The agent has learned the targeting strategy through the hindsight model, not through millions of steps of sparse reward.

### What Gets Discovered and When

- **Phase 1 (0-5M steps, curiosity only):** No strategic understanding. The agent is a novelty-seeking random walk. But it builds an incredibly diverse replay buffer covering the full state space. Win rate is ~33% (random).
- **Phase 2 transition (5M steps):** The hindsight model trains in minutes on GPU. Immediately identifies dist_to_edge and enemy_proximity as predictive of winning. No training steps needed -- this is supervised learning on existing data.
- **Phase 3 early (5-8M steps):** Agent rapidly improves because the hindsight model provides dense reward from step 1 of blended training. Win rate jumps from 33% to 50% in ~500 rollouts.
- **Phase 3 mid (8-15M steps):** Agent refines strategy. Hindsight model is retrained on newer (better) data. The model's predictions become more nuanced -- it starts distinguishing between "enemy near edge and I'm close enough to push" vs "enemy near edge but I'm too far away." Win rate ~65%.
- **Phase 3 late (15-20M steps):** Hindsight weight decays. Agent transitions to pure terminal reward. If the hindsight model was good, the policy retains the learned behavior. Win rate ~70%.

**The "aha moment"** is structurally different here. The agent never has a single "aha" -- the hindsight model discovers the game's structure all at once during Phase 2 training (supervised learning on the replay buffer). The policy then absorbs this knowledge over Phase 3. The discovery happens in the reward model, not the policy.

**Credit assignment:** Hindsight labeling completely sidesteps temporal credit assignment. Instead of asking "what action 8 steps ago caused this win?", the hindsight model asks "what states appeared in winning games?" This is a fundamentally different approach: it converts the temporal credit assignment problem into a supervised classification problem (win-state vs loss-state).

### Architecture Specifics

- **RND target network:** 89 -> 64 (ReLU) -> 32. Parameters: 89*64 + 64 + 64*32 + 32 = 5,696 + 2,080 = **7,776 params** (~31KB). Frozen.
- **RND predictor network:** 89 -> 64 (ReLU) -> 64 (ReLU) -> 32. Parameters: 89*64 + 64 + 64*64 + 64 + 64*32 + 32 = 5,696 + 4,160 + 2,080 = **11,936 params** (~48KB). Trainable.
- **Hindsight reward model:** 89 -> 64 (ReLU) -> 32 (ReLU) -> 1. Parameters: 89*64 + 64 + 64*32 + 32 + 32*1 + 1 = 5,696 + 2,080 + 33 = **7,809 params** (~31KB).
- **Transition reward model:** 178 -> 128 (ReLU) -> 64 (ReLU) -> 1. Parameters: 178*128 + 128 + 128*64 + 64 + 64*1 + 1 = 22,784 + 8,256 + 65 = **31,105 params** (~124KB).
- **Policy network:** Same 19,843 params as always.
- **Replay buffer for hindsight:** Stores full episodes. 500 rollouts * 128 steps * 64 envs * 3 agents = ~12M transitions. Each transition: 89 floats obs + 1 float outcome = 90 * 4 bytes = 360 bytes. Total: **~4.3GB**. This is the major memory cost. Can be reduced by subsampling (store every 4th step) to ~1GB.
- **GPU utilization:** Phase 1 is standard PPO (compute-bound on policy forward/backward). Phase 2 is supervised learning on a large dataset -- GPU-bound on the hindsight model training (small model, large dataset = data-loading-bound). Phase 3 adds RND forward pass and hindsight model forward pass per step -- minimal overhead (~5% slower than vanilla PPO).

### What Could Go Wrong

1. **Curiosity is undirected in a 6-agent game.** With 6 penguins and complex physics, the state space is enormous. 500 rollouts of curiosity may not cover enough of the strategically relevant states (e.g., situations where 3 agents converge on one enemy). **Diagnosis:** After Phase 1, check the replay buffer: does it contain knockout events? If not, curiosity never found them. **Fix:** Increase Phase 1 length, or seed the buffer with a few hand-played games.

2. **Hindsight model memorizes instead of generalizing.** With 12M training points, a 7K-parameter model should generalize. But if the state distribution is very concentrated (agent spends 90% of time in the center), the model may only learn "center = good" and not generalize to edge situations. **Diagnosis:** Evaluate model on held-out states from the tail of the distribution. **Fix:** Use weighted sampling (upweight rare states) or increase model capacity.

3. **Phase 3 blend schedule is wrong.** If curiosity decays too fast, the agent stops exploring before the hindsight model is good. If too slow, the agent wastes time on undirected exploration after the model is ready. **Diagnosis:** Plot curiosity coefficient vs win rate -- if win rate jumps when curiosity drops, curiosity was holding it back. **Fix:** Use an adaptive schedule: reduce curiosity when hindsight model loss drops below a threshold.

4. **Hindsight model becomes stale.** During Phase 3, the agent's behavior changes, making the hindsight model's training distribution outdated. A state that was "losing" under random play might be "winning" under the improved policy. **Diagnosis:** Hindsight model loss increases over time even as win rate improves. **Fix:** Retrain more frequently (every 50 rollouts instead of every 100).

### What Makes It Novel for Research

**Unique data generated:** The RND prediction error landscape over the state space is a "curiosity map" -- a topographic map of what the agent finds surprising. This is unlike any other approach's output. Overlaying the curiosity map with the hindsight value map shows where exploration discovers useful states.

**Foundation model connection:** The (curiosity_map, hindsight_reward_surface, policy) tuple is a rich representation of game understanding. The curiosity map shows "what is there to discover," the hindsight surface shows "what matters for winning," and the policy shows "what to do about it." A foundation model trained on these triplets across many games could learn to reason about novel games' structure from their curiosity maps.

**Paper angle:** "Curiosity as Scientific Exploration: Using Intrinsic Motivation to Bootstrap Reward Discovery" -- show that curiosity-driven exploration discovers strategically relevant states faster than random play, then demonstrate that hindsight labeling converts this exploration into effective reward shaping. Compare the curiosity map to a human's "interesting game situations."

---

## Approach E: Meta-Gradient Reward Learning

### Detailed Mechanics

**Pseudocode (using Evolution Strategies for the outer loop, since full backprop through PPO is impractical):**

```
# Initialize reward network
R_phi = MetaRewardNetwork(89 -> 32 (Tanh) -> 16 (Tanh) -> 1)
outer_optimizer = Adam(R_phi.parameters(), lr=1e-3)

for meta_step in 1..500:
    # INNER LOOP: Short PPO training with current reward
    policy = ActorCritic(89, 2)  # Fresh or continued
    inner_optimizer = Adam(policy.parameters(), lr=3e-4)

    for inner_step in 1..5:
        rollout = collect_rollout(env, policy)
        # Compute rewards from reward network
        for transition in rollout:
            transition.reward = R_phi(transition.obs).detach() + transition.terminal
        ppo_update(policy, inner_optimizer, rollout)

    # OUTER LOOP: Evaluate and update reward network
    # Using ES (Evolution Strategies) to approximate meta-gradient
    base_fitness = evaluate(policy, n_games=50)  # win rate

    # Perturb phi in N random directions
    perturbations = [randn_like(phi) for _ in range(20)]
    perturbed_fitnesses = []

    for eps in perturbations:
        # Perturb reward network
        R_perturbed = copy(R_phi)
        R_perturbed.params += 0.01 * eps

        # Train a short PPO with perturbed reward
        policy_copy = copy(policy)
        for inner_step in 1..5:
            rollout = collect_rollout(env, policy_copy)
            for t in rollout:
                t.reward = R_perturbed(t.obs).detach() + t.terminal
            ppo_update(policy_copy, rollout)

        perturbed_fitnesses.append(evaluate(policy_copy, n_games=50))

    # ES gradient estimate
    grad_phi = 0
    for eps, fitness in zip(perturbations, perturbed_fitnesses):
        grad_phi += (fitness - base_fitness) * eps
    grad_phi /= (20 * 0.01)

    # Update reward network
    outer_optimizer.step(-grad_phi)  # ascend fitness
```

**Meta-step 1.** R_phi is randomly initialized. It maps observations to random scalars (roughly uniform in [-0.5, 0.5] due to Tanh + small weights). The inner loop trains a policy using these random rewards plus terminal signal. In 5 inner steps, the policy barely learns anything. Evaluation: ~33% win rate (random). The ES perturbations also produce ~33% (all perturbations equally bad). The meta-gradient is pure noise. R_phi takes a random walk.

**Meta-step 50.** R_phi has drifted away from its initial random state. Some perturbation directions happen to improve the inner policy's win rate. Through the noisy ES signal, R_phi slowly discovers that outputting higher values for states with high `ego.dist_to_edge` and low `enemy.dist_to_edge` leads to slightly better inner policies. R_phi's function landscape begins to resemble: "center = good, edge = bad, enemies near edge = good."

**Meta-step 300.** R_phi has been optimized over hundreds of meta-steps. It now outputs a smooth reward landscape where: approaching enemies yields moderate positive reward, pushing enemies toward the edge yields high reward, being near the edge yourself yields negative reward. The inner policy, trained with these rewards, reaches ~60% win rate in just 5 PPO steps -- the reward function is so well-shaped that learning is rapid.

**Meta-step 500.** R_phi converges. Probing with synthetic states reveals a reward surface that closely resembles a human-designed reward: it's highest when the agent is centered, enemy is near the edge, and closing velocity is positive. Win rate: ~68%.

### Concrete Example Walkthrough

Same scenario: `penguin_0` at (-60, -10), `penguin_3` at (60, -10).

**Meta-step 1 (random R_phi).** R_phi(obs) produces 0.23 (essentially random given these inputs). The terminal reward is 0 (game continues). Total reward: 0.23. This number is meaningless -- it doesn't correlate with any useful strategy. The inner policy receives these random numbers as "rewards" and learns nothing useful.

**Meta-step 100 (partially trained R_phi).** R_phi has learned some structure. For the ego at (-60, -10), it outputs 0.15 (moderate -- the agent is alive, not near the edge, but not centered either). For a hypothetical state where the ego is at (0, 0) and the enemy is near the edge at (90, 0), it outputs 0.45. The difference (0.45 - 0.15 = 0.30) provides gradient for the inner policy to move from the current state toward the better state. The agent learns to approach center and seek enemy-near-edge configurations.

**Meta-step 300 (well-trained R_phi).** R_phi outputs:
- Current state (ego far from enemy): 0.12
- After approaching enemy: 0.28
- After pushing enemy toward edge: 0.52
- After enemy falls off: terminal +1.0

This creates a smooth reward gradient that guides the agent through the full sequence: approach -> push -> knockout. The inner policy can learn this sequence in 5 PPO steps because the reward is so dense and well-shaped.

### What Gets Discovered and When

- **Meta-steps 1-50 (~25M total steps across all perturbations):** Noise. R_phi wanders randomly. Win rate ~33%.
- **Meta-steps 50-150 (~75-175M steps):** R_phi discovers survival. The meta-gradient signal becomes consistent: perturbations that make R_phi reward center-states lead to higher win rates. Win rate ~45%.
- **Meta-steps 150-300 (~175-350M steps):** R_phi discovers aggression. Perturbations that make R_phi reward enemy-near-edge states lead to higher win rates. Win rate ~58%.
- **Meta-steps 300-500 (~350-500M steps):** Refinement. R_phi learns nuanced reward shaping -- timing, targeting, approach angles. Win rate ~68%.

**The "aha moment"** is diffuse -- there's no discrete event where "the agent discovers edges are dangerous." Instead, R_phi smoothly deforms from random noise into a structured reward landscape. If you plot the reward for `ego_at_center, enemy_at_edge` over meta-steps, it rises gradually from ~0 to ~0.5 between meta-steps 50 and 200.

**Credit assignment:** Meta-gradients handle credit assignment at two levels. The inner loop (5 PPO steps) handles per-step credit via GAE. The outer loop handles reward-design credit via ES: "did this change to R_phi produce better policies?" The 5-step inner loop is short, which limits the agent's ability to discover very long-horizon strategies. Increasing inner steps improves this but increases compute proportionally.

### Architecture Specifics

- **Reward network (R_phi):** 89 -> 32 (Tanh) -> 16 (Tanh) -> 1. Parameters: 89*32 + 32 + 32*16 + 16 + 16*1 + 1 = 2,848 + 528 + 17 = **3,393 params** (~14KB). Intentionally small to keep the meta-optimization tractable.
- **Policy network:** Standard 19,843 params.
- **Total compute per meta-step:** (1 base + 20 perturbations) * 5 inner steps * 1 rollout each = 105 rollouts. Plus 21 evaluations * 50 games each. Per meta-step: ~105 * 128 * 64 * 3 = ~2.6M training steps + ~21 * 50 * ~30 * 6 = ~189K eval steps. Total per meta-step: ~2.8M steps.
- **Total compute for 500 meta-steps:** ~1.4B steps. At 50K steps/sec on GPU, this is ~7.8 hours. Comparable to Approach C.
- **Memory:** Need to store 21 copies of the policy (base + 20 perturbations) simultaneously if parallelized. Each is ~80KB. Plus 21 rollout buffers at ~9MB each = ~189MB. Feasible on a single GPU.
- **GPU utilization:** The 21 inner training runs per meta-step are independent and parallelizable. With batched environments, you could run all 21 in parallel using a batched policy network (21 * 19843 params = ~416K params, still tiny). The bottleneck is the sheer number of environment steps, not GPU compute.

### What Could Go Wrong

1. **ES gradient is too noisy.** With only 20 perturbations and 50 evaluation games, the fitness estimates are noisy (win rate has inherent variance from stochastic games). The meta-gradient may point in random directions. **Diagnosis:** Track fitness variance across meta-steps. If the baseline fitness varies by more than +/-10% between meta-steps without R_phi changing, the signal-to-noise ratio is too low. **Fix:** Increase perturbations to 50-100, increase evaluation games to 200, or use antithetic sampling (pair each +eps with -eps).

2. **Inner loop is too short.** 5 PPO steps may not be enough to reveal whether a reward function is good. A reward function that requires 100 steps of training to show its value gets zero credit in a 5-step evaluation. **Diagnosis:** Train a known-good reward function for 5 steps and check if it outperforms terminal-only in 5 steps. If not, inner loop is too short. **Fix:** Increase inner steps to 20-50 (proportionally increases compute).

3. **Reward network overfits.** R_phi has 3,393 parameters learning from noisy fitness signals. It might memorize specific state-fitness associations rather than learning a smooth reward landscape. **Diagnosis:** Probe R_phi with synthetic states and check for jagged or nonsensical reward surfaces. **Fix:** Add weight decay to the outer optimizer, reduce network capacity, or add noise regularization.

4. **R_phi collapses to constant output.** If the meta-gradient pushes all outputs toward the same value, R_phi provides no shaping signal and training reverts to terminal-only. **Diagnosis:** Measure the variance of R_phi's output across a batch of states. If it drops below a threshold, R_phi has collapsed. **Fix:** Add a diversity penalty to the meta-loss: `L_meta += lambda * (1 / var(R_phi(states)))`.

5. **The approach is conceptually hard to debug.** There are two nested optimization loops, each with its own learning rate, each affecting the other. Diagnosing whether the problem is the inner loop (policy not learning), the outer loop (R_phi not improving), or the evaluation (too noisy) requires careful ablation. This is the main practical difficulty.

### What Makes It Novel for Research

**Unique data generated:** The trajectory of R_phi's parameters across 500 meta-steps is a "movie" of reward function learning. You can visualize R_phi's reward landscape at meta-step 1, 50, 100, 200, 500 and see it evolving from noise to structure. No other approach produces this.

**Foundation model connection:** The (R_phi_parameters, resulting_policy_win_rate) trajectory teaches how reward functions influence learning dynamics. A meta-meta-model could learn: "to teach an agent to play a physics game, the reward function should evolve from survival rewards to aggression rewards at this rate." This connects to the optimal reward shaping theory (Ng et al., 1999) -- we can empirically check whether R_phi converges to a potential-based form.

**Paper angle:** "Learning to Reward: Meta-Gradient Optimization of Neural Reward Functions" -- show convergence of R_phi, compare the learned reward landscape to human-designed baselines, verify or falsify the potential-based shaping theory, analyze the compute tradeoff of inner vs outer loop length.

---

## Approach F: Contrastive Trajectory Reward Mining

### Detailed Mechanics

**Pseudocode:**

```
for iteration in 1..20:
    # DATA COLLECTION
    if iteration == 1:
        policy = RandomPolicy()
    # Play 1000 games
    wins, losses = [], []
    for game in 1..1000:
        trajectory = []
        obs = env.reset()
        while not done:
            action = policy.act(obs)
            next_obs, reward, done = env.step(action)
            trajectory.append(obs)
            obs = next_obs
        if reward > 0:
            wins.append(trajectory)
        else:
            losses.append(trajectory)

    # TRAJECTORY FINGERPRINTING
    for feature_idx in 0..88:
        for trajectory in all_trajectories:
            values = [obs[feature_idx] for obs in trajectory]
            fingerprint[feature_idx] = {
                "mean": mean(values),
                "min": min(values),
                "max": max(values),
                "std": std(values),
                "trend": linregress(range(len(values)), values).slope
            }

    # CONTRASTIVE ANALYSIS
    components = []
    for stat in ["mean", "min", "max", "std", "trend"]:
        for feature_idx in 0..88:
            win_values = [fingerprint[feature_idx][stat] for t in wins]
            loss_values = [fingerprint[feature_idx][stat] for t in losses]
            # Cohen's d effect size
            pooled_std = sqrt((std(win)**2 + std(loss)**2) / 2)
            d = (mean(win) - mean(loss)) / pooled_std
            if abs(d) > 0.3:  # medium effect size threshold
                components.append(ContrastiveRewardComponent(
                    feature_index=feature_idx,
                    statistic=stat,
                    sign=sign(d),
                    weight=abs(d) * 0.1,  # scale by effect size
                    effect_size=d
                ))

    # REWARD CONSTRUCTION
    # Keep top 10 components by effect size
    components.sort(key=lambda c: abs(c.effect_size), reverse=True)
    components = components[:10]
    reward_fn = ContrastiveRewardFunction(components)

    # TRAIN
    trainer = PPOTrainer(reward_fn=reward_fn)
    for rollout in 1..200:
        collect_and_train(trainer)

    policy = trainer.agent

    # FEATURE INTERACTION DISCOVERY (iterations 5+)
    if iteration >= 5:
        for i, j in combinations(range(89), 2):
            # Check if the product feature distinguishes wins from losses
            win_products = [mean(obs[i] * obs[j] for obs in t) for t in wins]
            loss_products = [mean(obs[i] * obs[j] for obs in t) for t in losses]
            d = cohens_d(win_products, loss_products)
            if abs(d) > 0.5:  # higher threshold for interactions
                components.append(interaction_component(i, j, d))
```

**Iteration 1 (random policy).** The agent plays randomly. Approximately 33% of games are wins (random chance). The contrastive analysis finds:

- `ego.alive.mean` has the largest effect size (d ~ 1.5): in winning games, the ego survives longer. This is obvious but not actionable.
- `ego.dist_to_edge.mean` has d ~ 0.6: winners spend more time away from edges.
- `ego.speed.mean` has d ~ -0.3: winners move slightly slower (fast agents fly off).
- Enemy features show weak effects (d < 0.2) because random play rarely produces meaningful enemy interactions.

The reward function after iteration 1: `r = 0.06 * obs[5] - 0.03 * obs[6]` (reward dist-to-edge, penalize speed). This is a pure survival reward.

**Iteration 5 (survival-trained policy).** The agent now survives well (~60% survival rate). It wins ~45% of games. The contrastive analysis now finds new signals because the behavior distribution has changed:

- `enemy1.dist_to_edge.mean` has d ~ -0.5: in winning games, enemies are closer to the edge. This was invisible in iteration 1 because random agents don't push enemies consistently.
- `team_alive_advantage.trend` has d ~ 0.4: in winning games, the team-alive advantage trends upward over the game.

The reward function adds: `-0.05 * obs[47] + 0.04 * (obs[86] - obs[87])`. Now the agent has incentive to push enemies toward edges and maintain a numerical advantage.

**Iteration 15.** Feature interaction discovery kicks in. The analysis finds:

- `obs[55] * obs[47]` (enemy1 distance-to-ego * enemy1 distance-to-edge) has d ~ -0.6: in winning games, when the enemy is close to the ego, the enemy is also close to the edge. This encodes: "approach enemies who are near the edge."
- This is a conditional strategy that neither per-feature analysis could discover alone.

### Concrete Example Walkthrough

Same scenario: `penguin_0` at (-60, -10), `penguin_3` at (60, -10).

**Iteration 1 reward function:** `r = 0.06 * obs[5] - 0.03 * obs[6]`

Feature values: obs[5] = 0.393 (dist_to_edge), obs[6] = 0.0 (speed).
Reward: 0.06 * 0.393 - 0.03 * 0.0 = 0.024. Small positive reward for being away from edge. No incentive to engage the enemy at (60, -10).

**Iteration 5 reward function:** `r = 0.06 * obs[5] - 0.03 * obs[6] - 0.05 * obs[47]`

Feature values: obs[47] = 0.393 (enemy1 dist_to_edge).
Reward: 0.024 - 0.05 * 0.393 = 0.024 - 0.020 = 0.004. Almost zero. The enemy is moderately far from the edge, providing slight reward. But if the agent pushes the enemy toward the edge (obs[47] drops to 0.1), the reward becomes 0.024 - 0.005 = 0.019. The difference (0.019 vs 0.004) is small but creates a gradient toward pushing enemies to the edge.

**Iteration 15 reward function (with interactions):** Adds `- 0.04 * obs[55] * obs[47]` (penalize when near an enemy who is not near the edge, reward being near an enemy who IS near the edge).

For the starting state: obs[55] = 1.0 (enemy far), obs[47] = 0.393.
Interaction term: -0.04 * 1.0 * 0.393 = -0.016.
This penalizes the current state (far from enemy who isn't near edge).

But if the agent approaches and pushes the enemy to the edge: obs[55] = 0.2, obs[47] = 0.1.
Interaction term: -0.04 * 0.2 * 0.1 = -0.0008. Almost zero -- the penalty vanishes when the agent is near an enemy who IS near the edge. The reward landscape now explicitly encodes "approach enemies who are near edges."

### What Gets Discovered and When

- **Iteration 1 (1M steps):** Pure survival: dist_to_edge, speed. Win rate ~40%.
- **Iterations 2-5 (2-5M steps):** Survival refines. Speed penalty becomes more precise, positional awareness improves. Win rate ~48%.
- **Iterations 5-10 (5-10M steps):** Aggression: enemy_dist_to_edge, team_alive_advantage. Win rate ~58%.
- **Iterations 10-15 (10-15M steps):** Targeting: feature interactions reveal conditional strategies. Win rate ~65%.
- **Iterations 15-20 (15-20M steps):** Fine-tuning. Diminishing returns on new discoveries. Win rate ~68%.

**The "aha moment"** is iteration 5, when the behavioral distribution shifts enough that enemy-related features become statistically significant. This is visible in the effect-size log: `enemy1.dist_to_edge.mean` jumps from d=0.1 in iteration 4 to d=0.5 in iteration 5.

**Credit assignment:** Contrastive analysis handles credit assignment at the trajectory level, not the step level. It asks "what characterizes winning trajectories?" rather than "what action at step t caused winning at step T?" This means it naturally discovers features that are consistently different across the entire game (like average dist_to_edge), but it struggles with momentary events (like a single critical push at step 25 that won the game). The transition to delta-based and interaction features partially addresses this.

### Architecture Specifics

- **No neural networks in the discovery phase.** The contrastive analysis is pure numpy: compute means, standard deviations, and Cohen's d across trajectory sets. For 1000 games * 89 features * 5 statistics = 445,000 effect sizes to compute. On CPU, this takes under 1 second.
- **Policy network:** Standard 19,843 params during PPO training.
- **Trajectory database:** 1000 games * ~30 rounds/game * 89 features * 4 bytes = ~10.7MB per iteration. Over 20 iterations: ~214MB. Manageable.
- **Reward function:** A list of ~10 components, each with an index, sign, and weight. Under 1KB. The fastest possible reward computation at inference time: 10 multiplications and additions.
- **GPU utilization:** The contrastive analysis is CPU-bound (pure numpy). The PPO training is GPU-bound. The two phases alternate, so GPU utilization is ~90% (idle during the brief analysis phase). Overall the simplest and cheapest GPU profile of all six approaches.
- **Feature interactions:** Pairwise analysis of 89 features = 89*88/2 = 3,916 pairs. With 5 statistics each = 19,580 effect sizes. Still under 1 second on CPU. Cubic (triplet) interactions would be 89*88*87/6 = ~113K combinations -- still feasible but getting large. Cap at pairwise for tractability.

### What Could Go Wrong

1. **Correlation is not causation.** The contrastive analysis finds features that are *correlated* with winning, not features that *cause* winning. For example, `timestep.mean` might be lower in winning games (games end faster when you're winning), but rewarding low timestep would encourage the agent to lose quickly. **Diagnosis:** After adding a component, check if win rate improves. If it drops, the correlation was spurious. **Fix:** Validate each component by training with it and checking win rate before adding it permanently.

2. **Random policy produces weak signal.** In iteration 1, a random policy generates nearly identical win and loss distributions, so most effect sizes are small. The discovered features may be noise. **Diagnosis:** Check statistical significance (p-values) alongside effect sizes. With 333 wins and 667 losses out of 1000 games, even d=0.3 may not be significant. **Fix:** Play more games (5000 instead of 1000) to increase statistical power.

3. **Trajectory fingerprinting is too coarse.** Summarizing a 30-step trajectory by mean/min/max/std/trend loses temporal structure. A critical event at step 25 (a knockout) is averaged away. **Diagnosis:** Check if the contrastive analysis finds any "event-type" features (like enemy_alive.min = 0 in winning games, meaning an enemy was knocked out). **Fix:** Add event-based features to the fingerprint: "did any enemy die?", "what round did the first knockout happen?", "did the ego survive to the final round?"

4. **Feature interactions explode combinatorially.** With 89 features and 5 statistics, there are potentially hundreds of significant interactions. Many will be redundant or collinear. **Diagnosis:** If the reward function has 50+ components, it's probably overfitting. **Fix:** Apply a higher threshold for interactions (d > 0.5 instead of 0.3), and cap total components at 15.

### What Makes It Novel for Research

**Unique data generated:** Each iteration produces a ranked list of (feature_name, statistic_type, effect_size, sign) that reads like a game analysis report: "The strongest predictor of winning is enemy1_dist_to_edge.min (d = -1.2), followed by ego_dist_to_edge.mean (d = 0.9)." Over 20 iterations, this sequence shows how the "story of winning" changes as the agent improves.

**Foundation model connection:** These contrastive analysis reports are the most directly language-convertible data product. The sentence "In winning games, enemies spend more time near the edge (d=1.2, p<0.001)" is natural language game analysis that could train an LLM to analyze game replays and generate strategic insights. This connects to the project's NLP/strategy bot goal more directly than any other approach.

**Paper angle:** "Contrastive Trajectory Mining: Statistical Reward Discovery for Multi-Agent Physics Games" -- show that simple statistical analysis of win/loss trajectories discovers the same features as neural attribution methods (cross-reference with Approach A), demonstrate the interaction discovery mechanism, and show that the resulting reward functions accelerate PPO training 2-4x over sparse-only training.

---

## How They Interact and Combine

### Best Combination Strategy

The approaches are not mutually exclusive. The recommended combination:

**Layer 1: F (Contrastive) as Foundation.** Run first. Produces an interpretable baseline reward and, crucially, the feature-importance rankings and trajectory database that inform every other approach. Cost: ~1 week.

**Layer 2: A (Attention) for Validation.** Run with the same PPO training. Compare the gradient-attribution-discovered features to the contrastive-analysis-discovered features. Where they agree, you have high-confidence reward components (correlated with winning AND causally used by the value network). Where they disagree, you have interesting puzzles. Cost: minimal marginal effort (attribution is just a backward pass on the existing value network).

**Layer 3: B (LLM) as Integrator.** Feed the LLM the contrastive analysis from F and the attribution scores from A as additional context in the prompt. The LLM now has principled statistical evidence to guide its reward design, rather than guessing from raw statistics. This dramatically reduces the number of LLM iterations needed to reach a good reward function. Cost: ~1 week, but better results.

**Layer 4: D (Curiosity) for Exploration.** Run the curiosity phase (Phase 1) separately to build a diverse replay buffer. Then use this buffer for the contrastive analysis in F (instead of random-policy games, use curiosity-policy games). This gives contrastive analysis access to rarer, more informative states. Cost: ~500 rollouts extra for the curiosity phase.

**Layer 5: C (Evolutionary) or E (Meta-Gradient) for Polish.** Use the discovered reward components from F+A+B as the feature basis for evolutionary search, or as the initialization for the meta-gradient reward network. This avoids the slow random-start phase. Cost: depends on compute budget.

### Specific Combinations Worth Building

1. **F + A (Contrastive + Attention).** Almost free to combine. Run both analyses on the same training run. Produces a correlation-vs-causation comparison table. Implementation: add attribution scoring to the existing PPO training loop.

2. **F + B (Contrastive + LLM).** The LLM's prompt includes the contrastive report. Instead of "here are raw stats," the prompt says "contrastive analysis found that winning games have higher ego_dist_to_edge (d=0.9) and lower enemy_dist_to_edge (d=-1.2). Write a reward function that incentivizes these." The LLM writes better reward functions faster.

3. **D + F (Curiosity + Contrastive).** Curiosity builds a diverse replay buffer. Contrastive analysis runs on this buffer. Because curiosity explored unusual states, the contrastive analysis discovers features that random play would never reveal (e.g., "in winning games where a 3-agent collision occurred, the ego's speed was lower" -- a conditional strategy visible only in rare states).

4. **A + E (Attention + Meta-Gradient).** Use the attention-discovered features to initialize the meta-gradient reward network. Instead of starting R_phi from random weights, initialize it to output a linear combination of the top attention features. The meta-gradient then refines this initialization, potentially discovering nonlinear reward structures that linear attribution cannot.

---

## Specific Question Deep-Dives

### Q1: How does the world model (Approach D) actually learn physics?

Approach D's curiosity module does not build an explicit world model. The RND module learns a representation of "what states have I seen before," which implicitly encodes some physics structure (states that are physically reachable vs unreachable). It does NOT learn to predict what happens when two penguins collide.

If you wanted a true world model that predicts collision outcomes, you would add a **dynamics model** to Phase 1:

```python
class DynamicsModel(nn.Module):
    def __init__(self, obs_dim=89, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, obs_dim)  # predict next observation
        )
    
    def predict(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))
```

This model would train on (obs_t, action_t, obs_{t+1}) tuples from the replay buffer. After training on ~1M transitions, it could predict:
- **Simple motion:** If ego pushes right, ego moves right. Accuracy: very high (~95% within 1 step).
- **Wall/edge effects:** If ego is at the edge and pushes outward, ego falls off. Accuracy: moderate (~80%) -- the arena boundary is a discontinuity.
- **Two-penguin collisions:** If ego and enemy are close and both moving toward each other, predict the post-collision velocities. Accuracy: low (~50%) -- collisions involve complex Pymunk physics with friction, restitution, and angular momentum that a simple MLP struggles with.

The dynamics model would be most useful for planning (model-based RL), not for reward discovery. It could answer "if I push this way, will the enemy fall off?" but that is a different problem than "what should I reward?"

### Q2: GPU Tensor Renderer for Visual RL

The existing codebase uses `TensorVecEnv` for GPU-accelerated vectorized environments. A visual RL approach would add a GPU tensor renderer:

**Architecture:**
```python
class GPUTensorRenderer:
    def __init__(self, resolution=84, num_envs=64, device="cuda"):
        # Pre-allocate frame buffer on GPU
        self.frames = torch.zeros(num_envs, 3, resolution, resolution, device=device)
        # Arena circle (precomputed mask)
        self.arena_mask = precompute_circle_mask(resolution, radius=100)
    
    def render_batch(self, positions, velocities, alive_flags, arena_radius):
        """Render all envs in parallel on GPU."""
        self.frames.zero_()
        # Draw arena (vectorized circle fill)
        self.frames[:, :, self.arena_mask] = 0.3  # gray arena
        # Draw penguins (vectorized gaussian splats at positions)
        for team in [0, 1]:
            color = [1, 0, 0] if team == 0 else [0, 0, 1]
            for i in range(3):
                pos = positions[:, team*3+i]  # (num_envs, 2)
                alive = alive_flags[:, team*3+i]  # (num_envs,)
                # Convert world coords to pixel coords
                px = ((pos[:, 0] / arena_radius + 1) * resolution / 2).long()
                py = ((pos[:, 1] / arena_radius + 1) * resolution / 2).long()
                # Gaussian splat (radius ~5 pixels)
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if dx*dx + dy*dy <= 25:
                            x = (px + dx).clamp(0, resolution-1)
                            y = (py + dy).clamp(0, resolution-1)
                            for c in range(3):
                                self.frames[:, c, y, x] += color[c] * alive
        return self.frames
```

**Speed estimate:** At 84x84 resolution with 64 envs, rendering is ~6 Gaussian splats per frame * 64 envs = ~384 splats. Each splat touches ~80 pixels. Total: ~30K pixel operations. On a modern GPU (RTX 3080), this is negligible -- under 0.1ms per batch. The bottleneck would be the CNN policy network processing these frames, not the rendering.

**CNN policy for visual RL:**
```
Input: (64, 3, 84, 84) frames
Conv2d(3, 32, 8, stride=4) -> (64, 32, 20, 20)
Conv2d(32, 64, 4, stride=2) -> (64, 64, 9, 9)
Conv2d(64, 64, 3, stride=1) -> (64, 64, 7, 7)
Flatten -> (64, 3136)
Linear(3136, 512) -> (64, 512)
Actor: Linear(512, 2)
Critic: Linear(512, 1)
Total: ~1.7M parameters (~6.8MB)
```

This is 85x larger than the vector-obs network. Training would be ~10x slower due to the CNN forward/backward passes. On a single GPU, expect ~5K steps/sec instead of ~50K steps/sec.

### Q3: When Do Emergent Strategies Appear in Self-Play?

Self-play is not one of the 6 approaches in the document, but it connects to Approach C (evolutionary) and is part of the existing codebase (`self_play.py`). Emergent strategies in knockout self-play follow a predictable progression:

- **1M steps:** Random-but-alive. Agents learn not to fly off the edge. No inter-agent strategy.
- **3M steps:** Passive positioning. Agents cluster near center. Games last longer.
- **5M steps:** First contact. Agents discover that collisions sometimes knock enemies off. No targeting, just "bumping is sometimes good."
- **10M steps:** Directional aggression. Agents learn to push toward the edge (not just bump randomly). This is the first true "strategy" -- it requires understanding that the edge is where kills happen.
- **20M steps:** Targeting. Agents preferentially approach the enemy closest to the edge. This is a compound strategy (assess enemy position + approach + push).
- **30M steps:** Team tactics. Agents coordinate implicitly -- two agents converge on one enemy, or one agent acts as bait while another pushes. These are emergent, not programmed.
- **50M+ steps:** Meta-strategies. In self-play, the opponent also improves. This creates an arms race: defensive play (avoiding being pushed) emerges to counter aggressive play, then flanking emerges to counter defensive play. The ELO rating in `elo_rating.py` tracks this progression.

The strategies "look like" specific patterns in the action distribution: early agents have uniform angle distributions. Aggressive agents have angles concentrated toward enemies. Tactical agents have angles that shift based on enemy position relative to the edge.

### Q4: What Skills Would DIAYN Discover?

DIAYN (Diversity Is All You Need) is not explicitly one of the 6 approaches, but it relates to Approach D's curiosity framework. If you ran DIAYN on the knockout game with K=8 skills:

**Likely emergent skills:**
1. "Spin in place" (low speed, random angle)
2. "Rush toward center" (always move toward (0,0))
3. "Rush toward edge" (always move outward -- this agent dies quickly)
4. "Chase nearest enemy" (minimize distance to closest opponent)
5. "Flee from nearest enemy" (maximize distance)
6. "Circle the arena" (consistent angular movement)
7. "High-speed straight line" (one direction, max power)
8. "Oscillate" (alternate between two directions)

**Would any be useful?** Skills 2 and 4 are directly useful (survival and aggression). Skill 5 is useful defensively. The others (spin, rush-to-edge, circle, straight-line, oscillate) are not strategically useful but represent the diversity that DIAYN optimizes for. The fundamental issue is that DIAYN optimizes for discriminability (each skill should be distinguishable by a classifier), not usefulness. A "rush toward edge" skill is very discriminable (it produces unique states) but strategically terrible.

**To make DIAYN useful:** Use the discovered skills as building blocks for a hierarchical policy. A meta-controller selects which skill to activate at each timestep. Train the meta-controller with win/loss reward. This way, the meta-controller learns "use skill 4 (chase enemy) when enemies are near the edge, use skill 2 (rush center) when I'm near the edge." The skills provide temporal abstraction that makes long-horizon planning easier.

### Q5: Visualizing What the Contrastive Encoder Learned

Approach F does not use an encoder (it's pure statistics). But you can visualize its discoveries effectively.

**Visualization 1: Effect Size Heatmap.** Create an 89x5 heatmap where rows are observation features and columns are statistics (mean, min, max, std, trend). Cell color = Cohen's d between wins and losses. This immediately shows: feature 5 (ego_dist_to_edge) mean is bright green (d=+0.9, higher in wins), feature 47 (enemy1_dist_to_edge) mean is bright red (d=-1.2, lower in wins). You can literally "see it discover edge = bad" as a bright red row for features 47, 61, 75 (all three enemy dist_to_edge features).

**Visualization 2: Discovery Timeline.** Plot effect sizes over iterations. In iteration 1, only ego features have significant effect sizes. By iteration 5, enemy features appear. By iteration 10, interaction features appear. This shows the progression from "survive" to "attack" to "target."

**Visualization 3: Arena Heatmap.** For each (x, y) position in the arena, compute the average reward from the discovered reward function. Plot as a 2D heatmap. Center should be bright (high reward), edges should be dark (low reward), and positions near enemies should have gradients pointing toward the edge. This is a direct visualization of "what the agent thinks is good."

**"Can you see it discover edge = bad?"** Yes. In iteration 1, the effect size for ego_dist_to_edge is d=0.6 (moderate). By iteration 3, it's d=0.9 (large). Meanwhile, ego_dist_from_center goes from d=-0.4 to d=-0.7. On the arena heatmap, the reward is uniform in iteration 1 and has a clear center-high, edge-low gradient by iteration 3. You are watching the agent discover that edges are dangerous.

### Q6: Why Would LSTM Help in Approach E, and What Memory Would the Agent Need?

The existing `ActorCritic` network is feedforward (MLP). It sees only the current 89-dim observation and must decide an action. An LSTM adds temporal memory:

```python
class LSTMActorCritic(nn.Module):
    def __init__(self, obs_dim=89, hidden_size=128, action_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
```

**What memory would the agent need?**

1. **Opponent behavior modeling.** "Penguin_3 has been moving rightward for the last 4 rounds -- it's probably still going right." The current MLP sees only velocity at this instant, not trajectory over time. The LSTM can track multi-step patterns.

2. **Counting knockouts.** "I've already knocked out 2 enemies -- I should play defensively to protect my lead." The global features (team_alive_count) partially encode this, but the LSTM can track the sequence of knockouts and adjust strategy accordingly.

3. **Arena shrinkage timing.** "The arena shrank 2 rounds ago; it will shrink again soon." The timestep feature (obs[88]) partially encodes this, but the LSTM can learn the shrinkage schedule from observation sequences.

4. **Momentum tracking.** "I just pushed enemy_3 and he's heading toward the edge -- I should follow up." A single observation shows current velocity, but the LSTM can track whether velocity is increasing or decreasing over time, enabling follow-up pushes.

5. **Team coordination memory.** "My ally pushed enemy_1 left 3 rounds ago -- I should come from the right to trap it." This requires remembering past positions of allies and enemies.

**Why LSTM helps specifically for Approach E (meta-gradient):** The meta-reward network R_phi outputs reward based on a single observation. With an LSTM policy, the agent can learn from sequences of rewards, enabling R_phi to express temporally extended reward signals. For example, R_phi could learn to output low reward when the agent approaches an enemy but high reward when the approach continues for 3+ steps (encouraging sustained pursuit rather than one-step approach-then-retreat). The LSTM policy captures this temporal structure; the MLP cannot.

**LSTM architecture specifics:** An LSTM(89, 128) adds 89*128*4 + 128*128*4 + 128*4*2 = 45,568 + 65,536 + 1,024 = **112,128 parameters**. Total model: ~132K params (~528KB). About 6.5x larger than the MLP. Training is ~2x slower due to sequential LSTM processing (cannot parallelize across time steps within a sequence). The rollout buffer must store hidden states (2 * 128 = 256 floats per transition), increasing buffer size by ~3x.

---

## Summary Decision Matrix

For choosing which to build first, here is the pragmatic tradeoff:

| Factor | F (Contrastive) | A (Attention) | B (LLM) | D (Curiosity) | C (Evolutionary) | E (Meta-Grad) |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Implementation time | 1 wk | 1 wk | 1.5 wk | 1.5 wk | 1 wk | 2 wk |
| Compute cost | Very low | Low | Low | Medium | High | High |
| Foundation model data value | High | Good | Excellent | Good | Good | Moderate |
| Interpretability | Excellent | High | Excellent | Low | High | Low |
| Novelty for a paper | Moderate | Moderate | High | Moderate | Moderate | High |
| Risk of total failure | Low | Low | Medium | Medium | Medium | High |
| Depends on other approaches | No | No | Better with F+A | No | Better with F | Better with A |

**If you build only one:** F (Contrastive). Lowest risk, fastest to implement, produces immediately useful output, and provides the foundation data that makes every other approach work better.

**If you build two:** F + B (Contrastive + LLM). The LLM approach is the richest data generator for the foundation model goal, and feeding it contrastive analysis results dramatically improves the LLM's reward designs.

**If you build three:** F + A + B. Adding attention gives you a three-way comparison (statistical correlation vs neural attribution vs LLM reasoning) that is itself a compelling research narrative.
