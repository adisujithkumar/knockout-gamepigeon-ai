# Blank-Slate Reward Discovery for Knockout

## Preamble: What the Agent Starts With

The knockout game is a 3v3 penguin physics game on a shrinking square arena.
Each agent receives an 89-dimensional observation per round and outputs a
continuous action `[angle_degrees, power]`. The current `TensorVecEnv`
provides only a sparse terminal signal: `+1` for win, `-1` for loss, `0`
otherwise. The strategy_reward.py stub is empty.

The 89-dim observation decomposes as:

```
ego(14) + ally1(14) + ally2(14) + enemy1(14) + enemy2(14) + enemy3(14) + global(5)

Per-penguin features (14 each):
  0: position_x           1: position_y
  2: velocity_x           3: velocity_y
  4: dist_from_center     5: dist_to_edge
  6: speed                7: heading
  8: alive                9: rel_position_x
 10: rel_position_y      11: rel_velocity_x
 12: rel_velocity_y      13: dist_to_ego

Global features (5):
 84: team_a_alive/3       85: team_b_alive/3
 86: ego_team_alive/3     87: opp_team_alive/3
 88: timestep
```

The question: how does an agent discover WHAT to reward itself for, starting
from nothing but the win/loss outcome?

---

## Approach A: Feature Attention Discovery

### How It Works

The core insight: a value network trained on sparse win/loss rewards implicitly
learns which observation features predict winning. We can extract this implicit
knowledge and crystallize it into an explicit, interpretable reward function.

**Algorithm:**

1. **Phase 1 -- Pure Sparse Training (bootstrap).** Train a standard PPO agent
   for N rollouts using only the sparse `+1/-1` terminal reward. The value
   network `V(s)` learns to predict expected return from each state. This is
   slow but produces a crude value function.

2. **Phase 2 -- Feature Attribution.** After every K rollouts, compute
   feature-level attribution scores for the value network. For each of the 89
   observation dimensions, measure how much it contributes to the value
   estimate. Methods:
   - **Gradient attribution:** `dV/d(obs_i)` averaged over a buffer of
     states. Features with large absolute gradients are "attended to" by the
     value network.
   - **Permutation importance:** For each feature i, permute its values across
     the state buffer and measure the drop in value prediction accuracy.
   - **Integrated gradients:** Accumulate gradients along the path from a
     zero baseline to the actual observation, giving a principled attribution.

3. **Phase 3 -- Reward Crystallization.** Take the top-K features by
   attribution score. For each, construct a simple reward component:
   - If the feature is `enemy_i.dist_to_edge` and its gradient is negative
     (value increases as enemy approaches edge), create reward:
     `r_i = -w_i * obs[idx]` (reward for pushing enemies toward edge).
   - If the feature is `ego.dist_to_edge` and gradient is positive, create:
     `r_i = +w_i * obs[idx]` (reward for staying away from edge).
   - Weights `w_i` are proportional to attribution magnitude, normalized so
     the total shaping reward magnitude is bounded (e.g., sum of |w_i| = 0.5).

4. **Phase 4 -- Accelerated Training.** Continue PPO training using:
   `r_total = r_terminal + sum(w_i * component_i(obs))`.
   The reward function is re-crystallized every K rollouts, allowing the
   agent to discover progressively subtler features (first it finds "don't
   fall off", then "push enemies off", then "target enemies near edges").

5. **Phase 5 -- Reward Decay.** As training progresses, decay the shaping
   weights toward zero so the agent converges on the true objective. This
   prevents reward hacking on the shaped components.

### Data Structure for the Reward Function

```python
@dataclass
class CrystallizedReward:
    """An explicit reward component discovered from value network attention."""
    feature_index: int          # Index into 89-dim obs
    feature_name: str           # Human-readable name
    weight: float               # Reward weight (positive = reward high values)
    attribution_score: float    # How important the value network thinks this is
    discovered_at_rollout: int  # When this component was first discovered
    sign_explanation: str       # "higher is better" or "lower is better"

class AttentionRewardFunction:
    components: list[CrystallizedReward]
    terminal_weight: float = 1.0  # Always keep terminal reward

    def compute(self, obs: np.ndarray, terminal_reward: float) -> float:
        shaped = sum(c.weight * obs[c.feature_index] for c in self.components)
        return self.terminal_weight * terminal_reward + shaped
```

### Inner/Outer Loop Structure

- **Inner loop:** PPO training with current reward function (K = 50 rollouts).
- **Outer loop:** Feature attribution + reward crystallization (every K
  rollouts). Run attribution over the most recent replay buffer. Update the
  component list. Log what changed.

### Integration Points

| Component | File | Changes |
|-----------|------|---------|
| `AttentionRewardFunction` | `src/knockout/reward/attention_reward.py` | New file |
| Feature attribution engine | `src/knockout/reward/feature_attribution.py` | New file |
| Modified PPO trainer | `src/knockout/training/ppo.py` | Add `reward_fn` parameter to `collect_rollout_vec` |
| Modified TensorVecEnv | `src/knockout/env/tensor_env.py` | Add hook to apply shaped reward before returning |
| Discovery logger | `src/knockout/reward/discovery_log.py` | New file: logs what features were discovered when |

The key integration change is in `TensorVecEnv.step()` at line 159-167 where
rewards are computed. Instead of hardcoding sparse rewards, accept an optional
`RewardFunction` that receives the full observation tensor and terminal signal:

```python
# In TensorVecEnv.__init__:
self.reward_fn: RewardFunction | None = reward_fn

# In TensorVecEnv.step(), replace lines 159-167:
if self.reward_fn is not None:
    rewards_t = self.reward_fn.compute_batch(obs_tensor[:, :3], a_wins, b_wins)
else:
    # existing sparse reward logic
```

### What Makes It Interesting for Research

- **Interpretable discovery trajectory.** The log of "which features were
  discovered when" is a readable narrative of the agent's learning: "At
  rollout 50, the agent discovered that ego distance-to-edge matters. At
  rollout 150, it discovered that enemy distance-to-edge matters. At rollout
  300, it discovered that relative velocity toward enemies matters."
- **Emergent curriculum.** The agent self-sequences its own curriculum:
  survival first, then aggression, then tactics.
- **Foundation model training data.** Each crystallization step produces a
  (game_state_statistics, discovered_reward_component) pair. A language model
  could learn to predict what reward components an agent should discover next,
  given its current skill level.
- **Comparison to human intuition.** We can compare the order of feature
  discovery against the heuristic agent's hardcoded priorities (edge_score
  40%, distance_score 25%, vel_toward_edge 35%) to see if the RL agent
  converges on the same or different priorities.

### Estimated Implementation Effort

- **Core implementation:** 3-4 days (attribution engine, reward class, PPO
  integration, logging).
- **Tuning:** 2-3 days (attribution method selection, K schedule, weight
  normalization, decay schedule).
- **Total:** ~1 week.

---

## Approach B: LLM Reward Architect

### How It Works

An LLM observes game transcripts and statistics, then writes Python reward
functions. The reward function is a readable, editable artifact that evolves
over many iterations. This is a form of program synthesis where the LLM acts
as a reward designer.

**Algorithm:**

1. **Bootstrap.** Start with the minimal reward function:
   ```python
   def reward(obs, terminal, info):
       return terminal  # +1 win, -1 loss, 0 ongoing
   ```

2. **Play phase.** Train PPO for N rollouts (e.g., 200) using the current
   reward function. During training, collect detailed statistics:
   - Win rate, average game length, survival rate per penguin
   - Per-feature distributions at win vs. loss (which features differ
     between winning and losing states?)
   - Action distributions (what angles and powers are used?)
   - Knockout events (who knocked whom off, from what positions?)
   - Trajectory snippets (5-10 representative game traces as structured data)

3. **Reflect phase.** Package the statistics into an LLM prompt:
   ```
   You are designing a reward function for a 3v3 penguin knockout game.

   Current reward function:
   ```python
   {current_reward_code}
   ```

   Training results after {N} rollouts:
   - Win rate: {win_rate}%
   - Avg game length: {avg_length} rounds
   - Ego survival rate: {survival}%
   - Enemy knockouts per game: {knockouts}
   - Feature analysis: {feature_comparison}
   - Sample trajectory: {trajectory}

   The observation is 89-dimensional:
   {obs_layout_description}

   Write an improved reward function. You may add intermediate rewards
   for behaviors that lead to winning. The function signature is:

   def reward(obs: np.ndarray, prev_obs: np.ndarray | None,
              terminal: float, info: dict) -> float:

   Explain your reasoning, then write the code.
   ```

4. **Compile and validate.** Parse the LLM's response, extract the Python
   code, compile it, and run it through a validation suite:
   - Type check: function accepts the right arguments and returns float
   - Bounds check: run it on 1000 random observations, verify rewards are
     in a reasonable range (e.g., [-10, 10])
   - Monotonicity sanity check: does winning produce higher reward than
     losing?
   - If validation fails, re-prompt the LLM with the error

5. **Train phase.** Use the new reward function for the next N rollouts.
   Repeat from step 2.

6. **Archive.** Every iteration, save the reward function code, the prompt,
   the LLM's reasoning, and the resulting training metrics. This creates a
   rich dataset of (context, reward_function, outcome) tuples.

### Data Structure for the Reward Function

The reward function is **literal Python source code** stored as a string and
dynamically compiled:

```python
@dataclass
class RewardProgram:
    source_code: str              # The Python reward function
    version: int                  # Monotonically increasing
    author_reasoning: str         # LLM's explanation of why it wrote this
    training_context: dict        # Stats that prompted this version
    compiled_fn: Callable | None  # The compiled function (transient)
    metrics_after_training: dict | None  # Results of training with this reward

class LLMRewardArchitect:
    history: list[RewardProgram]  # Full history of all reward versions
    current: RewardProgram        # Active reward program

    def compile(self, source: str) -> Callable:
        """Safely compile reward function from source."""
        namespace = {"np": np, "math": math}
        exec(source, namespace)
        return namespace["reward"]

    def validate(self, fn: Callable, n_samples: int = 1000) -> list[str]:
        """Run validation checks, return list of errors (empty = pass)."""
        ...
```

### Inner/Outer Loop Structure

- **Inner loop:** PPO training for N = 200 rollouts with the current reward
  function (~50k-100k timesteps depending on num_envs).
- **Outer loop:** LLM reflection + reward rewrite. Each outer iteration costs
  one LLM API call plus validation. Run for ~20-50 outer iterations total.

### Integration Points

| Component | File | Changes |
|-----------|------|---------|
| `LLMRewardArchitect` | `src/knockout/reward/llm_architect.py` | New file |
| `RewardProgram` | `src/knockout/reward/llm_architect.py` | Data class in same file |
| Stats collector | `src/knockout/reward/game_stats.py` | New file: collects per-game statistics during rollouts |
| Prompt templates | `src/knockout/reward/prompts/` | New directory with prompt templates |
| Modified PPO trainer | `src/knockout/training/ppo.py` | Accept `reward_fn: Callable` in rollout collection |
| Outer loop driver | `src/knockout/training/reward_discovery_loop.py` | New file: orchestrates play-reflect-train cycle |

The key change to `collect_rollout_vec` is storing `prev_obs` and passing it
to the reward function, since the LLM might write delta-based rewards:

```python
# In collect_rollout_vec, after vec_env.step():
if self.reward_fn is not None:
    flat_rewards = self.reward_fn(
        obs=flat_obs, prev_obs=prev_flat_obs,
        terminal=flat_rewards, info=batch_info
    )
```

### What Makes It Interesting for Research

- **The reward function is a readable artifact.** Unlike neural reward
  functions, each version is human-interpretable Python. You can read the
  progression from "just win/loss" to a sophisticated multi-component reward.
- **LLM as scientist.** The LLM is performing a form of scientific inquiry:
  observe behavior, hypothesize what intermediate rewards would help, test,
  iterate. The reasoning traces are training data for teaching LLMs to design
  reward functions.
- **Bridges RL and NLP.** The reward function history is a natural language +
  code artifact that connects game understanding to reward engineering. This
  directly feeds the project's NLP/LLM strategy bot goal.
- **Failure modes are data.** When the LLM writes a reward function that
  causes reward hacking (agent farms a shaped reward without winning), the
  correction in the next iteration is valuable training data for teaching
  LLMs about reward design pitfalls.
- **Composable with other approaches.** The LLM could be given the feature
  attribution scores from Approach A as additional context.

### Estimated Implementation Effort

- **Core implementation:** 4-5 days (LLM integration, prompt engineering,
  code compilation/validation, stats collection, outer loop).
- **Prompt iteration:** 2-3 days (getting the prompt format right so the LLM
  writes valid, useful reward functions).
- **Total:** ~1.5 weeks.

---

## Approach C: Evolutionary Reward Search

### How It Works

Maintain a population of reward functions. Each reward function is a weighted
linear combination of predefined features computed from the observation. Use
evolutionary strategies (selection, mutation, crossover) to search the space
of reward functions, evaluating each by training a PPO agent and measuring
its win rate.

**Algorithm:**

1. **Define the feature basis.** Extract M = ~20 reward-relevant features
   from the 89-dim observation. These are not the raw observation features but
   computed quantities designed to be potential reward components:

   ```python
   REWARD_FEATURES = {
       # Survival
       "ego_dist_to_edge":       lambda obs: obs[5],
       "ego_alive":              lambda obs: obs[8],
       # Aggression
       "nearest_enemy_dist":     lambda obs: obs[55],  # enemy1.dist_to_ego
       "enemy1_dist_to_edge":    lambda obs: obs[47],  # enemy1.dist_to_edge
       "enemy2_dist_to_edge":    lambda obs: obs[61],  # enemy2.dist_to_edge
       "enemy3_dist_to_edge":    lambda obs: obs[75],  # enemy3.dist_to_edge
       # Team
       "team_alive_advantage":   lambda obs: obs[86] - obs[87],
       "allies_alive":           lambda obs: obs[86],
       "enemies_alive":          lambda obs: obs[87],
       # Positioning
       "ego_dist_from_center":   lambda obs: obs[4],
       "ego_speed":              lambda obs: obs[6],
       # Relative
       "closing_speed_enemy1":   lambda obs: -(obs[53] * ... ),  # computed
       # Delta-based (need prev_obs)
       "enemy_alive_delta":      lambda obs, prev: prev[87] - obs[87],
       "ego_edge_improvement":   lambda obs, prev: obs[5] - prev[5],
   }
   ```

2. **Represent each individual.** A reward function is a weight vector
   `w` of length M, plus a terminal weight `w_T`:
   `R(obs, terminal) = w_T * terminal + sum(w_i * feature_i(obs))`

3. **Initialize population.** Create P = 20 individuals. Half start with
   all-zero weights (pure terminal reward). Half start with small random
   weights drawn from N(0, 0.1).

4. **Evaluate fitness.** For each individual:
   - Initialize a fresh PPO agent
   - Train for E = 50 rollouts using this individual's reward function
   - Evaluate: play 100 games against a random opponent
   - Fitness = win rate (0.0 to 1.0)

5. **Selection.** Tournament selection: randomly pick 3 individuals, keep the
   one with highest fitness. Repeat P times to fill the next generation.

6. **Mutation.** For each selected individual, with probability p_mut = 0.3:
   - Perturb a random weight: `w_i += N(0, 0.05)`
   - With small probability (0.05), set a weight to zero (feature dropout)
   - With small probability (0.05), initialize a zero weight to N(0, 0.1)
     (feature discovery)

7. **Crossover.** For each pair of selected individuals, with probability
   p_cross = 0.3:
   - Uniform crossover: for each weight, randomly pick from parent A or B

8. **Elitism.** Always carry forward the top 2 individuals unchanged.

9. **Repeat** for G = 30 generations.

### Data Structure for the Reward Function

```python
@dataclass
class EvolvableReward:
    weights: np.ndarray           # (M,) weight vector
    terminal_weight: float = 1.0  # Weight on win/loss signal
    generation: int = 0           # Which generation this was born in
    parent_ids: tuple[int, ...] = ()  # Lineage tracking
    fitness: float = 0.0         # Win rate after training
    feature_names: list[str] = field(default_factory=list)

    def compute(self, obs: np.ndarray, terminal: float) -> float:
        features = np.array([f(obs) for f in FEATURE_EXTRACTORS])
        return self.terminal_weight * terminal + np.dot(self.weights, features)

class RewardPopulation:
    individuals: list[EvolvableReward]
    generation: int
    hall_of_fame: list[EvolvableReward]  # Best from each generation
    feature_registry: dict[str, Callable]  # Name -> extractor
```

### Inner/Outer Loop Structure

- **Innermost loop:** PPO rollout collection and training (E = 50 rollouts
  per individual).
- **Middle loop:** Evaluate each of P = 20 individuals (parallelizable across
  GPUs or even across machines).
- **Outer loop:** Evolutionary generation (G = 30 generations).
- **Total compute:** P * E * G = 20 * 50 * 30 = 30,000 rollouts. With 64
  parallel envs and 128 steps per rollout, that is ~245M timesteps. On GPU
  with TensorVecEnv this is feasible in a few hours.

### Integration Points

| Component | File | Changes |
|-----------|------|---------|
| `EvolvableReward` | `src/knockout/reward/evolutionary_reward.py` | New file |
| `RewardPopulation` | `src/knockout/reward/evolutionary_reward.py` | Same file |
| Feature basis | `src/knockout/reward/reward_features.py` | New file: defines the M feature extractors |
| Evolution driver | `src/knockout/training/evolutionary_search.py` | New file: selection, mutation, crossover, evaluation |
| Modified PPO | `src/knockout/training/ppo.py` | Accept reward_fn, add short-training mode |
| Fitness evaluator | `src/knockout/training/evaluation.py` | Extend existing file with tournament evaluation |

### What Makes It Interesting for Research

- **Phylogenetic tree of reward functions.** By tracking lineage (parent_ids),
  we can reconstruct the evolutionary tree of reward functions. Which features
  were discovered first? Which combinations survive? This phylogeny is a
  unique dataset.
- **Feature importance emerges from selection pressure.** Unlike gradient-based
  attribution, evolution discovers features that are causally useful for
  learning (not just correlated with winning). A feature might have high
  gradient attribution but be useless as a reward signal if it causes reward
  hacking.
- **Robust to reward hacking.** If a reward function causes the agent to
  exploit a shaped reward without actually winning, it gets low fitness and
  is eliminated. Evolution naturally selects against reward hacking.
- **Embarrassingly parallel.** Each individual's training is independent,
  making this ideal for multi-GPU or distributed setups.
- **Foundation model data.** The (feature_weights, resulting_behavior,
  win_rate) tuples across generations teach what reward functions lead to what
  behaviors. An LLM could learn to predict good reward weights from game
  descriptions.

### Estimated Implementation Effort

- **Core implementation:** 3-4 days (feature basis, evolution operators,
  evaluation loop, lineage tracking).
- **Compute optimization:** 1-2 days (parallel individual evaluation,
  checkpoint management).
- **Total:** ~1 week. Compute cost is the main bottleneck, not engineering.

---

## Approach D: Curiosity-Driven Discovery with Hindsight Reward Modeling

### How It Works

Phase 1 uses intrinsic curiosity to explore the state space without any task
reward. Phase 2 builds a learned reward model from hindsight analysis of which
state transitions correlated with winning. Phase 3 blends curiosity and the
learned reward model, gradually shifting weight from exploration to
exploitation.

**Algorithm:**

1. **Phase 1 -- Curiosity-Only Exploration.**

   Implement Random Network Distillation (RND):
   - **Target network:** A fixed, randomly initialized MLP that maps
     observations to a 32-dim embedding: `f_target(obs) -> R^32`.
   - **Predictor network:** A trainable MLP with the same architecture:
     `f_pred(obs) -> R^32`.
   - **Intrinsic reward:** `r_int = ||f_target(obs) - f_pred(obs)||^2`.
     Novel states (ones the predictor has not been trained on) produce high
     prediction error = high intrinsic reward.
   - Train the predictor to minimize its error on visited states.
   - Train PPO using only `r_int` as reward (no win/loss signal).
   - This produces an agent that explores diverse states: different positions,
     speeds, interactions with enemies, edge proximity, etc.
   - Run for N_explore = 500 rollouts.

2. **Phase 2 -- Hindsight Reward Model Construction.**

   After curiosity exploration, the agent has visited a wide distribution of
   states, some of which led to wins and some to losses. Now build a reward
   model:

   - Collect a large replay buffer of (observation, outcome) pairs from
     many games. `outcome = +1` if the agent's team eventually won from
     this state, `-1` if lost, `0` if draw.
   - Train a small neural network `R_model(obs) -> scalar` to predict
     the eventual outcome from each intermediate state. This is essentially
     a value function, but trained with hindsight labels rather than TD
     bootstrapping.
   - Crucially, also train `R_model` on **state transitions**:
     `R_model(obs_t, obs_{t+1}) -> scalar`. This captures which transitions
     are "good moves" (transitions that appear more often in winning games
     than losing games).
   - The transition-based model naturally discovers reward-worthy events:
     enemy knockouts, successful dodges, position improvements.

3. **Phase 3 -- Blended Training.**

   Train PPO with:
   `r_total = alpha * r_curiosity + beta * R_model(obs) + gamma * r_terminal`

   Schedule:
   - Start: alpha = 0.8, beta = 0.2, gamma = 1.0 (mostly curiosity)
   - Midpoint: alpha = 0.3, beta = 0.7, gamma = 1.0 (mostly learned reward)
   - End: alpha = 0.0, beta = 0.0, gamma = 1.0 (pure terminal reward)

   Every K = 100 rollouts, retrain `R_model` on the most recent replay
   buffer. As the agent improves, the distribution of winning states changes,
   so the reward model adapts.

4. **Phase 4 -- Reward Model Distillation.**

   After training completes, analyze what `R_model` has learned:
   - Run feature attribution on `R_model` (same as Approach A) to extract
     which observation features it relies on.
   - Fit a simple linear model to approximate `R_model`'s outputs, producing
     an interpretable reward function.
   - This interpretable approximation is the "discovered reward function."

### Data Structure for the Reward Function

```python
class RNDModule(nn.Module):
    """Random Network Distillation for curiosity."""
    def __init__(self, obs_dim: int = 89, embed_dim: int = 32):
        super().__init__()
        self.target = nn.Sequential(  # Fixed, random weights
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.predictor = nn.Sequential(  # Trainable
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False

    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_embed = self.target(obs)
        pred_embed = self.predictor(obs)
        return ((target_embed - pred_embed) ** 2).mean(dim=-1)


class HindsightRewardModel(nn.Module):
    """Learned reward model from hindsight outcome labeling."""
    def __init__(self, obs_dim: int = 89):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class TransitionRewardModel(nn.Module):
    """Reward model on state transitions."""
    def __init__(self, obs_dim: int = 89):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([obs, next_obs], dim=-1)
        return self.net(combined).squeeze(-1)


@dataclass
class CuriosityRewardState:
    rnd: RNDModule
    hindsight_model: HindsightRewardModel
    transition_model: TransitionRewardModel
    alpha: float  # Curiosity weight
    beta: float   # Hindsight model weight
    gamma: float  # Terminal reward weight
    replay_buffer: list[tuple]  # (obs_sequence, outcome) for hindsight training
```

### Inner/Outer Loop Structure

- **Inner loop:** PPO training with blended reward (standard rollout
  collection, 128 steps per rollout).
- **Middle loop:** Every K = 100 rollouts, retrain `R_model` on accumulated
  replay data. Update alpha/beta/gamma blend weights.
- **Outer loop:** Three phases (curiosity, blended, pure terminal), each
  ~500 rollouts.
- **Post-training:** Distillation of R_model into interpretable form.

### Integration Points

| Component | File | Changes |
|-----------|------|---------|
| `RNDModule` | `src/knockout/reward/curiosity.py` | New file |
| `HindsightRewardModel` | `src/knockout/reward/hindsight.py` | New file |
| `TransitionRewardModel` | `src/knockout/reward/hindsight.py` | Same file |
| Replay buffer for hindsight | `src/knockout/training/hindsight_buffer.py` | New file: stores full episodes with outcomes |
| Blend scheduler | `src/knockout/reward/blend_scheduler.py` | New file |
| Modified PPO trainer | `src/knockout/training/ppo.py` | Accept composite reward, store prev_obs |
| Distillation script | `src/knockout/reward/distill_reward.py` | New file |

### What Makes It Interesting for Research

- **Separation of exploration and exploitation.** Curiosity handles "what
  states exist?" while hindsight handles "which states are good?" This
  separation might discover reward components that pure RL misses because
  the agent never visits the relevant states.
- **The curiosity map is data.** The RND prediction error landscape over
  the state space shows what the agent finds "surprising." This is a
  topographic map of the game's state space, revealing structure that neither
  the agent nor a human designer explicitly programmed.
- **Hindsight labeling avoids credit assignment problems.** Instead of
  bootstrapping value estimates (which are noisy early in training), the
  hindsight model directly labels states with outcomes. This is more
  data-efficient for discovering reward signals.
- **Foundation model data.** The (curiosity_map, hindsight_reward_surface,
  final_policy) tuple is a rich representation of "what a game is about"
  that could teach an LLM to reason about novel games.

### Estimated Implementation Effort

- **Core implementation:** 5-6 days (RND module, hindsight buffer, reward
  models, blend scheduler, PPO integration).
- **Tuning:** 2-3 days (RND embedding dim, model capacity, blend schedule,
  hindsight labeling strategy).
- **Total:** ~1.5 weeks.

---

## Approach E: Meta-Gradient Reward Learning

### How It Works

The reward function is itself a differentiable neural network. The inner loop
trains the policy using rewards from this network. The outer loop updates the
reward network to maximize the true objective (win rate) via meta-gradients
propagated through the inner training process.

**Algorithm:**

1. **Reward network.** A small neural network `R_phi(obs, action, next_obs)`
   parameterized by phi that outputs a scalar reward. This replaces the
   handcrafted reward function entirely.

2. **Inner loop.** For each meta-step:
   - Initialize or continue a PPO policy theta.
   - Collect one rollout using the current policy.
   - Compute rewards using `R_phi` for each transition.
   - Perform one PPO update step on theta using these rewards.
   - This gives theta' = theta - alpha * grad_theta(L_PPO(theta, R_phi)).

3. **Outer loop.** After the inner loop produces theta':
   - Evaluate theta' on a validation batch: play a set of games and compute
     the true sparse reward (win/loss).
   - Compute the meta-loss: `L_meta = -E[r_terminal | theta']` (negative
     expected win rate under the updated policy).
   - Backpropagate through the inner loop update to get
     `dL_meta/d_phi`: how should the reward function change to produce
     policy updates that lead to higher win rates?
   - Update phi via gradient descent: `phi = phi - beta * dL_meta/d_phi`.

4. **Practical considerations.**
   - Full backprop through PPO updates is expensive. Use first-order
     approximation (DICE estimator or implicit differentiation) to make
     this tractable.
   - Alternatively, use Evolution Strategies on phi: perturb phi slightly
     in many random directions, measure which perturbations lead to higher
     win rate after inner training, update phi in the direction of positive
     perturbations.
   - Reward network should be small (2 layers, 32 units) to keep the
     meta-gradient tractable and prevent reward overfitting.

5. **Reward interpretation.** After training, analyze what `R_phi` has
   learned:
   - Feature attribution on R_phi (which inputs matter?).
   - Probe with synthetic states: vary one feature while holding others
     constant, plot the reward response. This reveals the learned reward
     landscape.
   - Fit a linear approximation for interpretability.

### Data Structure for the Reward Function

```python
class MetaRewardNetwork(nn.Module):
    """Small differentiable reward function."""
    def __init__(self, obs_dim: int = 89):
        super().__init__()
        # Takes current obs only (simplest version)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class MetaRewardLearner:
    reward_net: MetaRewardNetwork
    reward_optimizer: optim.Adam  # Outer loop optimizer for phi
    policy: ActorCritic           # Inner loop policy
    policy_optimizer: optim.Adam  # Inner loop optimizer for theta

    inner_steps: int = 5          # PPO updates per meta-step
    outer_lr: float = 1e-3        # Learning rate for reward network
    n_perturbations: int = 20     # For ES-based meta-gradient (if used)
    perturbation_scale: float = 0.01
```

### Inner/Outer Loop Structure

- **Innermost loop:** PPO rollout collection + training step (same as
  existing PPO, but rewards come from R_phi).
- **Inner loop:** K = 5 PPO update steps (short training segment).
- **Outer loop:** Evaluate updated policy, compute meta-gradient, update
  R_phi. Run for N_meta = 500 meta-steps.
- Each meta-step: K inner PPO steps + 1 evaluation rollout + 1 reward
  network update.

### Integration Points

| Component | File | Changes |
|-----------|------|---------|
| `MetaRewardNetwork` | `src/knockout/reward/meta_reward.py` | New file |
| `MetaRewardLearner` | `src/knockout/reward/meta_reward.py` | Same file |
| ES-based meta-gradient | `src/knockout/reward/meta_es.py` | New file (if using ES instead of backprop) |
| Modified PPO | `src/knockout/training/ppo.py` | Return policy gradient information for meta-gradient |
| Reward probing | `src/knockout/reward/probe_reward.py` | New file: synthetic state analysis |
| Meta-training driver | `src/knockout/training/meta_train.py` | New file: orchestrates inner/outer loop |

### What Makes It Interesting for Research

- **End-to-end reward learning.** The reward function is learned with
  end-to-end gradients through the training process. No human specifies
  what features to attend to or what behaviors to reward.
- **Reward function as a learned object.** The parameters phi encode
  everything the system has learned about "what makes a good reward for
  this game." This is a compact, transferable representation.
- **Meta-learning theory.** This connects to MAML, learned loss functions,
  and optimal reward shaping theory. The meta-gradient tells us the
  "gradient of win rate with respect to reward shape," which is
  theoretically novel data.
- **Foundation model data.** The trajectory of (R_phi_parameters,
  resulting_policy_behavior, win_rate) across meta-steps is a dataset of
  "how reward functions influence learning dynamics." An LLM trained on
  this could predict how changing a reward function will affect agent
  behavior.
- **Comparison to theory.** The optimal reward shaping theorem (Ng et al.,
  1999) says that potential-based shaping is the only form that preserves
  optimal policy. We can check whether the learned R_phi converges to a
  potential-based form.

### Estimated Implementation Effort

- **Core implementation:** 5-7 days (meta-reward network, ES-based outer
  loop, inner PPO integration, evaluation harness).
- **Tuning:** 3-4 days (inner/outer learning rates, number of inner steps,
  ES hyperparameters, reward network capacity).
- **Total:** ~2 weeks. This is the most complex approach.

---

## Approach F: Contrastive Trajectory Reward Mining

### How It Works

This approach discovers reward components by contrasting winning and losing
trajectories. It asks: "What is systematically different about the
observations in games we won versus games we lost?" and converts those
differences into reward signals.

**Algorithm:**

1. **Data collection.** Play N = 1000 games with a random policy (or a
   mildly trained one). For each game, store the full trajectory:
   `[(obs_0, action_0), (obs_1, action_1), ..., (obs_T, action_T)]`
   plus the outcome (win/loss/draw).

2. **Trajectory statistics.** For each game, compute per-feature statistics
   across the trajectory:
   - Mean, min, max, std, trend (linear regression slope) for each of the
     89 observation dimensions.
   - This produces a ~445-dim "trajectory fingerprint" for each game.

3. **Contrastive analysis.** Compare win-trajectory fingerprints to
   loss-trajectory fingerprints:
   - For each statistic, compute the effect size (Cohen's d) between wins
     and losses.
   - Features with large effect sizes are candidate reward components.
   - Example discoveries:
     - "In winning games, ego_dist_to_edge.mean is higher" -> reward staying
       away from edge
     - "In winning games, enemy_dist_to_edge.min is lower" -> reward pushing
       enemies near edge
     - "In winning games, ego_speed.mean is lower in early rounds" -> reward
       staying still early (don't waste energy)

4. **Reward construction.** For each discovered feature with effect size
   above a threshold:
   - If the feature is a per-step observation value (not a trajectory
     aggregate), create a direct reward: `r_i = sign * w_i * obs[idx]`
   - If the feature is a trajectory statistic (like trend), create a
     reward based on the observation delta: `r_i = w_i * (obs[idx] - prev_obs[idx])`
   - Weight `w_i` is proportional to effect size.

5. **Iterative refinement.** Train PPO with the discovered reward. After M
   rollouts, collect new trajectories from the improved policy and repeat
   the contrastive analysis. New features may emerge as the agent's behavior
   changes (e.g., once the agent stops falling off the edge, "push enemies
   off" becomes the dominant differentiator).

6. **Feature interaction discovery.** In later iterations, extend the
   contrastive analysis to feature pairs (e.g., "in winning games,
   ego_speed * enemy_dist_to_ego is low when enemy_dist_to_edge is low"
   = "slow down when near an enemy who is near the edge"). This discovers
   conditional reward components.

### Data Structure for the Reward Function

```python
@dataclass
class ContrastiveRewardComponent:
    feature_index: int            # Primary observation index
    feature_name: str
    statistic: str                # "value", "delta", "product_with_{idx}"
    sign: float                   # +1 or -1
    weight: float                 # Proportional to effect size
    effect_size: float            # Cohen's d between win and loss distributions
    discovered_at_iteration: int
    secondary_index: int | None = None  # For interaction features

class ContrastiveRewardFunction:
    components: list[ContrastiveRewardComponent]
    terminal_weight: float = 1.0

    def compute(self, obs: np.ndarray, prev_obs: np.ndarray | None) -> float:
        r = 0.0
        for c in self.components:
            if c.statistic == "value":
                r += c.sign * c.weight * obs[c.feature_index]
            elif c.statistic == "delta" and prev_obs is not None:
                r += c.weight * (obs[c.feature_index] - prev_obs[c.feature_index])
            elif c.statistic.startswith("product_with_") and c.secondary_index:
                r += c.weight * obs[c.feature_index] * obs[c.secondary_index]
        return r

class TrajectoryDatabase:
    wins: list[list[np.ndarray]]   # List of winning trajectories
    losses: list[list[np.ndarray]] # List of losing trajectories
    feature_names: list[str]       # 89 feature names for interpretability
```

### Inner/Outer Loop Structure

- **Data collection:** Play 1000 games (fast with TensorVecEnv).
- **Analysis:** Compute trajectory statistics and contrastive effect sizes
  (pure numpy, very fast).
- **Inner loop:** PPO training with discovered reward (200 rollouts).
- **Outer loop:** Collect new data, re-analyze, update reward. 10-20
  iterations.

### Integration Points

| Component | File | Changes |
|-----------|------|---------|
| `ContrastiveRewardFunction` | `src/knockout/reward/contrastive_reward.py` | New file |
| `TrajectoryDatabase` | `src/knockout/reward/trajectory_db.py` | New file |
| Contrastive analyzer | `src/knockout/reward/contrastive_analysis.py` | New file: effect size computation |
| Trajectory collector | `src/knockout/reward/trajectory_collector.py` | New file: plays games and stores trajectories |
| Feature name registry | `src/knockout/reward/feature_names.py` | New file: maps obs indices to names |
| Modified PPO | `src/knockout/training/ppo.py` | Accept reward_fn with prev_obs |
| Outer loop | `src/knockout/training/contrastive_discovery.py` | New file |

### What Makes It Interesting for Research

- **Statistically grounded discovery.** Effect sizes give a principled,
  quantitative measure of feature importance. No neural network needed
  for the discovery step (just statistics).
- **Interaction discovery.** Finding feature products that distinguish wins
  from losses reveals conditional strategies ("target enemies near the
  edge" is a product of enemy_proximity and enemy_edge_distance).
- **Human-readable output.** Each component has a name, effect size, and
  sign, making the discovered reward fully interpretable. The output
  reads like game analysis: "The strongest predictor of winning is having
  enemies with low distance-to-edge (d=1.3), followed by maintaining
  high ego distance-to-edge (d=0.9)."
- **Foundation model data.** The contrastive analysis reports are natural
  language descriptions of "what winning looks like" in this game. This
  is directly useful for training an LLM to understand and reason about
  game strategy.
- **Low computational cost.** The contrastive analysis is cheap (just
  statistics). The expensive part is playing the games, which is needed
  regardless.

### Estimated Implementation Effort

- **Core implementation:** 3-4 days (trajectory DB, contrastive analysis,
  reward construction, feature naming).
- **Interaction discovery:** 1-2 days (pairwise feature analysis,
  combinatorial explosion management).
- **Total:** ~1 week. This is the simplest approach conceptually.

---

## Cross-Cutting Design: The Reward Function Interface

All six approaches need the same interface to plug into PPO training. Define
a common protocol:

```python
# src/knockout/reward/base.py

from typing import Protocol, runtime_checkable
import numpy as np
import torch


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for all reward functions (discovered or handcrafted)."""

    def compute(
        self,
        obs: np.ndarray,
        prev_obs: np.ndarray | None,
        terminal_reward: float,
        info: dict,
    ) -> float:
        """Compute reward for a single transition."""
        ...

    def compute_batch(
        self,
        obs: torch.Tensor,
        prev_obs: torch.Tensor | None,
        terminal_rewards: torch.Tensor,
        infos: list[dict],
    ) -> torch.Tensor:
        """Compute rewards for a batch of transitions (tensor backend)."""
        ...

    def describe(self) -> str:
        """Return a human-readable description of the reward function."""
        ...
```

And a common modification to PPO:

```python
# In PPOTrainer.__init__, add:
self.reward_fn: RewardFunction | None = None

# In collect_rollout_vec, after computing flat_rewards:
if self.reward_fn is not None:
    # Apply discovered reward function
    shaped_rewards = self.reward_fn.compute_batch(
        obs=obs_t,
        prev_obs=prev_obs_t if step_idx > 0 else None,
        terminal_rewards=torch.as_tensor(flat_rewards),
        infos=[{} for _ in range(len(flat_rewards))],
    )
    flat_rewards = shaped_rewards.cpu().numpy()
```

---

## Cross-Cutting Design: The Discovery Log

Every approach should produce a structured log of the discovery process:

```python
# src/knockout/reward/discovery_log.py

@dataclass
class DiscoveryEvent:
    """A single event in the reward discovery process."""
    timestamp: float
    iteration: int
    event_type: str  # "component_added", "component_removed", "weight_changed",
                     # "reward_rewritten", "generation_evolved", "model_retrained"
    description: str  # Human-readable description
    reward_function_snapshot: str  # Serialized reward function at this point
    metrics: dict  # Win rate, loss, entropy, etc. at this point
    approach: str  # "attention", "llm", "evolutionary", etc.

class DiscoveryLog:
    events: list[DiscoveryEvent]
    approach: str

    def to_jsonl(self, path: str) -> None:
        """Write log as JSON Lines (one event per line)."""
        ...

    def to_narrative(self) -> str:
        """Generate a natural language narrative of the discovery process."""
        ...
```

The `to_narrative()` method produces text like:
```
Iteration 1: Starting with sparse win/loss reward only. Win rate: 33%.
Iteration 5: Discovered that ego_dist_to_edge (feature 5) predicts winning.
  Added reward component: +0.15 * obs[5]. Win rate improved to 41%.
Iteration 12: Discovered that enemy1_dist_to_edge (feature 47) predicts
  losing when high. Added component: -0.10 * obs[47]. Win rate: 48%.
...
```

This narrative is directly usable as foundation model training data.

---

## Comparison Matrix

| Criterion | A: Attention | B: LLM | C: Evolutionary | D: Curiosity | E: Meta-Grad | F: Contrastive |
|-----------|-------------|--------|-----------------|-------------|-------------|---------------|
| **Blank-slate purity** | High (starts with V network gradients) | High (LLM has no game-specific priors) | Medium (requires predefined feature basis) | Highest (curiosity has zero task knowledge) | Highest (reward network starts random) | Medium (requires feature naming) |
| **Interpretability** | High (named features with weights) | Highest (Python code with comments) | High (weight vector over named features) | Low (neural reward model) | Low (neural reward network) | Highest (effect sizes with names) |
| **Compute cost** | Low | Medium (LLM API costs) | High (P * E * G rollouts) | Medium | High (meta-gradient overhead) | Low |
| **Implementation complexity** | Medium | Medium-High | Medium | High | Very High | Low |
| **Robustness to reward hacking** | Medium | Medium (LLM can reason about it) | High (evolution selects against it) | Medium | Low (reward network can overfit) | High (based on actual win correlation) |
| **Foundation model data quality** | Good (feature discovery timeline) | Excellent (LLM reasoning traces) | Good (phylogenetic tree) | Good (curiosity landscape) | Moderate (parameter trajectories) | Excellent (statistical game analysis) |
| **Novel research contribution** | Moderate | High (LLM-as-reward-designer) | Moderate (well-studied) | Moderate | High (meta-gradient theory) | Moderate |

---

## Recommended Implementation Order

1. **Start with F (Contrastive Trajectory Mining).** Lowest complexity,
   lowest compute cost, produces immediately interpretable results. The
   contrastive analysis gives a baseline understanding of "what winning
   looks like" that informs all other approaches. Implement the common
   `RewardFunction` protocol and `DiscoveryLog` here. **~1 week.**

2. **Then A (Feature Attention Discovery).** Builds naturally on the PPO
   infrastructure. The gradient attribution provides a different lens on
   the same question (correlation vs. causation comparison between F and
   A is itself interesting). **~1 week.**

3. **Then B (LLM Reward Architect).** The contrastive analysis from F and
   attention data from A can be fed to the LLM as context, making its
   reward designs better-informed. This produces the richest training data
   for the project's foundation model goal. **~1.5 weeks.**

4. **Then C (Evolutionary) or D (Curiosity) based on compute budget.**
   Evolutionary is better if you have multi-GPU; curiosity is better for
   single-GPU with patience. **~1-1.5 weeks each.**

5. **E (Meta-Gradient) last.** Most complex, most theoretically interesting,
   but depends on solid infrastructure from earlier approaches. **~2 weeks.**

---

## Data Products for Foundation Model Training

Each approach generates distinct data products:

| Approach | Primary Data Product | Format | Volume |
|----------|---------------------|--------|--------|
| A | Feature importance timeline | `[(rollout, feature_name, attribution_score, weight)]` | ~1K entries |
| B | LLM reward design sessions | `[(prompt, reasoning, code, result_metrics)]` | ~50 sessions |
| C | Reward function phylogeny | `[(generation, individual, weights, fitness, parents)]` | ~600 individuals |
| D | Curiosity-reward landscape | `[(state_region, novelty_score, hindsight_value)]` | ~100K states |
| E | Meta-gradient trajectory | `[(meta_step, reward_params, policy_performance)]` | ~500 steps |
| F | Contrastive game analysis | `[(iteration, feature, effect_size, description)]` | ~200 entries |

All of these can be converted to natural language narratives by the
`DiscoveryLog.to_narrative()` method, producing training data for teaching
an LLM to reason about game strategy, reward design, and agent behavior.

The most valuable single data product is the LLM Reward Architect sessions
(Approach B), because they directly demonstrate the reasoning process of
designing reward functions from game observations. This is exactly the kind
of reasoning we want a foundation model to learn.

---

## Appendix: Observation Index Reference

For quick reference when implementing reward features:

```
Ego penguin (indices 0-13):
  0: pos_x    1: pos_y    2: vel_x    3: vel_y
  4: dist_center  5: dist_edge  6: speed  7: heading
  8: alive    9-13: relative (all 0 for ego)

Ally 1 (14-27), Ally 2 (28-41):
  Same 14 features, sorted by distance to ego

Enemy 1 (42-55), Enemy 2 (56-69), Enemy 3 (70-83):
  Same 14 features, sorted by distance to ego

Key enemy indices:
  42: enemy1.pos_x    47: enemy1.dist_edge    50: enemy1.alive
  55: enemy1.dist_to_ego
  56: enemy2.pos_x    61: enemy2.dist_edge    64: enemy2.alive
  69: enemy2.dist_to_ego
  70: enemy3.pos_x    75: enemy3.dist_edge    78: enemy3.alive
  83: enemy3.dist_to_ego

Global (84-88):
  84: team_a_alive/3  85: team_b_alive/3
  86: ego_team_alive/3  87: opp_team_alive/3
  88: timestep
```
