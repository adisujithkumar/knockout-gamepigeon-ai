"""LLM Reward Discovery via Claude CLI.

Uses the `claude` CLI binary (already authenticated) instead of the
Anthropic Python SDK to iterate on reward functions.  Each iteration:

    1. Collect game stats (win/loss feature distributions)
    2. Build a zero-knowledge prompt with statistics
    3. Call `claude --print` via subprocess
    4. Extract + validate the reward function
    5. Train PPO with the shaped reward
    6. Save all artifacts for research reproducibility
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from knockout.core.config import GameConfig, DEFAULTS
from knockout.env.tensor_env import TensorVecEnv
from knockout.reward.contrastive import (
    FEATURE_NAMES,
    TrajectoryCollector,
)
from knockout.reward.llm_architect import (
    RewardFunctionSandbox,
    build_obs_description,
)
from knockout.training.ppo import PPOTrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_INITIAL_PROMPT = """\
You are designing a reward function for a reinforcement learning agent.

The agent receives an 89-dimensional observation vector each step and outputs
a 2-dimensional continuous action. The only outcome signal is binary: +1 (win)
or -1 (loss) at the end of each episode.

{obs_description}

Here are statistics from {n_episodes} recent games.
For each feature, the mean and standard deviation are shown separately for
episodes that ended in a WIN vs a LOSS:

{stats_table}

Write a Python reward function that will help the agent learn to win.
The function signature must be:
    def reward(obs: torch.Tensor) -> torch.Tensor
where obs has shape (B, 89) and the return has shape (B,).

Only use torch operations (import torch). Reason about what the statistics
tell you about what matters for winning, then write the function inside a
single ```python ... ``` block."""

_EVOLVE_PROMPT = """\
You are designing a reward function for a reinforcement learning agent.

The agent receives an 89-dimensional observation vector each step and outputs
a 2-dimensional continuous action. The only outcome signal is binary: +1 (win)
or -1 (loss) at the end of each episode.

{obs_description}

Previous reward function:
```python
{previous_code}
```

Results after training with it for {training_steps} steps:
- Win rate vs random opponent: {win_rate:.1%}
- Average episode length: {avg_episode_length:.1f} rounds
- Win-steps collected: {n_win_steps}, Loss-steps collected: {n_loss_steps}

Updated statistics from {n_episodes} recent games:

{stats_table}

Analyze what improved and what did not. Then write an improved reward
function. The function signature must be:
    def reward(obs: torch.Tensor) -> torch.Tensor
where obs has shape (B, 89) and the return has shape (B,).

Only use torch operations (import torch). Write the function inside a
single ```python ... ``` block."""


def _build_stats_table(
    observations: np.ndarray,
    outcomes: np.ndarray,
    top_n: int = 25,
) -> str:
    """Build a text table comparing feature stats for wins vs losses."""
    win_mask = outcomes > 0
    loss_mask = outcomes < 0

    win_obs = observations[win_mask]
    loss_obs = observations[loss_mask]

    if len(win_obs) < 2 or len(loss_obs) < 2:
        return "(insufficient data: need both win and loss episodes)"

    win_mean = win_obs.mean(axis=0)
    win_std = win_obs.std(axis=0)
    loss_mean = loss_obs.mean(axis=0)
    loss_std = loss_obs.std(axis=0)

    # Rank features by |mean_win - mean_loss|
    diffs = np.abs(win_mean - loss_mean)
    ranked_indices = np.argsort(-diffs)

    lines = [
        f"{'Idx':>4}  {'Feature':<35}  {'WIN mean+/-std':>18}  {'LOSS mean+/-std':>18}  {'|diff|':>7}",
        "-" * 90,
    ]
    for idx in ranked_indices[:top_n]:
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feat_{idx}"
        lines.append(
            f"{idx:4d}  {name:<35}  "
            f"{win_mean[idx]:+7.3f} +/- {win_std[idx]:.3f}  "
            f"{loss_mean[idx]:+7.3f} +/- {loss_std[idx]:.3f}  "
            f"{diffs[idx]:7.4f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude CLI caller
# ---------------------------------------------------------------------------

def call_claude_cli(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    timeout: int = 180,
) -> str:
    """Call the claude CLI with --print and return the response text."""
    logger.info("Calling claude CLI (model=%s, timeout=%ds)...", model, timeout)
    t0 = time.time()

    result = subprocess.run(
        ["claude", "--print", "--model", model],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    elapsed = time.time() - t0
    logger.info("Claude CLI returned in %.1fs (exit code %d)", elapsed, result.returncode)

    if result.returncode != 0:
        logger.error("claude CLI stderr: %s", result.stderr[:500])
        raise RuntimeError(
            f"claude CLI failed with exit code {result.returncode}: "
            f"{result.stderr[:300]}"
        )

    return result.stdout


def extract_code(response: str) -> str | None:
    """Extract the last ```python ... ``` block from the response."""
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


# ---------------------------------------------------------------------------
# Stats collection (lightweight wrapper around TrajectoryCollector)
# ---------------------------------------------------------------------------

def collect_trajectory_stats(
    env: TensorVecEnv,
    policy: Any,
    num_episodes: int,
    device: str,
    config: GameConfig,
) -> dict[str, Any]:
    """Collect trajectories and compute summary statistics.

    Returns a dict with raw arrays and summary metrics.
    """
    collector = TrajectoryCollector(config=config)
    data = collector.collect(
        env=env,
        policy=policy,
        num_episodes=num_episodes,
        device=device,
    )

    obs = data["observations"]
    outcomes = data["outcomes"]
    n_wins = int((outcomes > 0).sum())
    n_losses = int((outcomes < 0).sum())
    n_episodes_actual = int(data["episode_ids"].max()) + 1 if len(data["episode_ids"]) > 0 else 0

    win_rate = n_wins / max(n_wins + n_losses, 1)

    # Per-episode lengths
    ep_ids = data["episode_ids"]
    unique_eps = np.unique(ep_ids)
    ep_lengths = [int((ep_ids == e).sum()) for e in unique_eps]
    avg_ep_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0

    return {
        "observations": obs,
        "outcomes": outcomes,
        "n_episodes": n_episodes_actual,
        "n_win_steps": n_wins,
        "n_loss_steps": n_losses,
        "win_rate": win_rate,
        "avg_episode_length": avg_ep_len,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_discovery(
    num_iterations: int = 3,
    steps_per_iteration: int = 500_000,
    collection_episodes: int = 200,
    num_envs: int = 64,
    device: str = "cpu",
    model: str = "claude-sonnet-4-20250514",
    output_dir: str = "runs/llm_cli",
    cli_timeout: int = 180,
) -> list[dict[str, Any]]:
    """Run the full LLM reward discovery loop."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sandbox = RewardFunctionSandbox()
    config = DEFAULTS
    obs_description = build_obs_description()

    current_code: str | None = None
    current_reward_fn = None
    current_trainer: PPOTrainer | None = None
    history: list[dict[str, Any]] = []

    for iteration in range(num_iterations):
        iter_dir = out / f"iteration_{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 60)
        logger.info("=== Iteration %d / %d ===", iteration, num_iterations - 1)
        logger.info("=" * 60)
        t0 = time.time()

        # -----------------------------------------------------------
        # 1. Collect game stats with current policy
        # -----------------------------------------------------------
        logger.info("Collecting %d episodes for statistics...", collection_episodes)

        env = TensorVecEnv(num_envs=num_envs, config=config, device=device)

        # Use trained policy from the previous iteration, or None (random)
        policy = current_trainer.agent.network if current_trainer is not None else None

        stats = collect_trajectory_stats(
            env=env,
            policy=policy,
            num_episodes=collection_episodes,
            device=device,
            config=config,
        )
        env.close()

        logger.info(
            "Stats: %d episodes, %d win-steps, %d loss-steps, win_rate=%.1f%%",
            stats["n_episodes"], stats["n_win_steps"], stats["n_loss_steps"],
            stats["win_rate"] * 100,
        )

        stats_table = _build_stats_table(stats["observations"], stats["outcomes"])

        # -----------------------------------------------------------
        # 2. Build prompt
        # -----------------------------------------------------------
        if current_code is None:
            prompt = _INITIAL_PROMPT.format(
                obs_description=obs_description,
                n_episodes=stats["n_episodes"],
                stats_table=stats_table,
            )
        else:
            prompt = _EVOLVE_PROMPT.format(
                obs_description=obs_description,
                previous_code=current_code,
                training_steps=steps_per_iteration,
                win_rate=stats["win_rate"],
                avg_episode_length=stats["avg_episode_length"],
                n_win_steps=stats["n_win_steps"],
                n_loss_steps=stats["n_loss_steps"],
                n_episodes=stats["n_episodes"],
                stats_table=stats_table,
            )

        # Save prompt
        (iter_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        logger.info("Prompt saved (%d chars)", len(prompt))

        # -----------------------------------------------------------
        # 3. Call Claude CLI
        # -----------------------------------------------------------
        max_retries = 3
        response = None
        code = None

        for attempt in range(max_retries):
            try:
                response = call_claude_cli(
                    prompt=prompt, model=model, timeout=cli_timeout,
                )
                (iter_dir / "response.txt").write_text(response, encoding="utf-8")

                code = extract_code(response)
                if code is None:
                    logger.warning(
                        "No Python code block found in response (attempt %d/%d)",
                        attempt + 1, max_retries,
                    )
                    continue

                # Validate
                reward_fn = sandbox.compile_reward_function(code)
                sandbox.validate_output(reward_fn)
                logger.info("Reward function validated successfully")
                break

            except (subprocess.TimeoutExpired, RuntimeError, ValueError) as e:
                logger.warning(
                    "Attempt %d/%d failed: %s", attempt + 1, max_retries, e,
                )
                code = None
                continue

        if code is None:
            # Fallback: simple center-seeking reward
            logger.warning("All attempts failed, using fallback reward function")
            code = (
                "import torch\n\n"
                "def reward(obs: torch.Tensor) -> torch.Tensor:\n"
                "    # Fallback: encourage staying near center (feature 4 = dist_from_center)\n"
                "    return -obs[:, 4] * 0.3\n"
            )
            reward_fn = sandbox.compile_reward_function(code)
            sandbox.validate_output(reward_fn)
            if response is None:
                response = "(all LLM attempts failed, using fallback)"

        # Save reward function code
        (iter_dir / "reward_function.py").write_text(code, encoding="utf-8")
        current_code = code
        current_reward_fn = reward_fn

        # -----------------------------------------------------------
        # 4. Train PPO with shaped reward
        # -----------------------------------------------------------
        logger.info(
            "Training PPO for %d steps with shaped reward...", steps_per_iteration,
        )

        trainer = PPOTrainer(
            config=config,
            device=device,
            num_envs=num_envs,
            backend="tensor",
        )
        current_trainer = trainer

        train_env = TensorVecEnv(num_envs=num_envs, config=config, device=device)

        steps_per_rollout = trainer.rollout_steps * num_envs * 3
        n_rollouts = max(steps_per_iteration // steps_per_rollout, 1)

        train_metrics: list[dict[str, float]] = []
        for rollout_idx in range(n_rollouts):
            buffer = trainer.collect_rollout_vec(train_env)

            # Inject shaped reward
            obs_tensor = torch.as_tensor(
                buffer.observations[: buffer.pos],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                shaped = current_reward_fn(obs_tensor).cpu().numpy()
            buffer.rewards[: buffer.pos] += shaped

            # Recompute GAE with updated rewards
            if hasattr(buffer, "compute_gae_structured"):
                buffer.compute_gae_structured(
                    gamma=trainer.gamma, gae_lambda=trainer.gae_lambda,
                )
            else:
                buffer.compute_gae(
                    gamma=trainer.gamma, gae_lambda=trainer.gae_lambda,
                )

            metrics = trainer.train_step(buffer)
            train_metrics.append(metrics)

            if (rollout_idx + 1) % 10 == 0 or rollout_idx == 0:
                logger.info(
                    "  rollout %d/%d  policy_loss=%.4f  value_loss=%.4f",
                    rollout_idx + 1, n_rollouts,
                    metrics.get("policy_loss", 0),
                    metrics.get("value_loss", 0),
                )

        train_env.close()

        # Aggregate training metrics
        agg_metrics = {}
        if train_metrics:
            agg_metrics = {
                k: float(np.mean([m[k] for m in train_metrics if k in m]))
                for k in train_metrics[0]
            }

        # Save agent checkpoint
        ckpt_path = iter_dir / "agent_checkpoint.pt"
        trainer.agent.save(ckpt_path)
        logger.info("Checkpoint saved: %s", ckpt_path)

        # Save metrics
        elapsed = time.time() - t0
        iteration_record = {
            "iteration": iteration,
            "elapsed_seconds": round(elapsed, 1),
            "stats": {
                "n_episodes": stats["n_episodes"],
                "win_rate": round(stats["win_rate"], 4),
                "avg_episode_length": round(stats["avg_episode_length"], 1),
                "n_win_steps": stats["n_win_steps"],
                "n_loss_steps": stats["n_loss_steps"],
            },
            "training": {
                "steps": steps_per_iteration,
                "n_rollouts": n_rollouts,
                **{k: round(v, 6) for k, v in agg_metrics.items()},
            },
            "reward_code_lines": len(code.strip().splitlines()),
            "used_fallback": "(fallback)" in (response or ""),
        }
        (iter_dir / "metrics.json").write_text(
            json.dumps(iteration_record, indent=2), encoding="utf-8",
        )
        history.append(iteration_record)

        logger.info(
            "Iteration %d complete in %.1fs. Win rate before training: %.1f%%",
            iteration, elapsed, stats["win_rate"] * 100,
        )

    # Save summary
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Reward Discovery via Claude CLI",
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of discovery iterations (default: 3)",
    )
    parser.add_argument(
        "--steps-per-iteration", type=int, default=500_000,
        help="PPO training steps per iteration (default: 500K)",
    )
    parser.add_argument(
        "--collection-episodes", type=int, default=200,
        help="Episodes to collect for statistics (default: 200)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=64,
        help="Parallel environments (default: 64)",
    )
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4-20250514",
        help="Claude model for --print calls (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/llm_cli",
        help="Output directory (default: runs/llm_cli)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: cpu, cuda, or auto (default: auto)",
    )
    parser.add_argument(
        "--cli-timeout", type=int, default=180,
        help="Timeout for each claude CLI call in seconds (default: 180)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    logger.info("LLM Reward Discovery (Claude CLI backend)")
    logger.info("  model=%s, device=%s, num_envs=%d", args.model, device, args.num_envs)
    logger.info("  iterations=%d, steps_per_iter=%d", args.iterations, args.steps_per_iteration)
    logger.info("  collection_episodes=%d", args.collection_episodes)
    logger.info("  output_dir=%s", args.output_dir)

    history = run_discovery(
        num_iterations=args.iterations,
        steps_per_iteration=args.steps_per_iteration,
        collection_episodes=args.collection_episodes,
        num_envs=args.num_envs,
        device=device,
        model=args.model,
        output_dir=args.output_dir,
        cli_timeout=args.cli_timeout,
    )

    logger.info("Done. %d iterations completed.", len(history))
    for h in history:
        logger.info(
            "  iter %d: win_rate=%.1f%%, elapsed=%.1fs",
            h["iteration"],
            h["stats"]["win_rate"] * 100,
            h["elapsed_seconds"],
        )


if __name__ == "__main__":
    main()
