"""Speed benchmarks comparing TensorPhysicsEngine vs Pymunk.

Usage:
    .venv/bin/python benchmarks/bench_tensor_physics.py
"""

from __future__ import annotations

import time
import sys

import numpy as np
import torch

from knockout.core.config import DEFAULTS
from knockout.core.physics_engine import PhysicsEngine
from knockout.core.tensor_physics import TensorPhysicsEngine
from knockout.training.vec_env import SingleTeamVecEnv
from knockout.env.tensor_env import TensorVecEnv


def _divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ======================================================================
# Benchmark 1: Single game — TensorPhysics (CPU, batch=1) vs Pymunk
# ======================================================================

def bench_single_game(rounds: int = 10) -> None:
    _divider(f"Single game ({rounds} rounds): Tensor CPU vs Pymunk")

    rng = np.random.default_rng(42)

    # Generate actions upfront
    all_angles = rng.uniform(0, 360, size=(rounds, 6))
    all_powers = rng.uniform(0, 400, size=(rounds, 6))

    # --- Pymunk ---
    t0 = time.perf_counter()
    pymunk_eng = PhysicsEngine(seed=0)
    pymunk_eng.initialize_game()
    for r in range(rounds):
        actions = {}
        for i in range(6):
            aid = f"penguin_{i}"
            if aid in pymunk_eng.penguins and pymunk_eng.penguins[aid].alive:
                actions[aid] = (all_angles[r, i], all_powers[r, i])
        pymunk_eng.apply_actions(actions)
        pymunk_eng.step_until_settled()
    pymunk_time = time.perf_counter() - t0

    # --- Tensor CPU ---
    t0 = time.perf_counter()
    tengine = TensorPhysicsEngine(device="cpu")
    pos, vel, alive = tengine.reset(1)
    for r in range(rounds):
        t_actions = torch.zeros(1, 6, 2)
        for i in range(6):
            t_actions[0, i, 0] = all_angles[r, i]
            t_actions[0, i, 1] = all_powers[r, i]
        vel = tengine.apply_impulses(vel, t_actions, alive)
        pos, vel, alive, _ = tengine.step_until_settled(
            pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH
        )
    tensor_cpu_time = time.perf_counter() - t0

    print(f"  Pymunk:      {pymunk_time:.4f}s")
    print(f"  Tensor CPU:  {tensor_cpu_time:.4f}s")
    print(f"  Ratio:       {pymunk_time / tensor_cpu_time:.2f}x (>1 = tensor faster)")


# ======================================================================
# Benchmark 2: Batched — TensorVecEnv (GPU, N=1000) vs SingleTeamVecEnv (8 envs)
# ======================================================================

def bench_batched(tensor_batch: int = 1000, pymunk_batch: int = 8, rounds: int = 10) -> None:
    _divider(f"Batched: TensorVecEnv (GPU, N={tensor_batch}) vs SingleTeamVecEnv (N={pymunk_batch})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  [WARNING] No CUDA available, running tensor on CPU")

    rng = np.random.default_rng(7)

    # --- SingleTeamVecEnv (Pymunk-based) ---
    t0 = time.perf_counter()
    vec_env = SingleTeamVecEnv(num_envs=pymunk_batch)
    vec_env.reset()
    for _ in range(rounds):
        actions = rng.uniform(
            low=[0, 0], high=[360, 400], size=(pymunk_batch, 3, 2)
        ).astype(np.float32)
        vec_env.step(actions)
    pymunk_time = time.perf_counter() - t0
    vec_env.close()
    pymunk_steps_per_sec = pymunk_batch * rounds / pymunk_time

    # --- TensorVecEnv (GPU/CPU) ---
    # Warm-up step (CUDA kernel compilation)
    tenv = TensorVecEnv(num_envs=tensor_batch, device=device)
    tenv.reset()
    warmup_actions = rng.uniform(
        low=[0, 0], high=[360, 400], size=(tensor_batch, 3, 2)
    ).astype(np.float32)
    tenv.step(warmup_actions)

    t0 = time.perf_counter()
    tenv.reset()
    for _ in range(rounds):
        actions = rng.uniform(
            low=[0, 0], high=[360, 400], size=(tensor_batch, 3, 2)
        ).astype(np.float32)
        tenv.step(actions)
    if device == "cuda":
        torch.cuda.synchronize()
    tensor_time = time.perf_counter() - t0
    tenv.close()
    tensor_steps_per_sec = tensor_batch * rounds / tensor_time

    print(f"  SingleTeamVecEnv ({pymunk_batch} envs): {pymunk_time:.4f}s  ({pymunk_steps_per_sec:.0f} env-steps/s)")
    print(f"  TensorVecEnv ({tensor_batch} envs, {device}): {tensor_time:.4f}s  ({tensor_steps_per_sec:.0f} env-steps/s)")
    print(f"  Throughput ratio: {tensor_steps_per_sec / pymunk_steps_per_sec:.1f}x")

    # --- Scaling test ---
    for n in [256, 1024, 4096]:
        tenv2 = TensorVecEnv(num_envs=n, device=device)
        tenv2.reset()
        # warmup
        wa = rng.uniform(low=[0, 0], high=[360, 400], size=(n, 3, 2)).astype(np.float32)
        tenv2.step(wa)

        t0 = time.perf_counter()
        tenv2.reset()
        for _ in range(rounds):
            a = rng.uniform(low=[0, 0], high=[360, 400], size=(n, 3, 2)).astype(np.float32)
            tenv2.step(a)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tenv2.close()
        print(f"  TensorVecEnv ({n:>5} envs, {device}): {dt:.4f}s  ({n * rounds / dt:.0f} env-steps/s)")


# ======================================================================
# Benchmark 3: Pymunk-vs-Tensor comparison (correctness)
# ======================================================================

def bench_comparison(games: int = 20, rounds: int = 8) -> None:
    _divider(f"Correctness comparison: {games} games x {rounds} rounds")

    rng = np.random.default_rng(999)
    alive_matches = 0
    total_checks = 0

    for g in range(games):
        # Pymunk
        pymunk_eng = PhysicsEngine(seed=g)
        pymunk_eng.initialize_game()

        # Tensor
        tengine = TensorPhysicsEngine(device="cpu")
        pos, vel, alive = tengine.reset(1)

        for r in range(rounds):
            angles = rng.uniform(0, 360, size=6)
            powers = rng.uniform(0, 400, size=6)

            # Pymunk
            pm_actions = {}
            for i in range(6):
                aid = f"penguin_{i}"
                if aid in pymunk_eng.penguins and pymunk_eng.penguins[aid].alive:
                    pm_actions[aid] = (angles[i], powers[i])
            pymunk_eng.apply_actions(pm_actions)
            pymunk_eng.step_until_settled()

            # Tensor
            t_actions = torch.zeros(1, 6, 2)
            for i in range(6):
                t_actions[0, i, 0] = angles[i]
                t_actions[0, i, 1] = powers[i]
            vel = tengine.apply_impulses(vel, t_actions, alive)
            pos, vel, alive, _ = tengine.step_until_settled(
                pos, vel, alive, DEFAULTS.ARENA_HALF_WIDTH
            )

            # Compare alive counts
            pm_a = pymunk_eng.get_alive_count(0)
            pm_b = pymunk_eng.get_alive_count(1)
            t_a = alive[0, :3].sum().item()
            t_b = alive[0, 3:].sum().item()

            total_checks += 1
            if pm_a == t_a and pm_b == t_b:
                alive_matches += 1

    match_pct = 100.0 * alive_matches / total_checks if total_checks else 0
    print(f"  Alive-count matches: {alive_matches}/{total_checks} ({match_pct:.1f}%)")
    if match_pct >= 90:
        print("  PASS (>= 90% match)")
    elif match_pct >= 70:
        print("  ACCEPTABLE (>= 70% match)")
    else:
        print("  WARN (< 70% match)")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    bench_single_game(rounds=10)
    bench_batched(tensor_batch=1000, pymunk_batch=8, rounds=10)
    bench_comparison(games=20, rounds=8)
    print("\nDone.")
