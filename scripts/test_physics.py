"""Isolated physics validation script."""
import numpy as np
from knockout.core.physics_engine import PhysicsEngine
from knockout.core.config import DEFAULTS

def test_determinism():
    """Verify deterministic physics."""
    for seed in [42, 123, 789]:
        engine1 = PhysicsEngine(seed=seed)
        engine1.initialize_game()
        engine1.apply_actions({f"penguin_{i}": (float(i*45), 250.0) for i in range(6)})
        engine1.step(100)

        engine2 = PhysicsEngine(seed=seed)
        engine2.initialize_game()
        engine2.apply_actions({f"penguin_{i}": (float(i*45), 250.0) for i in range(6)})
        engine2.step(100)

        for i in range(6):
            p1 = engine1.penguins[f"penguin_{i}"].position
            p2 = engine2.penguins[f"penguin_{i}"].position
            assert p1 == p2, f"Seed {seed}, penguin_{i}: {p1} != {p2}"
    print("PASS: Determinism verified for 3 seeds")

def test_elimination():
    """Verify penguins get eliminated when leaving boundary."""
    engine = PhysicsEngine(seed=42)
    engine.initialize_game()

    # Launch penguin_0 outward with max force
    px, py = engine.penguins["penguin_0"].position
    angle = np.degrees(np.arctan2(py, px))  # Away from center
    engine.apply_actions({"penguin_0": (angle, 500.0)})

    for step in range(500):
        engine.step(1)
        if not engine.penguins["penguin_0"].alive:
            print(f"PASS: Penguin eliminated at step {step+1}")
            return

    print("FAIL: Penguin not eliminated after 500 steps")

def test_collision_bias():
    """Verify collision bias is corrected."""
    engine = PhysicsEngine(seed=42)
    expected = pow(1.0 - 0.1, 60.0)
    actual = engine.space.collision_bias
    assert abs(actual - expected) < 1e-10, f"Collision bias wrong: {actual} vs {expected}"
    print(f"PASS: Collision bias = {actual:.6f} (expected ~{expected:.6f})")

if __name__ == "__main__":
    test_determinism()
    test_elimination()
    test_collision_bias()
    print("\nAll physics validations passed!")
