"""Test local LLM for game state reasoning."""
from llama_cpp import Llama
import time

# Try GPU first, fall back to CPU
try:
    llm = Llama(
        model_path="./models/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_gpu_layers=-1,  # all layers on GPU
        n_ctx=512,
        verbose=False,
    )
    print("Using GPU acceleration")
except Exception:
    llm = Llama(
        model_path="./models/qwen2.5-3b-instruct-q4_k_m.gguf",
        n_gpu_layers=0,
        n_ctx=512,
        verbose=False,
    )
    print("Using CPU only")

# Test 1: Can it reason about spatial positions?
prompt1 = """You are evaluating positions in a 2D game on a square arena (200x200 units, center at 0,0).
Objects near the edge (within 10 units of boundary at ±100) are in danger of elimination.

Current positions:
- Agent A: (5, 12) — near center, safe
- Agent B: (92, 45) — near right edge, danger!
- Agent C: (-30, -60) — moderate position

Rate each agent's safety from 0 (eliminated soon) to 100 (very safe).
Respond ONLY with JSON: {"A": score, "B": score, "C": score}"""

t0 = time.time()
r1 = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt1}],
    max_tokens=50,
    temperature=0.3,
)
t1 = time.time()
print(f"\nTest 1 ({t1-t0:.2f}s): {r1['choices'][0]['message']['content']}")

# Test 2: Can it suggest game actions?
prompt2 = """In a physics game, you control 3 agents on a square arena.
Each agent launches by choosing an angle (0-360 degrees) and power (0-400).
The goal is to push enemies off the arena edge.

Your agents: A at (20, 30), B at (50, 50), C at (80, 30)
Enemy agents: X at (20, 70), Y at (50, 60), Z at (80, 70)
Arena boundaries: ±100 in both x and y.

Suggest actions for your team. Respond ONLY with JSON:
{"actions": [{"agent": "A", "angle": <degrees>, "power": <0-400>}, ...]}"""

t0 = time.time()
r2 = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt2}],
    max_tokens=150,
    temperature=0.7,
)
t1 = time.time()
print(f"\nTest 2 ({t1-t0:.2f}s): {r2['choices'][0]['message']['content']}")

# Test 3: Quick position evaluation (for MCTS)
prompt3 = """Game state: Your team has 3 alive at center. Enemy has 2 alive near edges.
Arena is shrinking. Rate your winning probability 0-100. Answer with JUST a number."""

t0 = time.time()
r3 = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt3}],
    max_tokens=5,
    temperature=0.1,
)
t1 = time.time()
print(f"\nTest 3 ({t1-t0:.2f}s): {r3['choices'][0]['message']['content']}")

# Benchmark: many short evaluations (simulating MCTS)
print("\nBenchmark: 20 quick evaluations...")
t0 = time.time()
for i in range(20):
    llm.create_chat_completion(
        messages=[{"role": "user", "content": f"Game position score 0-100. Team: 3 alive center. Enemy: {3-i%3} alive, nearest at distance {20+i*5}. Just the number."}],
        max_tokens=5,
        temperature=0.1,
    )
t1 = time.time()
print(f"20 evaluations in {t1-t0:.2f}s ({(t1-t0)/20:.3f}s per eval, {20/(t1-t0):.1f} evals/sec)")
