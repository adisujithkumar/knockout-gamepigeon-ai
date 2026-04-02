"""MAPPO training script.

Usage:
    .venv/bin/python scripts/train_mappo.py --timesteps 100000 --num-envs 8 --opponent heuristic --lr 3e-4 --save-path checkpoints/mappo.pt
"""

from knockout.training.mappo import main

if __name__ == "__main__":
    main()
