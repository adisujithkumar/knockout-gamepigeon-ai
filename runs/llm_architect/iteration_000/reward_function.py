import torch

def reward(obs: torch.Tensor) -> torch.Tensor:
    # Reward for low distance_from_center (feature 4) -- basic survival
    return -obs[:, 4] * 0.5
