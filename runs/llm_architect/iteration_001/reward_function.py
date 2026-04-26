import torch

def reward(obs: torch.Tensor) -> torch.Tensor:
    # Multi-objective: center positioning + opponent distance + edge avoidance
    center_bonus = -obs[:, 4] * 0.2
    enemy_far = obs[:, 47] * 0.3
    edge_penalty = (1.0 - obs[:, 5]) * 0.2  # feature 5 = distance_to_edge
    return center_bonus + enemy_far + edge_penalty
