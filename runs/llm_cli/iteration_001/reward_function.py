def reward(obs: torch.Tensor) -> torch.Tensor:
    # Extract key survival features
    ego_alive = obs[:, 8]  # ego_agent.alive
    ally_alive = obs[:, [22, 36]]  # ally_0.alive, ally_1.alive  
    enemy_alive = obs[:, [50, 64, 78]]  # opponent_0.alive, opponent_1.alive, opponent_2.alive
    
    # Global team status
    ego_team_alive = obs[:, 86]  # global.ego_team_alive_count
    opponent_team_alive = obs[:, 87]  # global.opponent_team_alive_count
    
    # Core survival rewards
    ego_survival_reward = ego_alive * 0.3  # Ego survival is critical
    ally_survival_reward = ally_alive.sum(dim=1) * 0.15  # Reward keeping allies alive
    enemy_elimination_reward = (3 - enemy_alive.sum(dim=1)) * 0.12  # Slightly increased
    
    # Team advantage reward
    team_advantage_reward = (ego_team_alive - opponent_team_alive) * 0.2
    
    # Enhanced enemy positioning reward - enemies being pushed to edges
    enemy_distances_from_center = obs[:, [46, 60, 74]]  # opponent distance_from_center features
    enemy_positioning_reward = enemy_distances_from_center.mean(dim=1) * 0.08  # Increased weight
    
    # NEW: Ego center positioning reward - staying near center is strategic
    ego_distance_from_center = obs[:, 4]  # ego_agent.distance_from_center
    ego_center_reward = (1.0 - ego_distance_from_center) * 0.06  # Reward being near center
    
    # NEW: Distance maintenance reward - keeping enemies at distance
    enemy_distances_to_ego = obs[:, [55, 69, 83]]  # opponent distance_to_ego features
    # Only count alive enemies for distance reward
    enemy_alive_mask = enemy_alive  # [B, 3]
    masked_distances = enemy_distances_to_ego * enemy_alive_mask
    alive_enemy_count = enemy_alive.sum(dim=1, keepdim=True).clamp(min=1.0)  # Avoid division by zero
    distance_reward = (masked_distances.sum(dim=1) / alive_enemy_count.squeeze()) * 0.05
    
    # NEW: Edge safety for ego - avoid being too close to boundaries
    ego_distance_to_edge = obs[:, 5]  # ego_agent.distance_to_edge (negative when close to edge)
    edge_safety_reward = torch.clamp(ego_distance_to_edge + 0.2, min=0.0, max=0.2) * 0.1
    
    # Combine all reward components
    total_reward = (ego_survival_reward + 
                   ally_survival_reward + 
                   enemy_elimination_reward + 
                   team_advantage_reward + 
                   enemy_positioning_reward +
                   ego_center_reward +
                   distance_reward +
                   edge_safety_reward)
    
    return total_reward