def reward(obs: torch.Tensor) -> torch.Tensor:
    # Extract survival features
    ego_alive = obs[:, 8]
    ally_alive = obs[:, [22, 36]]  # ally_0.alive, ally_1.alive  
    enemy_alive = obs[:, [50, 64, 78]]  # opponent_0.alive, opponent_1.alive, opponent_2.alive
    
    # Global team status
    ego_team_alive = obs[:, 86]
    opponent_team_alive = obs[:, 87]
    
    # Core survival rewards
    ego_survival_reward = ego_alive * 0.35  # Increased ego survival importance
    ally_survival_reward = ally_alive.sum(dim=1) * 0.15
    
    # Progressive enemy elimination - reward eliminating each enemy more
    enemy_count = enemy_alive.sum(dim=1)
    enemy_elimination_reward = (3 - enemy_count) * 0.15  # Increased weight
    
    # Team advantage with exponential scaling for decisive wins
    team_advantage = ego_team_alive - opponent_team_alive
    team_advantage_reward = team_advantage * 0.25 + (team_advantage ** 2) * 0.1
    
    # CORRECTED: Optimal center positioning - closer to center but not too close
    ego_distance_from_center = obs[:, 4]
    # Optimal range appears to be around 0.4-0.6 based on win data (mean 0.493)
    optimal_center_distance = 0.5
    center_positioning_reward = -torch.abs(ego_distance_from_center - optimal_center_distance) * 0.15
    
    # Enhanced edge safety - more aggressive penalty for edge proximity
    ego_distance_to_edge = obs[:, 5]
    edge_safety_reward = torch.clamp(ego_distance_to_edge * 2.0, min=0.0, max=0.4) * 0.1
    
    # Enemy pressure - push enemies away from center and towards edges
    enemy_distances_from_center = obs[:, [46, 60, 74]]
    enemy_distances_to_edge = obs[:, [47, 61, 75]]
    enemy_alive_mask = enemy_alive
    
    # Reward enemies being far from center (weighted by whether they're alive)
    enemy_center_pressure = (enemy_distances_from_center * enemy_alive_mask).sum(dim=1) * 0.08
    
    # Reward enemies being close to edges (negative distance_to_edge means close to edge)
    enemy_edge_pressure = -(enemy_distances_to_edge * enemy_alive_mask).sum(dim=1) * 0.05
    
    # Team coordination - reward allies being in good formation
    ally_distances_to_ego = obs[:, [27, 41]]  # ally distance_to_ego
    ally_alive_mask = ally_alive
    
    # Optimal ally distance - not too close (clustering) but not too far (isolation)
    optimal_ally_distance = 0.3
    ally_coordination = -torch.abs(ally_distances_to_ego - optimal_ally_distance) * ally_alive_mask
    ally_coordination_reward = ally_coordination.sum(dim=1) * 0.08
    
    # Tactical positioning - reward maintaining pressure on nearest enemies
    enemy_distances_to_ego = obs[:, [55, 69, 83]]
    alive_enemy_distances = enemy_distances_to_ego * enemy_alive_mask
    
    # Find closest alive enemy distance (for alive enemies only)
    alive_enemy_distances_masked = alive_enemy_distances + (1 - enemy_alive_mask) * 999  # Mask dead enemies
    closest_enemy_distance = alive_enemy_distances_masked.min(dim=1)[0]
    
    # Reward maintaining optimal pressure distance (not too close, not too far)
    optimal_engagement_distance = 0.4
    pressure_reward = -torch.abs(closest_enemy_distance - optimal_engagement_distance) * 0.06
    
    # Combine all reward components
    total_reward = (ego_survival_reward + 
                   ally_survival_reward + 
                   enemy_elimination_reward + 
                   team_advantage_reward + 
                   center_positioning_reward +
                   edge_safety_reward +
                   enemy_center_pressure +
                   enemy_edge_pressure +
                   ally_coordination_reward +
                   pressure_reward)
    
    return total_reward