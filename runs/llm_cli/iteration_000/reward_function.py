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
    enemy_elimination_reward = (3 - enemy_alive.sum(dim=1)) * 0.1  # Reward eliminating enemies
    
    # Team advantage reward
    team_advantage_reward = (ego_team_alive - opponent_team_alive) * 0.2
    
    # Strategic positioning rewards based on statistics
    # Enemies being further from center correlates with wins
    enemy_distances_from_center = obs[:, [46, 60, 74]]  # opponent distance_from_center features
    enemy_positioning_reward = enemy_distances_from_center.mean(dim=1) * 0.05
    
    # Combine all reward components
    total_reward = (ego_survival_reward + 
                   ally_survival_reward + 
                   enemy_elimination_reward + 
                   team_advantage_reward + 
                   enemy_positioning_reward)
    
    return total_reward