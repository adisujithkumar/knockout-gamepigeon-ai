def reward(obs: torch.Tensor) -> torch.Tensor:
    # Extract key alive flags
    ally1_alive = obs[:, 22]     # ally_0.alive
    ally2_alive = obs[:, 36]     # ally_1.alive  
    enemy1_alive = obs[:, 50]    # opponent_0.alive
    enemy2_alive = obs[:, 64]    # opponent_1.alive
    enemy3_alive = obs[:, 78]    # opponent_2.alive
    
    # Global team status
    ego_team_alive = obs[:, 86]   # global.ego_team_alive_count
    opp_team_alive = obs[:, 87]   # global.opponent_team_alive_count
    
    # Enemy positioning (distance to edge - lower is better for us)
    enemy2_dist_to_edge = obs[:, 61]  # opponent_1.distance_to_edge
    
    # Core reward: team alive count differential
    # This is the strongest signal from the statistics
    team_advantage = ego_team_alive - opp_team_alive
    
    # Enemy elimination incentive (negative because fewer enemies = better)
    enemy_elimination_bonus = -(enemy1_alive + enemy2_alive + enemy3_alive)
    
    # Ally preservation incentive  
    ally_preservation_bonus = ally1_alive + ally2_alive
    
    # Tactical positioning: reward when key enemies are near edges
    edge_pressure_bonus = -enemy2_dist_to_edge  # negative distance = closer to edge
    
    # Weighted combination based on statistical importance
    reward = (
        3.0 * team_advantage +           # Highest weight: team balance is most predictive
        1.5 * enemy_elimination_bonus +  # Strong weight: eliminating enemies
        1.0 * ally_preservation_bonus +  # Moderate weight: keeping allies alive
        0.3 * edge_pressure_bonus        # Small weight: tactical positioning
    )
    
    return reward