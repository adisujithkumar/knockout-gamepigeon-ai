def reward(obs: torch.Tensor) -> torch.Tensor:
    # Core survival metrics
    ego_alive = obs[:, 8]                    # ego_agent.alive
    ego_dist_to_edge = obs[:, 5]             # ego_agent.distance_to_edge
    
    # Team status
    ego_team_alive = obs[:, 86]              # global.ego_team_alive_count  
    opp_team_alive = obs[:, 87]              # global.opponent_team_alive_count
    
    # Allies status
    ally1_alive = obs[:, 22]                 # ally_0.alive
    ally2_alive = obs[:, 36]                 # ally_1.alive
    
    # Primary reward: Stay alive and safe
    survival_reward = ego_alive * 3.0
    safety_reward = ego_dist_to_edge * 4.0  # Can be negative - heavily penalize edge proximity
    
    # Secondary reward: Team advantage  
    team_advantage = (ego_team_alive - opp_team_alive) * 2.0
    
    # Tertiary reward: Ally preservation
    ally_bonus = (ally1_alive + ally2_alive) * 0.5
    
    # Small timestep reward to encourage longer episodes
    timestep = obs[:, 88]  # global.timestep
    duration_bonus = timestep * 0.2
    
    return survival_reward + safety_reward + team_advantage + ally_bonus + duration_bonus