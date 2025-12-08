import torch

class ReplayBuffer():
    def __init__(self, horizon_size, state_size, action_size, num_episodes, episode_size, batch_size, device):
        self.horizon_size = horizon_size
        self.state_size = state_size
        self.action_size = action_size
        self.episode_size = episode_size
        self.num_episodes = num_episodes
        self.batch_size = batch_size  # NEW
        self.device = device    
        
        self.states = torch.empty((num_episodes, episode_size, state_size), dtype=torch.float32, device=device)
        self.final_states = torch.empty((num_episodes, state_size), dtype=torch.float32, device=device)
        self.actions = torch.empty((num_episodes, episode_size, action_size), dtype=torch.float32, device=device)
        self.rewards = torch.empty((num_episodes, episode_size), dtype=torch.float32, device=device)
        
        self.current_index = 0
        self.full = False
    
    def add(self, episode_index: int, states: torch.Tensor, final_state: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        self.states[episode_index] = states
        self.final_states[episode_index] = final_state
        self.actions[episode_index] = actions
        self.rewards[episode_index] = rewards
        
        self.current_index += 1
        if self.current_index >= self.num_episodes:
            self.full = True
            self.current_index = 0
    
    def sample_horizon(self):
        """
        Sample batch_size horizons from the buffer.
        Returns:
            states: (horizon_size, batch_size, state_size)
            actions: (horizon_size, batch_size, action_size)
            rewards: (horizon_size, batch_size)
            next_states: (horizon_size, batch_size, state_size)
        """
        # We need horizon_size + 1 states to get horizon_size next_states
        # So we can only start from positions 0 to (episode_size - horizon_size - 1)
        num_horizons_per_episode = self.episode_size - self.horizon_size
        
        # Sample batch_size random horizons
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        
        for _ in range(self.batch_size):
            # Sample random episode
            max_episodes = self.num_episodes if self.full else self.current_index
            episode_idx = torch.randint(0, max_episodes, (1,)).item()
            
            # Sample random start point within episode (leaving room for horizon + 1 for next_states)
            start_idx = torch.randint(0, num_horizons_per_episode, (1,)).item()
            end_idx = start_idx + self.horizon_size
            
            # Extract horizon
            states = self.states[episode_idx, start_idx:end_idx]  # (horizon, state_size)
            actions = self.actions[episode_idx, start_idx:end_idx]  # (horizon, action_size)
            rewards = self.rewards[episode_idx, start_idx:end_idx]  # (horizon,)
            
            # Next states are states shifted by 1
            next_states = self.states[episode_idx, start_idx+1:end_idx+1]  # (horizon, state_size)
            
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
        
        # Stack into batches: (batch, horizon, dim) -> (horizon, batch, dim)
        states = torch.stack(batch_states, dim=1)  # (horizon, batch, state_size)
        actions = torch.stack(batch_actions, dim=1)  # (horizon, batch, action_size)
        rewards = torch.stack(batch_rewards, dim=1)  # (horizon, batch)
        next_states = torch.stack(batch_next_states, dim=1)  # (horizon, batch, state_size)
        
        return states, actions, rewards.unsqueeze(-1), next_states  # Add dim for rewards: (horizon, batch, 1)