import torch
import numpy as np

# Adds episodes to the replay buffer to store and sample horizons from
class ReplayBuffer():
    def __init__(self, horizon_size, state_size, action_size, num_episodes, episode_size, device):
        self.horizon_size = horizon_size
        self.state_size = state_size
        self.action_size = action_size
        self.episode_size = episode_size
        self.num_episodes = num_episodes
        self.device = device    

        # stores from state 0 to state episode_size-1. Meaning we need to store the final state separately
        # Dont need to store next_states since its just state_i + 1
        self.states = torch.empty((num_episodes, episode_size, state_size), dtype=torch.float32, device=device)
        self.final_states = torch.empty((num_episodes, state_size), dtype=torch.float32, device=device)
        self.actions = torch.empty((num_episodes, episode_size, action_size), dtype=torch.float32, device=device)
        self.rewards = torch.empty((num_episodes, episode_size), dtype=torch.float32, device=device)
        
        self.current_index = 0
        self.full = False

    def add(self, episode_index: int, states: torch.Tensor, final_state: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor):
        # states is episode_size + 1 bc 0 to episode_size
        self.states[episode_index] = states
        self.final_states[episode_index] = final_state
        self.actions[episode_index] = actions
        self.rewards[episode_index] = rewards

        self.current_index += 1
        if self.current_index >= self.num_episodes:
            self.full = True
            self.current_index = 0

    # random sample a horizon from the replay buffer
    def sample_horizon(self): # return shape (horizon_size, state_size), (horizon_size, action_size), (horizon_size,), (state_size,)
        num_horizons_in_episode = self.episode_size // self.horizon_size
        num_horizons = self.num_episodes * num_horizons_in_episode

        # sample random horizon index
        horizon_index = torch.randint(0, num_horizons, (1,)).item() # sample uniform distribution
        episode_index = horizon_index // num_horizons_in_episode
        horizon_start = horizon_index % num_horizons_in_episode * self.horizon_size
        horizon_end = horizon_start + self.horizon_size

        return self.states[episode_index, horizon_start:horizon_end], self.actions[episode_index, horizon_start:horizon_end], self.rewards[episode_index, horizon_start:horizon_end], self.final_states[episode_index]

    
