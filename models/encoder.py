import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
import copy

class Encoder(nn.Module):

    def __init__(self, observation_space_size: int, latent_size: int, hidden_size: int) -> None:
        super(Encoder, self).__init__()
            
        self.model = nn.Sequential(
            nn.Linear(observation_space_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, latent_size)
        )
        
        
        #need to build the weights before copying
        self.target = copy.deepcopy(self.model)


    def forward(self, state):
        return self.model(state)
    
    @torch.no_grad()
    def update_target_model(self):
        #call this occasionally during training
        self.target.load_state_dict(self.model.state_dict())
        
    # def loss_func(self, state, next_state, action, dream_world, discount_factor=0.99):
    #     zed = self.model(state) #returns latent of t+1
    #     pred_next_latent = dream_world(zed, action) #this the world model yo
    #     next_state_latent = self.target(next_state)
        
    #     # loss_consistency := mean(d(z_t, a_t) - h_{-theta}(next_state)) **2)
    #     # d := world model pred of next latent
    #     # h := encoding of next_state
        
    #     return torch.mean((pred_next_latent - next_state_latent)**2)
        
        