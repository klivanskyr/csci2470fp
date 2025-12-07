import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
import copy

ENCODE_SIZE = 8

class Encoder(nn.Module):
    
    def __init__(self, observation_space_size: int) -> None:
        super(Encoder, self).__init__()
        
        self.hidden_size = 16
        
        learning_rate = 1e-3
        self.loss_fn = nn.cr
        
        
        self.model = nn.Sequential(
            nn.Linear(observation_space_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, ENCODE_SIZE)
        )
        
        
        #need to build the weights before copying
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(learning_rate)


    def __call__(self, state):
        return self.model(state)
    
    @torch.no_grad()
    def update_model(self):
        #call this occasionally during training
        self.target.load_state_dict(self.model.state_dict())
        
    def loss_func(self, state, next_state, action, dream_world, discount_factor=0.99):
        zed = self.model(state) #returns latent of t+1
        pred_next_latent = dream_world(zed, action) #this the world model yo
        next_state_latent = self.target(next_state)
        
        # loss_consistency := mean(d(z_t, a_t) - h_{-theta}(next_state)) **2)
        # d := world model pred of next latent
        # h := encoding of next_state
        
        return torch.mean((pred_next_latent - next_state_latent)**2)
        
        