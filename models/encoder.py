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

    def __init__(self, observation_space_size: int, latent_size: int, hidden_size: int, device: str="cuda" if torch.cuda.is_available() else "cpu") -> None:
        super(Encoder, self).__init__()
        self.device = device
        self.to(device)
        
        self.model = nn.Sequential(
            nn.Linear(observation_space_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, latent_size)
        )


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)
        
        