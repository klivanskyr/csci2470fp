import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size=64, std=0.05, learning_rate=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Policy, self).__init__()
        self.std = std
        self.device = device
        self.to(device)

        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, action_size),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, latent, use_std=False):
        if use_std:
            noise = torch.randn_like(self.model(latent)) * self.std
            return self.model(latent) + noise

        return self.model(latent)