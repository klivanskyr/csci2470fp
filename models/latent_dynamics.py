import torch
import torch.nn as nn

class LatentDynamics(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size=64, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(LatentDynamics, self).__init__()
        self.to(device)

        self.model = nn.Sequential(
            nn.Linear(latent_size + action_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        return self.model(x)