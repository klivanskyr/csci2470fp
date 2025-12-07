import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, latent_size, action_size, hiddent_size=64):
        super(RewardModel, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_size + action_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        return self.model(x)

