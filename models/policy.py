import torch

class Policy(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size=64):
        super(Policy, self).__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, action_size),
        )
    
    def forward(self, latent):
        return self.model(latent)