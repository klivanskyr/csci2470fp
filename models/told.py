import torch
import torch.nn as nn

from models.encoder import Encoder
from models.reward import RewardModel
from models.q import Q
from models.latent_dynamics import LatentDynamics
from models.policy import Policy

# Task-Oriented Latent Dynamics Model
class Told(nn.Module):
    def __init__(self, latent_size, action_size, state_size, hidden_size=64, horizon_steps=5, learning_rate=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Told, self).__init__()
        self.horizon_steps = horizon_steps
        self.device = device
        self.to(device)
        

        self.h = Encoder(observation_space_size=state_size, latent_size=latent_size, hidden_size=hidden_size)
        self.R = RewardModel(latent_size, action_size, hidden_size)
        self._Q1 = Q(latent_size, action_size, hidden_size)
        self._Q2 = Q(latent_size, action_size, hidden_size)
        self.d = LatentDynamics(latent_size, action_size, hidden_size)
        self.policy = Policy(latent_size, action_size, hidden_size)

        # set Q1, Q2 and Reward last layer to zeros to stabilize learning
        # take min of Q1 and Q2 to reduce overestimation bias
        for m in [self.R.model, self._Q1.model, self._Q2.model]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def Q(self, latent, action):
        q1 = self._Q1(latent, action)
        q2 = self._Q2(latent, action)
        return q1, q2
    
    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for q in [self._Q1, self._Q2]:
            for parameter in q.parameters():
                parameter.requires_grad_(enable)