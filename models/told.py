import torch
import torch.nn as nn

from encoder import Encoder
from reward import RewardModel
from q import Q
from latent_dynamics import LatentDynamics
from policy import Policy

# Task-Oriented Latent Dynamics Model
class Told(nn.Module):
    def __init__(self, latent_size, action_size, state_size, hidden_size=64, horizon_steps=5, learning_rate=1e-3):
        super(Told, self).__init__()
        self.horizon_steps = horizon_steps

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


        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def Q(self, latent, action):
        q1 = self._Q1(latent, action)
        q2 = self._Q2(latent, action)
        return q1, q2

    @torch.no_grad()
    def _compute_td(self, next_state, reward):
        next_z = self.h

    # replay_buffer (states, actions, rewards) tensors
    # shape (episodes(num), states(tensor), actions(tensor), rewards(tensor))
    def update(self, replay_buffer, rho=0.9, c1=1.0, c2=1.0, c3=1.0):
        states, actions, rewards = replay_buffer.sample() # sample choices a random episode?

        # Encode states to latent space
        z = self.h(states[0]) # encode first state using main encoder

        # Make predictions in latent space
        reward_loss = 0
        value_loss = 0
        consistency_loss = 0

        for t in range(self.horizon_steps):
            q1, q2 = self.Q(z, actions[t])
            z_next = self.d(z, actions[t])
            r = self.R(z, actions[t])
            action = self.policy(z)

            # Compute targets and losses
            z_target = self.h.target(states[t+1]) # encode next state using target encoder
            td_target = _compute_td(rewards[t], states[t+1])
            reward_loss += rho**t + torch.mean((r - rewards[t]) ** 2)
            value_loss += rho**t * torch.mean((q1 - td_target) ** 2) + torch.mean((q2 - td_target) ** 2)
            consistency_loss += rho**t * torch.mean((z - z_target) ** 2)

        # Update
        total_loss = c1 * reward_loss + c2 * value_loss + c3 * consistency_loss
        total_loss.backward()
        self.optimizer.step()

        # Update target networks
        self.h.update_target_model() #not sure what targets to update ?????
        self.Q.update_target_model()


        
