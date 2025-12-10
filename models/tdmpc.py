import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from .told import Told

class TDMPC(nn.Module):
    def __init__(self, action_size, state_size, latent_size=64, hidden_size=64, horizon_steps:int=5, learning_rate=1e-3, temperature=0.5, tau=0.01, iterations=6, mixture_coeff=0.05, num_samples=256, rho=0.9, c1=0.5, c2=0.1, c3=2, temporal_coeff=0.5, discount_factor=0.99, num_elites=64, momentum=0.1, min_std=0.05, device="cuda" if torch.cuda.is_available() else "cpu", grad_clip_norm=10.0, target_update_interval=2, duration=25000):
        super(TDMPC, self).__init__()

        self.model = Told(latent_size, action_size, state_size, hidden_size, horizon_steps, learning_rate)
        self.model.to(device)
        
        self.target_model = copy.deepcopy(self.model)
        self.target_model.to(device)

        self.target_model.eval()  # Use deterministic behavior
        for p in self.target_model.parameters():
            p.requires_grad = False  # No gradients

        self.device = device

        self.target_update_interval = target_update_interval

        self.horizon_steps = horizon_steps
        self.action_size = action_size
        self.state_size = state_size
        self.discount_factor = discount_factor
        self.num_elites = num_elites
        self.temperature = temperature
        self.tau = tau
        self.momentum = momentum
        self.min_std = min_std

        self.grad_clip_norm = grad_clip_norm
        
        self.iterations = iterations
        self.mixture_coeff = mixture_coeff
        self.num_samples = num_samples
        
        start_val = 0.5
        self.std_schedule = torch.linspace(start_val, self.min_std, duration) #want our std to decay over time

        # for update.pa
        self.rho = rho
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.temporal_coeff = temporal_coeff

        self.in_warmup_phase = True # initially in warmup phase, gets out after first num_episodes are stored in replay buffer

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    # for infernace use the policy
    def forward(self, state):
        with torch.no_grad():
            action = self.model.policy(self.model.h(state), use_std=False)
        return action
    
    def update_target_model(self):
        with torch.no_grad():
            for p, p_target in zip(self.model.parameters(), self.target_model.parameters()):
                p_target.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

    def update_pi(self, latent_states):
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        pi_loss = 0
        for t, z in enumerate(latent_states):
            action = self.model.policy(z, use_std=True) # z.shape = (batch_size, latent_size)
            q1, q2 = self.model.Q(z, action) # qi.shape = (batch_size, 1)
            # print(f"Policy update Step {t}: q1 min {q1.min().item()}, q1 max {q1.max().item()}, q2 min {q2.min().item()}, q2 max {q2.max().item()}")
            Q = torch.min(q1, q2)
            pi_loss += -Q.mean() * (self.rho ** t) # Q.mean() average over batch_size

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.grad_clip_norm, error_if_nonfinite=False)

        self.model.policy.optimizer.step()
        self.model.track_q_grad(True)

        return pi_loss.item() # return the loss value for logging


    @torch.no_grad()
    def _td_target(self, next_state, reward):
        #td loss ?
        #gamma_t = Reward(t+1) + learning_rate[=1] * Value(state_t+1) - Value(state_t)
        # self.target_model.R()

        next_latent = self.model.h(next_state)
        q1, q2 = self.target_model.Q(next_latent, self.model.policy(next_latent, use_std=True)) #need to fix ts later
        return reward + self.discount_factor*torch.min(q1, q2)
    
    @torch.no_grad()
    def estimate_value(self, z, actions):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G = 0
        discount = 1.0

        for t in range(self.horizon_steps):
            next_z = self.model.d(z, actions[t])
            reward = self.model.R(next_z, actions[t])

            G += reward * discount
            discount *= self.discount_factor
            z = next_z

        q1, q2 = self.target_model.Q(z, self.model.policy(z, use_std=True))
        G += discount * torch.min(q1,q2)
        return G

    # c1 reward loss, c2 value loss, c3 consistency loss, lambda temporal coeff
    def update(self, replay_buffer, step):
        self.optimizer.zero_grad(set_to_none=True)

        # Sample returns batched data now
        # states: (horizon, batch_size, state_size)
        # actions: (horizon, batch_size, action_size)
        # rewards: (horizon, batch_size, 1)
        # next_states: (horizon, batch_size, state_size)
        states, actions, rewards, next_states = replay_buffer.sample_horizon()

        # move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)

        model = self.model
        model.train()

        target = self.target_model

        # Encode states to latent space
        z = model.h(states[0]) # encode first state using main encoder
        # print(f"Encoded latent state z: {z.shape}")

        seen_latents = [z.detach()] # store seen latents for calculating policy loss later 

        # Make predictions in latent space
        reward_loss = 0
        value_loss = 0
        consistency_loss = 0

        for t in range(self.horizon_steps): # loop over horizon
            q1, q2 = model.Q(z, actions[t])
            z_next = model.d(z, actions[t])
            r = model.R(z_next, actions[t])

            # store latent for policy loss
            seen_latents.append(z_next.detach())

            # Compute targets and losses
            with torch.no_grad():
                z_target = target.h(next_states[t]) # encode next state using target TOLD encoder
                td_target = self._td_target(next_states[t], rewards[t])

            # # print min and max of values for debugging
            rho = self.rho ** t
            # print(f"Step {step}, Time {t}: q1 min {q1.min().item()}, q1 max {q1.max().item()}, q2 min {q2.min().item()}, q2 max {q2.max().item()}, r min {r.min().item()}, r max {r.max().item()}, td_target min {td_target.min().item()}, td_target max {td_target.max().item()}")
            reward_loss += rho * F.mse_loss(r, rewards[t], reduction='mean')
            value_loss += rho * (
                F.mse_loss(q1, td_target, reduction='mean') + 
                F.mse_loss(q2, td_target, reduction='mean')
            )
            consistency_loss += rho * F.mse_loss(z_next, z_target, reduction='mean')

            # iterate z
            z = z_next

        # Print loss values after accumulation
        # print(f"Step {step}: Reward loss: {reward_loss.item()}, Value loss: {value_loss.item()}, Consistency loss: {consistency_loss.item()}")

        # clamp losses
        reward_loss = torch.clamp(reward_loss, max=1e4)
        value_loss = torch.clamp(value_loss, max=1e4)
        consistency_loss = torch.clamp(consistency_loss, max=1e4)

        # Update
        total_loss = (
            self.c3 * consistency_loss +
            self.c1 * reward_loss + 
            self.c2 * value_loss
        )

        total_loss.register_hook(lambda grad: grad * (1/self.horizon_steps)) # during backprop, scale the loss to account for horizon
        total_loss.backward()

        # clip gradients to avoid exploding gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm, error_if_nonfinite=False)
        self.optimizer.step() # update gradients

        # update policy
        policy_loss = self.update_pi(seen_latents)
        # print(f"Step {step}: Policy loss: {policy_loss}")

        # Update target networks
        if step % self.target_update_interval == 0:
            self.update_target_model()

        self.model.eval() # set model to eval mode

        return {
            'total_loss': total_loss.item(),
            'reward_loss': reward_loss.item(),
            'value_loss': value_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'policy_loss': policy_loss,
            'grad_norm': grad_norm.item()
        }

    # plan trajectory given initial state and return action sequence
    # Use mixture coefficient to mix random actions and policy actions to both explore and exploit 
    @torch.no_grad()
    def traject(self, initial_state, step, eval_mode = False):
        if step > len(self.std_schedule):
            self.std = self.min_std
        else:
            self.std = self.std_schedule[step]

        if self.in_warmup_phase and not eval_mode: # no planning during warmup phase
            # return random action sequence [-1, 1]
            return torch.empty(self.action_size, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # calculate how many trajectories to sample from policy vs random
        num_pi_trajs = int(self.mixture_coeff * self.num_samples)
    
        # create a sequency of actions based on our policy
        pi_actions = torch.empty(self.horizon_steps, num_pi_trajs, self.action_size, dtype=torch.float32, device=self.device)
        # print("PI actions shape:", pi_actions.shape)
        latent = self.model.h(initial_state).repeat(num_pi_trajs, 1)  # Repeat latent for each policy trajectory
        for t in range(self.horizon_steps):
            pi_actions[t] = self.model.policy(latent, use_std=True)  # (num_pi_trajs, action_size)
            latent = self.model.d(latent, pi_actions[t])  # Update latent state

        # initalize mean and std for CEM/MPPI
        mean = torch.zeros(self.horizon_steps, self.action_size, dtype=torch.float32, device=self.device)

        if step > 0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]  # shift mean from previous step

        std = 2*torch.ones(self.horizon_steps, self.action_size, dtype=torch.float32, device=self.device)
        # print(f"mean shape: {mean.shape}, std shape: {std.shape}")
        z = self.model.h(initial_state).repeat(self.num_samples + num_pi_trajs, 1) #need to have states for every traj

        for i in range(self.iterations): #now we do mppi for our actions
            trajs = torch.randn(self.horizon_steps, self.num_samples, self.action_size, dtype=torch.float32, device=self.device) * std.unsqueeze(1) + mean.unsqueeze(1)
            trajs = torch.clamp(trajs, -1, 1)  # Ensure actions are within valid range [-1, 1]
            actions = torch.cat((trajs, pi_actions), dim=1)

            value = self.estimate_value(z, actions).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim=0).indices
            best_values, best_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters

            # un fuck this later?
            max_value = best_values.max(0)[0]
            score = torch.exp(self.temperature*(best_values - max_value))
            score /= score.sum(0)
            # print(f"score shape: {score.shape}, best_actions shape: {best_actions.shape}")
            _mean = torch.sum(score.unsqueeze(0) * best_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (best_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            # _std = _std.clamp_(self.std, 2)
            # mean, std = self.momentum * mean + (1 - self.momentum) * _mean, _std #paper's code
            std = _std.clamp_(self.std, 2)
            # print(f"momentum: {self.momentum}, mean: {mean.shape}, _mean: {_mean}")
            mean = self.momentum * mean + (1 - self.momentum) * _mean #hopefully functional code ?

		# Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = best_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.action_size, device=std.device)
        return a