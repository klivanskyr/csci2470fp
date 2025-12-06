import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit
import argparse
import os
import numpy as np
import torch
import torch.nn as nn

# Agent (state) -> action
class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(Agent, self).__init__()

        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, action_dim),
            nn.Tanh() # actions from -1 to 1
        )

    def forward(self, state):
        return self.model(state)

# DreamWorld (state, action) -> (next_state, reward)
class DreamWorld(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, reward_dim=1):
        super(DreamWorld, self).__init__()

        self.hidden_size = hidden_size
        self.reward_dim = reward_dim
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, state_dim + self.reward_dim) # predict next state and reward
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output", type=str, help="Directory to save results")
    parser.add_argument("--num_episodes", default=10, type=int, help="Number of episodes to run")
    parser.add_argument("--num_steps", default=50, type=int, help="Max steps per episode")
    parser.add_argument("--save_every_n_episodes", default=5, type=int, help="Save video every n episodes")
    parser.add_argument("--dataset_size", default=1000, type=int, help="Size of the dataset to collect")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make("Reacher-v5", render_mode="rgb_array")
    env = TimeLimit(env.unwrapped, max_episode_steps=args.num_steps)
    env = RecordVideo(env, video_folder=args.output_dir, episode_trigger=lambda ep: ep % args.save_every_n_episodes == 0)

    state_space = env.observation_space
    action_space = env.action_space

    agent = Agent(state_space.shape[0], action_space.shape[0])
    dream_world = DreamWorld(state_space.shape[0], action_space.shape[0])

    dataset = []
    for episode in range(args.num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = agent(state_tensor).detach().numpy()[0]

            # build dataset using realworld first.
            if len(dataset) < args.dataset_size:
                next_state, reward, terminated, truncated, info = env.step(action) # RealWorld Step
                dataset.append((state, action, reward, next_state, terminated or truncated))
                
            else: # use dreamworld to predict next state and reward
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                next_state, reward = dream_world(state_tensor, action_tensor).detach().numpy()[0]

            done = terminated or truncated
            total_reward += reward
            steps += 1

        if episode == 0 or (episode + 1) % args.save_every_n_episodes == 0:
            print(f"Episode {episode + 1}: steps={steps}, total_reward={total_reward:.2f}")

    env.close()
    print(f"Videos saved to {args.output_dir}")

if __name__ == "__main__":
    main()
