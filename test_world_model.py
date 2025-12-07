import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from main import DreamWorld

def test_prediction_accuracy(env, dream_world, num_episodes=5, steps_per_episode=20):
    """Compare real environment vs predicted environment over multiple steps"""
    
    print("\n=== Testing Prediction Accuracy ===")
    total_state_error = 0.0
    total_reward_error = 0.0
    num_predictions = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        
        for step in range(steps_per_episode):
            action = env.action_space.sample()
            
            # Get real transition
            real_next_state, real_reward, terminated, truncated, _ = env.step(action)
            
            # Get predicted transition
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            pred = dream_world(state_tensor, action_tensor)
            pred_next_state = pred[0, :-1].detach().cpu().tolist()
            pred_reward = float(pred[0, -1].detach().cpu())
            
            # Calculate errors
            state_error = np.mean(np.abs(np.array(real_next_state) - np.array(pred_next_state)))
            reward_error = abs(real_reward - pred_reward)
            
            total_state_error += state_error
            total_reward_error += reward_error
            num_predictions += 1
            
            state = real_next_state
            
            if terminated or truncated:
                break
    
    avg_state_error = total_state_error / num_predictions
    avg_reward_error = total_reward_error / num_predictions
    
    print(f"Average State Prediction Error: {avg_state_error:.6f}")
    print(f"Average Reward Prediction Error: {avg_reward_error:.6f}")
    
    return avg_state_error, avg_reward_error


def test_dream_trajectory(dream_world, env, num_steps=15):
    """Generate a dream trajectory using only the learned model"""
    
    print(f"\n=== Generating Dream Trajectory ({num_steps} steps) ===")
    state, _ = env.reset()
    
    dream_states = [state.copy()]
    dream_rewards = []
    
    for step in range(num_steps):
        action = env.action_space.sample()
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        pred = dream_world(state_tensor, action_tensor)
        
        next_state = pred[0, :-1].detach().cpu().tolist()
        reward = float(pred[0, -1].detach().cpu())
        
        dream_states.append(next_state)
        dream_rewards.append(reward)
        
        state = next_state
    
    total_dream_reward = sum(dream_rewards)
    print(f"Dream Total Reward: {total_dream_reward:.4f}")
    print(f"Dream Rewards: {[f'{r:.4f}' for r in dream_rewards[:5]]}... (showing first 5)")
    
    return dream_states, dream_rewards


def test_real_vs_dream(env, dream_world, num_steps=20):
    """Run real episode and dream episode starting from same state, compare"""
    
    print(f"\n=== Real vs Dream Episode ({num_steps} steps) ===")
    
    # Real episode
    state, _ = env.reset()
    real_states = [state.copy()]
    real_rewards = []
    
    for step in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        real_states.append(next_state.copy())
        real_rewards.append(reward)
        state = next_state
        
        if terminated or truncated:
            break
    
    # Dream episode from same initial state
    state, _ = env.reset()
    dream_states = [state.copy()]
    dream_rewards = []
    
    for step in range(len(real_states) - 1):  # Match real episode length
        action = env.action_space.sample()
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        pred = dream_world(state_tensor, action_tensor)
        
        next_state = pred[0, :-1].detach().cpu().tolist()
        reward = float(pred[0, -1].detach().cpu())
        
        dream_states.append(next_state)
        dream_rewards.append(reward)
        state = next_state
    
    # Compare
    real_total = sum(real_rewards)
    dream_total = sum(dream_rewards)
    
    print(f"Real Episode:  {len(real_states)} steps, total reward = {real_total:.4f}")
    print(f"Dream Episode: {len(dream_states)} steps, total reward = {dream_total:.4f}")
    print(f"Reward Difference: {abs(real_total - dream_total):.4f}")
    
    # State divergence
    state_diffs = []
    for real_s, dream_s in zip(real_states, dream_states):
        diff = np.mean(np.abs(np.array(real_s) - np.array(dream_s)))
        state_diffs.append(diff)
    
    print(f"Average State Divergence: {np.mean(state_diffs):.6f}")
    print(f"Max State Divergence: {np.max(state_diffs):.6f}")
    
    return real_rewards, dream_rewards, state_diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved DreamWorld checkpoint")
    parser.add_argument("--num_steps", default=50, type=int, help="Max steps per episode")
    parser.add_argument("--trajectory_length", default=15, type=int, help="Dream trajectory length")
    
    args = parser.parse_args()
    
    # Create environment
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    env = TimeLimit(env.unwrapped, max_episode_steps=args.num_steps)
    
    state_space = env.observation_space
    action_space = env.action_space
    
    # Load DreamWorld model
    dream_world = DreamWorld(
        state_dim=state_space.shape[0],
        action_dim=action_space.shape[0],
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    dream_world.load_state_dict(checkpoint)
    dream_world.eval()
    
    print(f"Loaded DreamWorld from {args.checkpoint}")
    
    # Run tests
    state_error, reward_error = test_prediction_accuracy(env, dream_world, num_episodes=5, steps_per_episode=20)
    
    dream_states, dream_rewards = test_dream_trajectory(dream_world, env, num_steps=args.trajectory_length)
    
    real_rewards, dream_rewards_compare, state_diffs = test_real_vs_dream(env, dream_world, num_steps=args.trajectory_length)
    
    print("\n=== Test Summary ===")
    print(f"Prediction State Error: {state_error:.6f}")
    print(f"Prediction Reward Error: {reward_error:.6f}")
    print(f"Dream vs Real State Divergence: {np.mean(state_diffs):.6f}")
    
    env.close()


if __name__ == "__main__":
    main()
