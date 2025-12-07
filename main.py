import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import wandb

# Agent (state) -> action
# class Agent(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_size=128, learning_rate=1e-3, weight_decay=1e-5, batch_size=32, epochs=100):
#         super(Agent, self).__init__()

#         # hyperparameters for training
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay

#         # model architecture
#         self.hidden_size = hidden_size
#         self.model = nn.Sequential(
#             nn.Linear(state_dim, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, action_dim),
#             nn.Tanh() # actions from -1 to 1
#         )

    # self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # self.loss_fn = nn.MSELoss()

#     def forward(self, state):
#         return self.model(state)

# DreamWorld (state, action) -> (next_state, reward)
class DreamWorld(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, reward_dim=1, trajectory_length=15, learning_rate=1e-3, batch_size=32, epochs=100, weight_decay=1e-5, checkpoint_dir="./checkpoints"):
        super(DreamWorld, self).__init__()

        # hyperparameters for training
        self.checkpoint_dir = checkpoint_dir
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # model architecture
        self.hidden_size = hidden_size
        self.reward_dim = reward_dim
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, state_dim + self.reward_dim) # predict next state and reward
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()


    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)

    def train_model(self, dataset):
        self.train() # sets model to training mode

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, len(dataset), self.batch_size):
                batch = dataset[start:start + self.batch_size]

                states = torch.tensor([item[0] for item in batch], dtype=torch.float32)
                actions = torch.tensor([item[1] for item in batch], dtype=torch.float32)
                rewards = torch.tensor([[item[2]] for item in batch], dtype=torch.float32)
                next_states = torch.tensor([item[3] for item in batch], dtype=torch.float32)
                dones = torch.tensor([[item[4]] for item in batch], dtype=torch.float32)
    
                pred = self.forward(states, actions)
                pred_next_states = pred[:, :-1]
                pred_rewards = pred[:, -1].unsqueeze(-1)

                loss_next_state = self.loss_fn(pred_next_states, next_states)
                loss_reward = self.loss_fn(pred_rewards, rewards)
                loss = loss_next_state + loss_reward

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            if (epoch + 1) % 10 == 0:
                wandb.log({"dreamworld_loss": avg_loss, "dreamworld_epoch": epoch + 1})
                # Save checkpoint every 10 epochs
                checkpoint_path = os.path.join(self.checkpoint_dir, f"dreamworld_epoch_{epoch + 1}.pth")
                torch.save(self.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        print(f"[DreamWorld] Trained on {len(dataset)} samples.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output", type=str, help="Directory to save results")
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str, help="Directory to save checkpoints")

    parser.add_argument("--wandb_project", default="world-model", type=str, help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", default=None, type=str, help="Weights & Biases entity name")

    parser.add_argument("--max_episodes", default=1000, type=int, help="Number of episodes to run")
    parser.add_argument("--num_steps", default=50, type=int, help="Max steps per episode")
    parser.add_argument("--save_every_n_episodes", default=5, type=int, help="Save video every n episodes")
    parser.add_argument("--dataset_size", default=500, type=int, help="Size of the dataset to collect")

    # DreamWorld specific arguments
    parser.add_argument("--num_trajectories", default=10, type=int, help="Number of trajectories to collect for DreamWorld")
    parser.add_argument("--trajectory_length", default=15, type=int, help="Length of each trajectory for DreamWorld")
    parser.add_argument("--dreamworld_hidden_size", default=128, type=int, help="Hidden size for DreamWorld network")
    parser.add_argument("--dreamworld_lr", default=1e-3, type=float, help="Learning rate for DreamWorld")
    parser.add_argument("--dreamworld_batch_size", default=32, type=int, help="Batch size for DreamWorld training")
    parser.add_argument("--dreamworld_epochs", default=100, type=int, help="Number of epochs for DreamWorld training")
    parser.add_argument("--dreamworld_weight_decay", default=1e-5, type=float, help="Weight decay for DreamWorld optimizer")

    # Agent specific arguments
    parser.add_argument("--agent_hidden_size", default=128, type=int, help="Hidden size for Agent network")
    parser.add_argument("--agent_lr", default=1e-3, type=float, help="Learning rate for Agent")
    parser.add_argument("--agent_batch_size", default=32, type=int, help="Batch size for Agent training")
    parser.add_argument("--agent_epochs", default=100, type=int, help="Number of epochs for Agent training")
    parser.add_argument("--agent_weight_decay", default=1e-5, type=float, help="Weight decay for Agent optimizer")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name="world-model-run"
    )

    env = gym.make("Reacher-v5", render_mode="rgb_array")
    env = TimeLimit(env.unwrapped, max_episode_steps=args.num_steps)
    env = RecordVideo(env, video_folder=args.output_dir, episode_trigger=lambda ep: ep % args.save_every_n_episodes == 0)

    state_space = env.observation_space
    action_space = env.action_space

    # agent does not train. Just random actions for data collection.
    # agent = Agent(
    #     state_space.shape[0],
    #     action_space.shape[0],
    #     hidden_size=args.agent_hidden_size,
    #     learning_rate=args.agent_lr,
    #     weight_decay=args.agent_weight_decay,
    #     batch_size=args.agent_batch_size,
    #     epochs=args.agent_epochs,
    # )

    dream_world = DreamWorld(
        state_dim=state_space.shape[0],
        action_dim=action_space.shape[0],
        hidden_size=args.dreamworld_hidden_size,
        learning_rate=args.dreamworld_lr,
        batch_size=args.dreamworld_batch_size,
        epochs=args.dreamworld_epochs,
        weight_decay=args.dreamworld_weight_decay,
        checkpoint_dir=args.checkpoint_dir,
    )

    dataset = [] # (state, action, reward, next_state, done)
    use_dreamworld = False
    best_reward = float('-inf')
    best_episode = 0
    
    for episode in range(args.max_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            # agent makes an action
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # action = agent(state_tensor).detach().numpy()[0]
            action = env.action_space.sample()  # Random action for data collection

            # build dataset using realworld first.
            if not use_dreamworld:
                next_state, reward, terminated, truncated, info = env.step(action) # RealWorld Step
                dataset.append((state, action, reward, next_state, terminated or truncated)) # build dataset
            else: # use dreamworld to predict next state and reward
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

                pred = dream_world(state_tensor, action_tensor)
                next_state = pred[0, :-1].detach().cpu().tolist()
                reward = float(pred[0, -1].detach().cpu())
                terminated = False
                truncated = False

            # check if we now have enough data to train dreamworld
            if len(dataset) >= args.dataset_size:
                use_dreamworld = True
                dream_world.train_model(dataset)
                dataset = [] # clear dataset after training

            # trajectory length cutoff for done
            if use_dreamworld and steps >= args.trajectory_length: 
                done = True
            else:
                done = terminated or truncated
                
            state = next_state
            total_reward += reward
            steps += 1

        if episode == 0 or (episode + 1) % args.save_every_n_episodes == 0:
            print(f"Episode {episode + 1}: steps={steps}, total_reward={total_reward:.2f}")
            
            # Track best reward
            if total_reward > best_reward:
                best_reward = total_reward
                best_episode = episode + 1
            
            log_dict = {
                "episode": episode + 1,
                "steps": steps,
                "total_reward": total_reward,
                "use_dreamworld": use_dreamworld,
                "best_reward": best_reward,
                "best_episode": best_episode
            }
            
            # Upload video if it exists
            video_path = os.path.join(args.output_dir, f"rl-video-episode-{episode}.mp4")
            if os.path.exists(video_path):
                log_dict["video"] = wandb.Video(video_path, fps=30, format="mp4")
            
            wandb.log(log_dict)

    env.close()
    print(f"Videos saved to {args.output_dir}")
    wandb.finish()

if __name__ == "__main__":
    main()
