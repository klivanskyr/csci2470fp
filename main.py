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
    def __init__(
        self, 
        state_space, 
        action_space, 
        hidden_size=128, 
        reward_dim=1, 
        trajectory_length=15, 
        num_trajectories=5, 
        learning_rate=1e-3, 
        batch_size=32, 
        epochs=100, 
        weight_decay=1e-5, 
        checkpoint_dir="./checkpoints",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):

        super(DreamWorld, self).__init__()

        # hyperparameters for training
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

        # for horizon planning
        self.trajectory_length = trajectory_length
        self.num_trajectories = num_trajectories
        self.state_space = state_space
        self.action_space = action_space

        # model architecture
        self.hidden_size = hidden_size
        self.reward_dim = reward_dim

        action_dim = action_space.shape[0]
        state_dim = state_space.shape[0]
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, state_dim + self.reward_dim) # predict next state and reward
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()

        self.to(self.device)


    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)

    def horizon_planning(self, state):
        self.eval()

        max_reward = float("-inf")
        best_starting_action = None

        # state comes from env.reset() / env.step() as a numpy array
        initial_state = state.copy()

        for _ in range(self.num_trajectories):
            trajectory_reward = 0.0
            current_state = initial_state.copy()  # numpy array or list
            first_action = self.action_space.sample()  # numpy array
            action = first_action

            for _ in range(self.trajectory_length):
                # convert to tensors on the right device
                state_tensor = torch.tensor(current_state, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, state_dim]
                action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, action_dim]

                with torch.no_grad():
                    pred = self.forward(state_tensor, action_tensor)[0]  # [state_dim + 1]

                # next_state as plain Python list to avoid .numpy() issues
                next_state = pred[:-1].detach().cpu().tolist()  # length = state_dim
                reward = float(pred[-1].item())

                current_state = next_state
                trajectory_reward += reward

                # random next action (you can later make this smarter)
                action = self.action_space.sample()

            if trajectory_reward > max_reward or best_starting_action is None:
                max_reward = trajectory_reward
                best_starting_action = first_action

        return best_starting_action



    def train_model(self, dataset):
        self.train() # sets model to training mode

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, len(dataset), self.batch_size):
                batch = dataset[start:start + self.batch_size]

                states = torch.tensor([item[0] for item in batch], dtype=torch.float32, device=self.device)
                actions = torch.tensor([item[1] for item in batch], dtype=torch.float32, device=self.device)
                rewards = torch.tensor([[item[2]] for item in batch], dtype=torch.float32, device=self.device)
                next_states = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=self.device)
                dones = torch.tensor([[item[4]] for item in batch], dtype=torch.float32, device=self.device)

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

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="Device to use: cpu or cuda")
    parser.add_argument("--max_episodes", default=100, type=int, help="Number of episodes to run")
    parser.add_argument("--num_steps", default=50, type=int, help="Max steps per episode")
    parser.add_argument("--save_every_n_episodes", default=5, type=int, help="Save video every n episodes")
    parser.add_argument("--dataset_size", default=1000, type=int, help="Size of the dataset to collect")

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

    dream_world = DreamWorld(
        state_space=state_space,
        action_space=action_space,
        hidden_size=args.dreamworld_hidden_size,
        learning_rate=args.dreamworld_lr,
        batch_size=args.dreamworld_batch_size,
        epochs=args.dreamworld_epochs,
        weight_decay=args.dreamworld_weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        trajectory_length=args.trajectory_length,
        num_trajectories=args.num_trajectories
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
            if not use_dreamworld: # warmpup with realworld
                action = env.action_space.sample() # random action
            else:
                # planned actions through horizons in dreamworld
                # try num_trajectory trajectories of length trajectory_length and pick best one
                action = dream_world.horizon_planning(state) 

            next_state, reward, terminated, truncated, info = env.step(action) # RealWorld Step
            dataset.append((state, action, reward, next_state, terminated or truncated)) # build dataset

            # train or retrain when dataset is large enough
            if len(dataset) >= args.dataset_size:
                use_dreamworld = True
                dream_world.train_model(dataset)
                dataset = [] # clear dataset after training

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
