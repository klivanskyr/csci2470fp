import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit
import torch
import wandb

from helpers.replay_buffer import ReplayBuffer
from models.tdmpc import TDMPC

# config = {
#     "train_steps": 25000,
#     "episode_length": 50,
#     "latent_size": 64,
#     "hidden_size": 256,
#     "horizon_steps": 5,
#     "iterations": 6,
#     "num_elites": 32,
#     "warmup_steps": 1000,
#     "eval_interval": 5000,
#     "eval_episodes": 5,
#     "num_episodes": 2000,
# }

config = {
    "train_steps": 10000,
    "episode_length": 50,
    "latent_size": 32,
    "hidden_size": 128,
    "horizon_steps": 5,
    "iterations": 4,
    "num_elites": 16,
    "warmup_steps": 500,
    "eval_interval": 1000,
    "eval_episodes": 5,
    "num_episodes": 1000,
}

# Initialize wandb
wandb.init(project="tdmpc-training", config=config)

# Log hyperparameters
wandb.config.update(config)

def train():
    env = gym.make("Reacher-v5", render_mode="rgb_array")
    env = TimeLimit(env.unwrapped, max_episode_steps=config["episode_length"])
    env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: x % 5 == 0)

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    model = TDMPC(
        latent_size=config["latent_size"],
        action_size=action_size,
        state_size=state_size,
        hidden_size=config["hidden_size"],
        horizon_steps=config["horizon_steps"],
        iterations=config["iterations"],
        num_elites=config["num_elites"],
    )

    replay_buffer = ReplayBuffer(horizon_size=config["horizon_steps"], state_size=state_size, action_size=action_size, num_episodes=config["num_episodes"], episode_size=config["episode_length"], device=model.device)

    # collect inital data for replay buffer by taking random actions from the real env
    episode_idx = 0
    for step in range(0, config["train_steps"]+config["episode_length"], config["episode_length"]):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        episode = {
            "states": torch.empty((config["episode_length"], state_size), device=model.device),
            "actions": torch.empty((config["episode_length"], action_size), device=model.device),
            "rewards": torch.empty((config["episode_length"], 1), device=model.device),
            "last_state": torch.empty((1, state_size), device=model.device),
        }

        i = 0
        # store initial state
        while not done:
            state = torch.from_numpy(state).to(model.device).float()
            action: torch.Tensor = model.traject(state, step)

            episode["states"][i] = state

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated

            # store states, actions, rewards
            episode["actions"][i] = action
            episode["rewards"][i] = torch.tensor(reward, device=model.device, dtype=torch.float32).unsqueeze(0)


            state = next_state
            episode_reward += reward
            i += 1

        # store last state
        episode["last_state"][0] = torch.tensor(next_state, device=model.device, dtype=torch.float32)

        print(f"Episode {episode_idx} - Reward: {episode_reward}")
        wandb.log({"episode_reward": episode_reward, "episode": episode_idx})

        # store episode in replay buffer
        replay_buffer.add(
            episode_index=episode_idx,
            states=episode["states"],
            final_state=episode["last_state"][0],
            actions=episode["actions"],
            rewards=episode["rewards"].squeeze(1),
        )

        # Log info
        episode_idx += 1

        # if we are in a step greater than warmup_steps, we can start training the model
        if step >= config["warmup_steps"]:
            print("Training model... @ step ", step)
            model.in_warmup_phase = False
            for _ in range(config["episode_length"]):
                metrics = model.update(replay_buffer, step)
                wandb.log(metrics)

        # Evaluate agent every while
        if episode_idx % (config["eval_interval"] // config["episode_length"]) == 0:
            metrics = evaluate(env, model, step)
            wandb.log(metrics)

    print("Training completed.")
    wandb.finish()

def evaluate(env, model, step):
    episode_rewards = []
    for i in range(config["eval_episodes"]):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state = torch.from_numpy(state).to(model.device).float()
            action = model.traject(state, step, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = terminated or truncated

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {i} - Reward: {episode_reward}")

    return {
        "average_reward": sum(episode_rewards) / len(episode_rewards),
        "min_reward": min(episode_rewards),
        "max_reward": max(episode_rewards),
    }

if __name__ == "__main__":
    train()