import argparse
import torch
import gymnasium
from gymnasium.wrappers import RecordVideo

from models.tdmpc import TDMPC

config = {
    "hidden_size": 256,
}

# runs 1 episode of inference using the trained model
def inferance(weights_path: str, output_dir: str): 
    env = gymnasium.make("Reacher-v5", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=output_dir, episode_trigger=lambda x: True)

    weights = torch.load(weights_path, map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'))
    model = TDMPC(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        hidden_size=config["hidden_size"],
    )
    model.load_state_dict(weights)
    model.eval()

    state, _ = env.reset()
    done = False

    step = 0

    print("Starting inference...")
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            state = torch.from_numpy(state).to(model.device).float()
            action: torch.Tensor = model(state)

        state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = terminated or truncated

        total_reward += reward

        step += 1

        if step % 10 == 0:
            print(f"Step: {step}, Reward: {reward}")

    print(f"Total reward: {total_reward}")
    print(f"Saved video to {output_dir}")

    env.close()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--output_dir", type=str, default="./videos/inferance/", help="Directory to save the output videos")
    args = parser.parse_args()

    inferance(weights_path=args.weights, output_dir=args.output_dir)