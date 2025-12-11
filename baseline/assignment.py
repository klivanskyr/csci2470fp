import os
import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import tensorflow as tf
try:
    from reinforce import Reinforce
    from reinforce_with_baseline import ReinforceWithBaseline
    from deep_q import DeepQModel
except:
    print("Please make sure to have the 'reinforce.py', 'reinforce_with_baseline.py', 'deep_q.py' files in the same directory as this file.")
from matplotlib.pyplot import plot, xlabel, ylabel, title, grid, show
from numpy import arange


def visualize_episode(model, env_name):
    """
    HELPER - do not edit.
    Visualize model actions for one episode in a human window.
    NOTE: This creates a new env in 'human' render mode so it will NOT be the wrapped env used for recording.
    """
    done = False
    env = gym.make(env_name, render_mode="human")
    state, _ = env.reset()
    while not done:
        newState = np.reshape(state, [1, state.shape[0]])
        state_tensor = tf.convert_to_tensor(newState, dtype=tf.float32)
        # Handle continuous vs discrete action spaces
        if hasattr(env.action_space, 'n'):
            prob = model.call(state_tensor)
            if isinstance(prob, tuple):
                action, _ = model.sample_action(state_tensor)
                action = action[0].numpy()
                if hasattr(env.action_space, 'low'):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
            else:
                newProb = np.reshape(prob, prob.shape[1])
                if np.sum(newProb) != 1:
                    action = np.argmax(newProb)
                else:
                    action = np.random.choice(np.arange(newProb.shape[0]), p=newProb)
        else:
            action, _ = model.sample_action(state_tensor)
            action = action[0].numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

        state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
    env.close()

def visualize_data(total_rewards):
    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()

def discount(rewards, discount_factor=.99):
    length = len(rewards)
    tracker = [0] * length
    running_total = 0
    for i in reversed(range(length)):
        running_total = rewards[i] + discount_factor * running_total
        tracker[i] = running_total
    return tracker

def generate_trajectory(env, model):
    states = []
    actions = []
    rewards = []
    state, _ = env.reset()
    done = False

    while not done:
        states.append(state)
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            action_low = tf.constant(env.action_space.low, dtype=tf.float32)
            action_high = tf.constant(env.action_space.high, dtype=tf.float32)
            action, _ = model.sample_action(state_tensor, action_low, action_high)
        else:
            action, _ = model.sample_action(state_tensor)
        action = action[0].numpy()
        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.zeros_like(action)
            if hasattr(env.action_space, 'low'):
                action = np.clip(action, env.action_space.low, env.action_space.high)
        actions.append(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        state = next_state

    return states, actions, rewards

def train_reinforce_episode(env, model):
    states, actions, rewards = generate_trajectory(env, model)
    discounted_rewards = discount(rewards)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    discounted_rewards = np.array(discounted_rewards, dtype=np.float32)

    action_low = None
    action_high = None
    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
        action_low = tf.constant(env.action_space.low, dtype=tf.float32)
        action_high = tf.constant(env.action_space.high, dtype=tf.float32)

    with tf.GradientTape() as tape:
        loss = model.loss_func(states, actions, discounted_rewards, action_low, action_high)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 0.5) if g is not None else g for g in grads]
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return np.sum(rewards)

def train_deep_q_episode(env, model, batch_size, memory, epsilon=.1):
    state, _ = env.reset()
    done = False
    episode_rewards = []
    num_batches = 10
    tf.random.set_seed(0)

    while not done:
        if tf.random.uniform([1]) < epsilon:
            action = env.action_space.sample()
        else:
            state_reshaped = np.expand_dims(state, axis=0)
            q_values = model.call(state_reshaped)
            action = np.argmax(q_values.numpy()[0])

        next_state, reward, term, trunc, _ = env.step(action)
        env.render()
        done = term or trunc

        memory.append((state.copy(), action, reward, next_state.copy(), done))
        episode_rewards.append(reward)
        state = next_state

    if len(memory) >= batch_size:
        for _ in range(num_batches):
            batch_indices = np.random.choice(len(memory), size=batch_size, replace=False)
            batch = [memory[i] for i in batch_indices]
            states_batch = np.array([item[0] for item in batch])
            actions_batch = np.array([item[1] for item in batch])
            rewards_batch = np.array([item[2] for item in batch], dtype=np.float32)
            next_states_batch = np.array([item[3] for item in batch])
            done_batch = np.array([item[4] for item in batch], dtype=bool)

            states_tensor = tf.convert_to_tensor(states_batch, dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(actions_batch, dtype=tf.int32)
            rewards_tensor = tf.convert_to_tensor(rewards_batch, dtype=tf.float32)
            next_states_tensor = tf.convert_to_tensor(next_states_batch, dtype=tf.float32)
            done_tensor = tf.convert_to_tensor(done_batch, dtype=tf.bool)

            batch_tuple = (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            with tf.GradientTape() as tape:
                loss = model.loss_func(batch_tuple)

            gradients = tape.gradient(loss, model.model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.model.trainable_variables))

        model.target_model.set_weights(model.model.get_weights())

    return sum(episode_rewards), memory

def train(env, model, memory=None, epsilon=.1):
    if isinstance(model, DeepQModel):
        if memory is None:
            memory = []
            state, _ = env.reset()
            for _ in range(50):
                action = env.action_space.sample()
                next_state, reward, term, trunc, _ = env.step(action)
                env.render()
                done = term or trunc
                memory.append((state.copy(), action, reward, next_state.copy(), done))
                if done:
                    state, _ = env.reset()
                else:
                    state = next_state

        if len(memory) > 1000:
            memory = memory[-1000:]

        batch_size = 64
        return train_deep_q_episode(env, model, batch_size, memory, epsilon)
    else:
        return train_reinforce_episode(env, model)

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE", "DEEP_Q"}:
        print("USAGE: python assignment.py <Model Type> <optional: env_name>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE/DEEP_Q]")
        exit()

    # Determine env_name
    if len(sys.argv) == 3:
        env_name = sys.argv[2]
    else:
        env_name = "Reacher-v5"

    # Make sure the videos directory exists
    os.makedirs("./videos", exist_ok=True)

    try:
        env = gym.make(env_name, render_mode="rgb_array")
        # wrap for recording; record every 100th episode
        env = RecordVideo(env, video_folder="./videos", name_prefix="run", episode_trigger=lambda ep: ep % 100 == 0)
    except Exception as e:
        print(f"Incorrect Environment Name or failed to create environment '{env_name}': {e}")
        return

    # rest of setup
    state_shape = env.observation_space.shape
    print("State size: ", state_shape)
    state_size = state_shape[0]

    if hasattr(env.action_space, 'n'):
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    print("Action space: ", env.action_space)
    print("Number of actions: ", num_actions)

    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions)
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)
    elif sys.argv[1] == "DEEP_Q":
        model = DeepQModel(state_size, num_actions)

    totalReward = []
    num_episodes = 1500
    viz_every = 100
    memory = None
    for episode in range(num_episodes):
        print(episode, end="\r")
        if sys.argv[1] == "DEEP_Q":
            reward, memory = train(env, model, memory=memory, epsilon=1 - episode / num_episodes)
        else:
            reward = train(env, model)
        totalReward.append(reward)
        # Keep this commented out to avoid issues with video recording
        # if episode % viz_every == 0:
        #     visualize_episode(model, env_name)

    # Keep this commented out to avoid issues with video recording
    # visualize_episode(model, env_name)

    env.close()
    print(sum(totalReward)/len(totalReward)) 

    visualize_data(totalReward)

if __name__ == '__main__':
    main()
