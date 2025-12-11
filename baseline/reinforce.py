import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but having this parameter may streamline your implementation.
        :param num_actions: number of actions in an environment. You do need to use this in your implementation.
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
        # TODO: Define network parameters and optimizer
        dense1_size = 64
        dense2_size = 64
        
        learning_rate = 3e-4  # Lower learning rate for more stable training
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Shared layers for policy
        self.shared_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(dense1_size, activation="relu"),
            tf.keras.layers.Dense(dense2_size, activation="relu")
        ])
        
        # Policy head: outputs mean and log_std for Gaussian distribution (continuous actions)
        # Use tanh activation and small initial weights to keep actions bounded
        self.mean_layer = tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros()
        )
        # Initialize log_std to allow more exploration initially
        self.log_std_layer = tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            bias_initializer=tf.keras.initializers.Constant(-0.5)  # Start with std ~0.6 for more exploration
        )

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this ~
        shared = self.shared_layers(states)
        mean = self.mean_layer(shared)
        # Use tanh to bound mean to [-1, 1], then we'll scale in sample_action if needed
        mean = tf.tanh(mean)
        log_std = self.log_std_layer(shared)
        # Clamp log_std for numerical stability (std between ~0.00002 and ~7.4)
        log_std = tf.clip_by_value(log_std, -10, 2)
        return mean, log_std

    def sample_action(self, states, action_low=None, action_high=None):
        """
        Samples an action from the policy distribution.
        
        :param states: A state or batch of states
        :param action_low: Lower bound for actions (optional, for scaling)
        :param action_high: Upper bound for actions (optional, for scaling)
        :return: A tuple (action, log_prob) where action is sampled and log_prob is its log probability
        """
        mean, log_std = self.call(states)
        std = tf.exp(log_std)
        
        # Sample from standard normal and transform
        noise = tf.random.normal(tf.shape(mean))
        action = mean + std * noise
        
        # Scale action to environment bounds if provided
        if action_low is not None and action_high is not None:
            # Scale from [-1, 1] (tanh output) to [action_low, action_high]
            action = (action + 1.0) / 2.0  # [0, 1]
            action = action_low + (action_high - action_low) * action
            # Also scale mean for log prob calculation
            mean_scaled = (mean + 1.0) / 2.0
            mean_scaled = action_low + (action_high - action_low) * mean_scaled
            # Scale std proportionally
            std_scaled = std * (action_high - action_low) / 2.0
        else:
            mean_scaled = mean
            std_scaled = std
        
        # Compute log probability: log N(x|mu, sigma) = -0.5 * log(2*pi*sigma^2) - 0.5 * ((x-mu)/sigma)^2
        # Use scaled values for proper probability calculation
        log_prob = -0.5 * (tf.math.log(2 * np.pi) + 2 * tf.math.log(std_scaled + 1e-8) + tf.square((action - mean_scaled) / (std_scaled + 1e-8)))
        # Sum over action dimensions to get total log prob
        log_prob = tf.reduce_sum(log_prob, axis=-1)
        return action, log_prob

    def loss_func(self, states, actions, discounted_rewards, action_low=None, action_high=None):
        # Forward pass
        mean, log_std = self.call(states)
        std = tf.exp(log_std)

        # Scale mean/std if environment bounds provided
        if action_low is not None and action_high is not None:
            # Convert mean from [-1,1] → env range
            mean_scaled = (mean + 1.0) / 2.0
            mean_scaled = action_low + (action_high - action_low) * mean_scaled

            # Scale std proportionally
            std_scaled = std * (action_high - action_low) / 2.0
        else:
            mean_scaled = mean
            std_scaled = std

        # Gaussian log probability
        var = std_scaled ** 2
        log_probs = -0.5 * (
            ((actions - mean_scaled) ** 2) / var +
            2 * tf.math.log(std_scaled + 1e-8) +
            tf.math.log(2 * np.pi)
        )

        # Sum over action dimensions
        log_probs = tf.reduce_sum(log_probs, axis=1)

        # Policy gradient loss
        loss = -tf.reduce_mean(log_probs * discounted_rewards)

        return loss
