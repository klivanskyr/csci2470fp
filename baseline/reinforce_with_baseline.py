import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but having this parameter may streamline your implementation.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size
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
        
        self.critic_model = tf.keras.Sequential([  
            tf.keras.layers.Dense(dense1_size, activation="relu"),
            tf.keras.layers.Dense(dense2_size, activation="relu"),
            
            tf.keras.layers.Dense(1)
        ])

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action distribution parameters.
        For continuous actions, returns mean and log_std of a Gaussian distribution.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A tuple (mean, log_std) where each is [episode_length, num_actions]
        """
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
    
    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        return self.critic_model(states)
    
    def loss_func(self, states, actions, discounted_rewards, action_low=None, action_high=None):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 2, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length, action_dim] array for continuous)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :param action_low: Lower bound of action space (optional)
        :param action_high: Upper bound of action space (optional)
        :return: loss, a TensorFlow scalar
        """
        mean, log_std = self.call(states)
        std = tf.exp(log_std)
        
        # Actions are stored in the environment's scale, but mean is in [-1, 1] from tanh
        # Scale mean and std to match the action scale for proper log prob calculation
        if action_low is not None and action_high is not None:
            # Scale mean from [-1, 1] to [action_low, action_high]
            mean_scaled = (mean + 1.0) / 2.0  # [0, 1]
            mean_scaled = action_low + (action_high - action_low) * mean_scaled
            # Scale std proportionally
            std_scaled = std * (action_high - action_low) / 2.0
        else:
            # If no bounds provided, assume actions are already in [-1, 1] range
            mean_scaled = mean
            std_scaled = std
        
        # Compute log probability: log N(x|mu, sigma) = -0.5 * log(2*pi*sigma^2) - 0.5 * ((x-mu)/sigma)^2
        # Add small epsilon to std for numerical stability
        log_probs = -0.5 * (tf.math.log(2 * np.pi) + 2 * tf.math.log(std_scaled + 1e-8) + tf.square((actions - mean_scaled) / (std_scaled + 1e-8)))
        log_probs = tf.reduce_sum(log_probs, axis=-1)  # Sum over action dimensions
        
        value = tf.squeeze(self.value_function(states))
        discounted_rewards_tensor = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
        adv = discounted_rewards_tensor - value
        
        # Normalize advantages for more stable training
        adv_mean = tf.reduce_mean(adv)
        adv_std = tf.math.reduce_std(adv) + 1e-8
        adv_normalized = (adv - adv_mean) / adv_std
        
        # Actor loss: negative log prob weighted by normalized advantage
        act_loss = -tf.reduce_sum(log_probs * tf.stop_gradient(adv_normalized))
        
        # Add entropy bonus to encourage exploration (entropy of Gaussian: 0.5 * log(2*pi*e*sigma^2))
        # Higher entropy = more exploration
        entropy = tf.reduce_sum(0.5 * (tf.math.log(2 * np.pi * np.e) + 2 * log_std))
        entropy_bonus = 0.01 * entropy  # Small entropy bonus to encourage exploration
        act_loss = act_loss - entropy_bonus
        
        # Critic loss: MSE between value estimates and returns
        crit_loss = tf.reduce_sum(tf.square(adv))
        
        return act_loss + crit_loss