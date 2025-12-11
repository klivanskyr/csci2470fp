import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DeepQModel(tf.keras.Model):

    def __init__(self, state_size, num_actions):
        super(DeepQModel, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        # TODO: Define network parameters and optimizer

        # We require that you use tf.keras.Sequential to define the model and call it self.model
        #   (This is for auto-grading purposes)
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_actions)
        ])

        self.model.build(input_shape=(state_size,))
        
        # We require that you call your target model self.target_model
        #    (This is for auto-grading purposes)
        #    Hints: You can clone the model using tf.keras.models.clone_model
        #           You can get the weights of model using get_weights
        #           You can set the weights of target_model using set_weights

        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, states):
        return self.model(states)
    

    def loss_func(self, batch, discount_factor = 0.99):
        
        # Compute the loss for the agent
        
        states, actions, rewards, next_states, done = batch
        q_values = self.model(states)
        q_tilda = self.target_model(next_states)
        
        batch_indices = tf.range(tf.shape(actions)[0])
        indices = tf.stack([batch_indices, actions], axis=1)
        q_values_selected = tf.gather_nd(q_values, indices)
        

        next_q_max = tf.reduce_max(q_tilda, axis=1)
        done_float = tf.cast(done, tf.float32)
        target_q = rewards + discount_factor * next_q_max * (1 - done_float)
        
        loss = tf.square(q_values_selected - target_q)
        return tf.reduce_mean(loss)