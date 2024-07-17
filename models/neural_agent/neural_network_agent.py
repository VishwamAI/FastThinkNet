import tensorflow as tf
import numpy as np
from fastthinknet.base import BaseAgent


class NeuralNetworkAgent(BaseAgent):
    def __init__(
        self,
        input_shape=(64, 64, 3),
        action_space=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=self.input_shape
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.action_space, activation="linear"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
            loss="mse",
        )
        return model

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        q_values = self.model.predict(state[np.newaxis, ...])
        return np.argmax(q_values[0])

    def train(self, states, actions, rewards, next_states, dones):
        target_q_values = self.target_model.predict(next_states)
        max_target_q_values = np.max(target_q_values, axis=1)
        target_q_values = rewards + self.gamma * max_target_q_values * \
            (1 - dones)

        q_values = self.model.predict(states)
        q_values[np.arange(q_values.shape[0]), actions] = target_q_values

        self.model.fit(states, q_values, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.target_model = tf.keras.models.load_model(filepath)

    def reset(self):
        # Reset any episode-specific variables here
        pass