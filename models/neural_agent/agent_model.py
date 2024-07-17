import tensorflow as tf
import numpy as np


class NeuralNetworkAgent:
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu',
                                  input_shape=self.input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_actions, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def get_action(self, state):
        state = np.reshape(state, [1, *self.input_shape])
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, *self.input_shape])
        next_state = np.reshape(next_state, [1, *self.input_shape])
        target = reward
        if not done:
            target = reward + 0.99 * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = tf.keras.models.load_model(file_path)
