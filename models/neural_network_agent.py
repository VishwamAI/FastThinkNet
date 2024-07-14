import tensorflow as tf
import numpy as np


class NeuralNetworkAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    64, activation="relu", input_shape=(self.state_size,)
                ),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(self.action_size, activation="linear"),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss="mse",
        )
        return model

    def act(self, state, epsilon=0):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state[np.newaxis, ...])
        return np.argmax(q_values[0])

    def learn(self, state, action, reward, next_state, done):
        target = self.model.predict(state[np.newaxis, ...])
        if done:
            target[0][action] = reward
        else:
            q_future = np.max(
                self.model.predict(next_state[np.newaxis, ...])[0]
            )
            target[0][action] = reward + 0.99 * q_future
        self.model.fit(state[np.newaxis, ...], target, epochs=1, verbose=0)

    def train(
        self,
        env,
        episodes,
        max_steps,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        epsilon = epsilon_start
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = self.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                self.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            print(
                (f"Episode: {episode + 1}, Total Reward: {total_reward}, "
                 f"Epsilon: {epsilon:.2f}")
            )

    def evaluate(self, env, episodes):
        total_rewards = []
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                episode_reward += reward
            total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {episodes} episodes: {avg_reward}")
        return avg_reward

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

    def get_model_parameters(self):
        return self.model.get_weights()


# Helper function
def preprocess_state(state):
    return np.array(state, dtype=np.float32)