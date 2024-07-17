import tensorflow as tf
import numpy as np

class NeuralNetworkAgent:
    def __init__(self, input_shape=(64, 64, 3), action_space=2):
        self.input_shape = input_shape
        self.action_space = action_space
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=self.input_shape, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(self.action_space, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def learn(self, initial_state, action, reward, next_state, done):
        initial_state = np.expand_dims(initial_state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        gamma = 0.99  # You might want to make this a class attribute or parameter
        self.update(self.model, initial_state, next_state, [reward], [action], gamma)

    def train(self, env, episodes=2000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=32):
        epsilon = epsilon_start
        memory = []
        episode_rewards = []

        for episode in range(episodes):
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            total_reward = 0

            while not done:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = self.model.predict(state)
                    action = np.argmax(q_values[0])

                next_state, reward, done, _ = env.step(action)
                next_state = np.expand_dims(next_state, axis=0)
                total_reward += reward

                memory.append((state, action, reward, next_state, done))
                state = next_state

                if len(memory) > batch_size:
                    batch = np.random.choice(len(memory), batch_size, replace=False)
                    states, actions, rewards, next_states, dones = zip(*[memory[i] for i in batch])

                    states = np.concatenate(states)
                    next_states = np.concatenate(next_states)

                    self.update(self.model, states, next_states, rewards, actions, gamma)

            episode_rewards.append(total_reward)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if episode % 100 == 0:
                print(f"Episode: {episode}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}, Epsilon: {epsilon:.2f}")

        return episode_rewards

    def update(self, target_model, states, next_states, rewards, actions, gamma):
        q_values = self.model.predict(states)
        next_q_values = target_model.predict(next_states)

        for i in range(len(states)):
            q_values[i][actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])

        self.model.fit(states, q_values, verbose=0)

    def evaluate(self, env, episodes=100):
        evaluation_rewards = []

        for _ in range(episodes):
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            total_reward = 0

            while not done:
                q_values = self.model.predict(state)
                action = np.argmax(q_values[0])
                next_state, reward, done, _ = env.step(action)
                next_state = np.expand_dims(next_state, axis=0)
                total_reward += reward
                state = next_state

            evaluation_rewards.append(total_reward)

        avg_reward = np.mean(evaluation_rewards)
        return avg_reward, evaluation_rewards

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

    def get_model_parameters(self):
        return self.model.get_weights()