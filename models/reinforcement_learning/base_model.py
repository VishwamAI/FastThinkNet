import tensorflow as tf
import numpy as np


class FastThinkNetRL:
    def __init__(self, state_dim, action_dim,
                 learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Policy network
        self.policy_network = self._build_network(
            state_dim, action_dim, "policy"
        )
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Value network
        self.value_network = self._build_network(state_dim, 1, "value")
        self.target_value_network = self._build_network(
            state_dim, 1, "target_value"
        )
        self.value_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        )

        # PPO hyperparameters
        self.epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def _build_network(self, input_dim, output_dim, name):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                64, activation="relu", input_shape=(input_dim,)
            ),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(
                output_dim,
                activation="softmax" if name == "policy" else None
            ),
        ])
        return model

    def choose_action(self, state):
        state = np.array(state).reshape(1, -1)
        action_probs = self.policy_network(state).numpy()[0]
        return np.random.choice(self.action_dim, p=action_probs)

    def collect_experience(self, env, num_episodes):
        experiences = []
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_experience = []
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_experience.append(
                    (state, action, reward, next_state, done)
                )
                state = next_state
            experiences.extend(episode_experience)
        return experiences

    @tf.function
    def update_policy(self, states, actions, old_probs, advantages):
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(states)
            selected_action_probs = tf.reduce_sum(
                action_probs * tf.one_hot(actions, self.action_dim),
                axis=1
            )
            ratio = selected_action_probs / old_probs
            clipped_ratio = tf.clip_by_value(
                ratio, 1 - self.epsilon, 1 + self.epsilon
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )
            entropy = -tf.reduce_sum(
                action_probs * tf.math.log(action_probs + 1e-8),
                axis=1
            )
            loss = policy_loss - self.entropy_coef * tf.reduce_mean(entropy)

        grads = tape.gradient(loss, self.policy_network.trainable_variables)
        self.policy_optimizer.apply_gradients(
            zip(grads, self.policy_network.trainable_variables)
        )
        return loss

    @tf.function
    def update_value_function(self, states, returns):
        with tf.GradientTape() as tape:
            predicted_values = self.value_network(states)
            loss = tf.keras.losses.MSE(returns, predicted_values)

        grads = tape.gradient(loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(
            zip(grads, self.value_network.trainable_variables)
        )
        return loss

    def train(self, env, num_episodes, gamma=0.99):
        for episode in range(num_episodes):
            experiences = self.collect_experience(env, 1)
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)

            # Compute returns and advantages
            values = self.value_network(states).numpy().flatten()
            next_values = self.value_network(next_states).numpy().flatten()
            returns = rewards + gamma * next_values * (1 - dones)
            advantages = returns - values

            # Update policy and value function
            policy_loss = self.update_policy(states, actions, advantages)
            value_loss = self.update_value_function(states, returns)

            if episode % 10 == 0:
                print(
                    f"Episode {episode}, "
                    f"Policy Loss: {policy_loss.numpy():.4f}, "
                    f"Value Loss: {value_loss.numpy():.4f}"
                )

    def integrate_with_dl_model(self, dl_model):
        # This method would be implemented to integrate with the deep learning
        # component. For example, it could use the DL model's output as part
        # of the state representation
        pass


# Example usage:
# env = gym.make('CartPole-v1')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# rl_model = FastThinkNetRL(state_dim, action_dim)
# rl_model.train(env, num_episodes=1000)

# Dummy comment to trigger CI/CD run
# New line to trigger git change detection
# Trivial change to trigger git detection