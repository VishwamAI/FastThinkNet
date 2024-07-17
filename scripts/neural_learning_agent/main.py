import tensorflow as tf
import numpy as np
import gym
from environments.custom_env import CustomEnv
import copy
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import argparse
from tqdm import tqdm
import json

def create_model(input_shape, action_space):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(action_space, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train_model(model, env, episode_rewards, episodes=2000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=32):
    target_model = tf.keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    epsilon = epsilon_start
    replay_buffer = []

    for episode in tqdm(range(episodes)):
        state = env.reset()
        state = np.reshape(state, [1, 64, 64, 3])
        done = False
        episode_reward = 0
        states, next_states, rewards, actions = [], [], [], []

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(state))

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 64, 64, 3])

            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            actions.append(action)

            state = next_state
            episode_reward += reward

            if len(states) >= batch_size:
                update_model(model, target_model, np.array(states), np.array(next_states),
                             np.array(rewards), np.array(actions), gamma)
                states, next_states, rewards, actions = [], [], [], []

        if states:  # Handle remaining samples
            update_model(model, target_model, np.array(states), np.array(next_states),
                         np.array(rewards), np.array(actions), gamma)

        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)

    return model, episode_rewards

def update_model(model, target_model, states, next_states, rewards, actions, gamma):
    reshaped_next_states = next_states.reshape(-1, 64, 64, 3)
    reshaped_states = states.reshape(-1, 64, 64, 3)
    targets = rewards + gamma * np.max(target_model.predict(reshaped_next_states), axis=1)
    targets = np.column_stack([targets] * model.output_shape[-1])
    targets[np.arange(len(actions)), actions] = rewards + gamma * np.max(target_model.predict(reshaped_next_states), axis=1)
    model.fit(reshaped_states, targets, epochs=1, verbose=0)





def test_model(model, env, episodes=100):
    total_reward = 0
    evaluation_rewards = []
    for _ in tqdm(range(episodes), desc="Testing episodes"):
        state = env.reset()
        state = np.reshape(state, [1, 64, 64, 3])
        episode_reward = 0
        done = False
        while not done:
            action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 64, 64, 3])
            state = next_state
            episode_reward += reward
        total_reward += episode_reward
        evaluation_rewards.append(episode_reward)

    avg_reward = total_reward / episodes
    print(f"Evaluation complete. Average reward over {episodes} episodes: {avg_reward}")

    # Save evaluation results
    with open('evaluation_rewards.json', 'w') as f:
        json.dump(evaluation_rewards, f)

    return avg_reward, evaluation_rewards

def main():
    parser = argparse.ArgumentParser(description="Neural Learning Agent")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--model", type=str, default="trained_model.h5", help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes for testing")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    env = gym.make('CustomEnv-v0')

    if args.test:
        model = tf.keras.models.load_model(args.model)
        test_model(model, env, episodes=args.episodes)
    else:
        # Create model
        state_shape = (64, 64, 3)
        action_space = env.action_space.n
        model = create_model(state_shape, action_space)

        # Initialize list to store episode rewards
        episode_rewards = []

        # Train model with improved parameters
        episodes = 100  # Reduced from 2000
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        model, episode_rewards = train_model(model, env, episode_rewards, episodes=episodes,
                                             epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                                             epsilon_decay=epsilon_decay, batch_size=args.batch_size)

        # Save episode rewards and model
        with open('episode_rewards.json', 'w') as f:
            json.dump(episode_rewards, f)

        model.save('trained_model.h5')

        # Evaluate the trained model
        test_model(model, env)

    # Close the environment
    env.close()

if __name__ == '__main__':
    main()