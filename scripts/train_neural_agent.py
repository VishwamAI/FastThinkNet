import os
import sys

print("Current working directory:", os.getcwd())
print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))
print("sys.path:", sys.path)

import argparse
import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from models.meta_learning.base_model import FastThinkNetMeta as BaseModel
from models.self_play.base_model import FastThinkNetSelfPlay as SelfPlayBaseModel
from config import Config

# Add this import to ensure the custom environment is registered
import models.neural_learning_agent.environments.custom_env

print(
    "Contents of 'models' directory:", os.listdir(os.path.join(os.getcwd(), "models"))
)


def create_model(input_shape, action_space):
    model = keras.Sequential(
        [
            keras.layers.Dense(24, activation="relu", input_shape=input_shape),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(action_space, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss="mse")
    return model


def train_model(model, env, episodes, batch_size):
    for episode in range(episodes):
        env.reset()
        done = False
        while not done:
            # Replace with action selection logic
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            # Implement your training logic here
        if episode % 10 == 0:
            logging.info(f"Episode {episode} completed")
    return model


def update_model(model, memory, batch_size):
    # Implement your model update logic here
    pass


def test_model(model, env, episodes):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(model.predict(np.array([state]))[0])
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        logging.info(f"Test Episode {episode} - Total Reward: {total_reward}")


def main():
    parser = argparse.ArgumentParser(
        description="Neural Learning Agent for FastThinkNet"
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument(
        "--model",
        type=str,
        default="trained_model.h5",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for training/testing",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--use_neural_agent",
        action="store_true",
        help="Use Neural Learning Agent instead of base model",
    )
    parser.add_argument(
        "--use_self_play",
        action="store_true",
        help="Use Self-Play base model instead of Meta-Learning base model",
    )
    args = parser.parse_args()

    config = Config()
    env = gym.make("CustomEnv-v0")

    if args.use_neural_agent:
        if args.test:
            model = tf.keras.models.load_model(args.model)
            test_model(model, env, episodes=args.episodes)
        else:
            model = create_model(env.observation_space.shape, env.action_space.n)
            trained_model = train_model(
                model, env, episodes=args.episodes, batch_size=args.batch_size
            )
            trained_model.save(args.model)
            logging.info(f"Trained model saved to {args.model}")
            test_model(trained_model, env, episodes=10)
    else:
        # Use FastThinkNet's base model
        if args.use_self_play:
            model = SelfPlayBaseModel(config)
        else:
            model = BaseModel(config)
        if args.test:
            model.load(args.model)
            model.test(env, episodes=args.episodes)
        else:
            model.train(env, episodes=args.episodes)
            model.save(args.model)
            logging.info(f"Trained base model saved to {args.model}")
            model.test(env, episodes=10)

    env.close()


if __name__ == "__main__":
    main()
