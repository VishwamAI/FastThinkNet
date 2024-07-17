import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from environments.custom_env import CustomEnv

def load_model(model_path):
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)

def run_single_episode(env, model, render=True):
    """Run and visualize a single episode."""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = np.argmax(model.predict(state.reshape(1, -1)))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
        state = next_state

        if render:
            env.render()

    return total_reward, steps

def calculate_performance_metrics(env, model, num_episodes=100):
    """Compute average reward, success rate, etc."""
    rewards = []
    success_count = 0

    for _ in range(num_episodes):
        reward, steps = run_single_episode(env, model, render=False)
        rewards.append(reward)
        if reward > 0:  # Assuming positive reward indicates success
            success_count += 1

    avg_reward = np.mean(rewards)
    success_rate = success_count / num_episodes

    return avg_reward, success_rate

def demonstrate_meta_learning(env, model):
    """Show adaptation to environment variations."""
    print("Original environment performance:")
    orig_avg_reward, orig_success_rate = calculate_performance_metrics(env, model, num_episodes=50)
    print(f"Average Reward: {orig_avg_reward:.2f}, Success Rate: {orig_success_rate:.2%}")

    # Modify environment (e.g., change reward structure)
    env.modify_reward_structure(new_reward_factor=0.5)

    print("\nPerformance after environment modification:")
    mod_avg_reward, mod_success_rate = calculate_performance_metrics(env, model, num_episodes=50)
    print(f"Average Reward: {mod_avg_reward:.2f}, Success Rate: {mod_success_rate:.2%}")

    # Allow model to adapt (simplified adaptation process)
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(model.predict(state.reshape(1, -1)))
            next_state, reward, done, _ = env.step(action)
            # Update model (simplified, in practice you'd use a proper training loop)
            model.fit(state.reshape(1, -1), np.array([action]), epochs=1, verbose=0)
            state = next_state

    print("\nPerformance after adaptation:")
    adapted_avg_reward, adapted_success_rate = calculate_performance_metrics(env, model, num_episodes=50)
    print(f"Average Reward: {adapted_avg_reward:.2f}, Success Rate: {adapted_success_rate:.2%}")

    return [orig_avg_reward, mod_avg_reward, adapted_avg_reward], [orig_success_rate, mod_success_rate, adapted_success_rate]

def demonstrate_self_play(env, model):
    """Showcase agent vs. agent gameplay."""
    num_games = 100
    agent1_wins = 0
    agent2_wins = 0
    draws = 0

    for _ in range(num_games):
        state = env.reset()
        done = False
        while not done:
            # Agent 1's turn
            action1 = np.argmax(model.predict(state.reshape(1, -1)))
            next_state, reward, done, _ = env.step(action1)
            if done:
                if reward > 0:
                    agent1_wins += 1
                elif reward < 0:
                    agent2_wins += 1
                else:
                    draws += 1
                break

            # Agent 2's turn
            action2 = np.argmax(model.predict((-next_state).reshape(1, -1)))  # Invert state for agent 2
            state, reward, done, _ = env.step(action2)
            if done:
                if reward > 0:
                    agent2_wins += 1
                elif reward < 0:
                    agent1_wins += 1
                else:
                    draws += 1

    print(f"Self-play results over {num_games} games:")
    print(f"Agent 1 wins: {agent1_wins}")
    print(f"Agent 2 wins: {agent2_wins}")
    print(f"Draws: {draws}")

    return agent1_wins, agent2_wins, draws

def visualize_results(rewards, success_rates):
    """Create plots for decision-making, state representations, and rewards."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(success_rates)
    plt.title('Success Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    plt.tight_layout()
    plt.savefig('demo_results.png')
    plt.show()

def main():
    # Set up the environment and load the model
    env = gym.make('CustomEnv-v0')
    model = load_model('path/to/trained_model.h5')

    # Run demonstrations
    print("Running single episode demonstration...")
    reward, steps = run_single_episode(env, model)
    print(f"Single episode result: Reward = {reward}, Steps = {steps}")

    print("\nCalculating performance metrics...")
    avg_reward, success_rate = calculate_performance_metrics(env, model)
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")

    print("\nDemonstrating meta-learning...")
    meta_rewards, meta_success_rates = demonstrate_meta_learning(env, model)

    print("\nDemonstrating self-play...")
    agent1_wins, agent2_wins, draws = demonstrate_self_play(env, model)

    print("\nVisualizing results...")
    visualize_results(meta_rewards, meta_success_rates)

if __name__ == "__main__":
    main()