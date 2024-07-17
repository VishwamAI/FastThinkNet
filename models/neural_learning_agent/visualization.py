import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_learning_curve(episodes, rewards):
    """
    Plot the learning curve (reward vs. episode number).

    Args:
    episodes (list): List of episode numbers.
    rewards (list): List of corresponding rewards.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards)
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

def plot_action_distribution(actions):
    """
    Visualize the distribution of actions taken by the agent.

    Args:
    actions (list): List of actions taken by the agent.
    """
    unique, counts = np.unique(actions, return_counts=True)
    plt.figure(figsize=(8, 6))
    plt.bar(unique, counts)
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.show()

def plot_state_heatmap(states):
    """
    Create a heatmap of the agent's visited states.

    Args:
    states (list): List of states visited by the agent.
    """
    state_counts = {}
    for state in states:
        state_tuple = tuple(state)
        state_counts[state_tuple] = state_counts.get(state_tuple, 0) + 1

    max_x = max(state[0] for state in state_counts.keys())
    max_y = max(state[1] for state in state_counts.keys())

    heatmap = np.zeros((max_y + 1, max_x + 1))
    for state, count in state_counts.items():
        heatmap[state[1], state[0]] = count

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visit Count')
    plt.title('State Visitation Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def animate_episode(states, actions):
    """
    Generate an animation of a single episode, showing the agent's movement and decisions.

    Args:
    states (list): List of states visited by the agent in the episode.
    actions (list): List of actions taken by the agent in the episode.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, max(state[0] for state in states) + 1)
    ax.set_ylim(0, max(state[1] for state in states) + 1)
    ax.set_title('Agent Movement')

    agent, = ax.plot([], [], 'ro', markersize=10)

    def init():
        agent.set_data([], [])
        return agent,

    def animate(i):
        agent.set_data(states[i][0], states[i][1])
        ax.set_title(f'Step {i}, Action: {actions[i]}')
        return agent,

    anim = plt.animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(states), interval=500, blit=True)
    plt.close(fig)
    return anim

if __name__ == "__main__":
    # Example usage with dummy data
    episodes = list(range(100))
    rewards = np.random.randn(100).cumsum()
    plot_learning_curve(episodes, rewards)

    actions = np.random.randint(0, 4, 1000)
    plot_action_distribution(actions)

    states = [(np.random.randint(0, 10), np.random.randint(0, 10)) for _ in range(1000)]
    plot_state_heatmap(states)

    episode_states = [(i, i) for i in range(10)]
    episode_actions = np.random.randint(0, 4, 10)
    anim = animate_episode(episode_states, episode_actions)
    # To save the animation:
    # anim.save('episode_animation.gif', writer='pillow')