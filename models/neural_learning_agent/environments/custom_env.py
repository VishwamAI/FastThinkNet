import gym
from gym import spaces
import numpy as np


class CustomEnv(gym.Env):
    """A custom environment for our neural network agent to interact with."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        self.state = None

    def step(self, action):
        # Execute one time step within the environment
        self.state = np.random.randint(0, 255, (64, 64, 3))
        reward = np.random.rand()
        done = np.random.choice([True, False])
        info = {}
        return self.state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.random.randint(0, 255, (64, 64, 3))
        return self.state

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        if close:
            return
        print(f"Current state: {self.state}")


# Register the custom environment
gym.envs.registration.register(
    id="CustomEnv-v0",
    entry_point=(
        "models.neural_learning_agent.environments.custom_env:"
        "CustomEnv"
    ),
)
