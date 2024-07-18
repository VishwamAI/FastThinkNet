import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Tuple


class FastThinkNetSelfPlay(nn.Module):
    def __init__(
        self,
        input_shape=(64, 64, 3),
        output_size=5,
        learning_rate=0.001
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(self._get_conv_output(input_shape), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.past_versions: List[nn.Module] = []

    def _get_conv_output(self, shape):
        batch_size = 1
        input_tensor = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_conv(input_tensor)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                break
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def act(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.model[-1].out_features - 1)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return torch.argmax(q_values).item()

    def generate_self_play_episode(
        self, env
    ) -> List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]:
        experiences = []
        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.act(state_tensor)
            next_state, reward, done, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            experiences.append((state_tensor, action, reward,
                                next_state_tensor, done))
            state = next_state

        return experiences

    def update_model(self, replay_buffer, batch_size: int = 32):
        if len(replay_buffer) < batch_size:
            return

        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.forward(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones.float()) * 0.99 * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

    def curriculum_learning(
        self,
        env,
        num_episodes: int,
        difficulty_increase_freq: int,
        batch_size: int = 32,
    ):
        replay_buffer = []
        for episode in range(num_episodes):
            if episode % difficulty_increase_freq == 0:
                env.increase_difficulty()

            experiences = self.generate_self_play_episode(env)
            replay_buffer.extend(experiences)
            self.update_model(replay_buffer, batch_size)

    def store_current_version(self):
        self.past_versions.append(self.state_dict())

    def load_random_past_version(self):
        if self.past_versions:
            past_version = random.choice(self.past_versions)
            self.load_state_dict(past_version)

    def integrate_with_components(
        self, deep_learning_model, rl_model, meta_learning_model
    ):
        """
        Integrate the self-play model with other components of FastThinkNet.
        """
        # Ensure all models are on the same device
        device = next(self.parameters()).device
        deep_learning_model.to(device)
        rl_model.to(device)
        meta_learning_model.to(device)

        # Use deep learning model for feature extraction
        self.feature_extractor = deep_learning_model.feature_extractor

        # Combine policies using an ensemble approach
        def ensemble_policy(state):
            self_play_action = self.act(state)
            rl_action = rl_model.act(state)
            meta_action = meta_learning_model.act(state)

            # Simple majority voting
            actions = [self_play_action, rl_action, meta_action]
            return max(set(actions), key=actions.count)

        self.integrated_act = ensemble_policy

        # Implement knowledge transfer
        def transfer_knowledge():
            # Transfer learned features from deep learning model
            self.model[0].weight.data = deep_learning_model.feature_extractor[
                0
            ].weight.data.clone()

            # Transfer policy from RL model
            self.model[-1].weight.data = (
                rl_model.policy_net[-1].weight.data.clone()
            )

        self.transfer_knowledge = transfer_knowledge

        print("Components integrated successfully.")


# Example usage
if __name__ == "__main__":
    input_shape = (64, 64, 3)
    output_size = 5
    self_play_model = FastThinkNetSelfPlay(input_shape, output_size)

    # Assuming we have an environment 'env' defined
    # env = YourEnvironment()

    # Generate self-play episode
    # experiences = self_play_model.generate_self_play_episode(env)

    # Update model
    # replay_buffer = []
    # replay_buffer.extend(experiences)
    # self_play_model.update_model(replay_buffer)

    # Curriculum learning
    # self_play_model.curriculum_learning(
    #     env,
    #     num_episodes=1000,
    #     difficulty_increase_freq=100
    # )

    # Store and load past versions
    # self_play_model.store_current_version()
    # self_play_model.load_random_past_version()

    # Integration with other components
    # deep_learning_model = None  # Placeholder for deep learning model
    # rl_model = None  # Placeholder for reinforcement learning model
    # meta_learning_model = None  # Placeholder for meta-learning model
    # self_play_model.integrate_with_components(
    #     deep_learning_model,
    #     rl_model,
    #     meta_learning_model
    # )

# End of base_model.py
