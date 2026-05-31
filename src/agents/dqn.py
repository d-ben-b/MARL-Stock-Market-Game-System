import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQNAgent:
    def __init__(
        self,
        network,
        learning_rate,
        gamma=0.99,
        batch_size=256,
        buffer_size=50000,
        target_update_freq=500,
    ):
        self.device = next(network.parameters()).device
        self.q_network = network
        self.target_network = type(network)(*network.init_args).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.buffer = deque(maxlen=buffer_size)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.steps = 0

    def select_action(self, obs, env_action_space):
        # Epsilon-Greedy 策略
        if random.random() < self.epsilon:
            return env_action_space.sample()

        with torch.no_grad():
            q_values = self.q_network(obs)
            return torch.argmax(q_values).item()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.Tensor(np.array(obs)).to(self.device)
        actions = torch.LongTensor(actions).view(-1, 1).to(self.device)
        rewards = torch.Tensor(rewards).view(-1, 1).to(self.device)
        next_obs = torch.Tensor(np.array(next_obs)).to(self.device)
        dones = torch.Tensor(dones).view(-1, 1).to(self.device)

        # 計算當前 Q 值
        q_values = self.q_network(obs).gather(1, actions)

        # 計算目標 Q 值 (Double DQN 機制)
        with torch.no_grad():
            next_actions = self.q_network(next_obs).argmax(1, keepdim=True)
            target_q_values = self.target_network(next_obs).gather(1, next_actions)
            expected_q_values = rewards + self.gamma * target_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 衰減 Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()
