import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


# --- 1. Actor 神經網路 (負責輸出動作) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # 依照講義 Ch 10 架構：三個全連接層
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        # 使用 tanh 將輸出壓縮到 [-1, 1]，再乘上 max_action
        return self.max_action * torch.tanh(self.l3(a))


# --- 2. Critic 神經網路 (負責幫動作打分數) ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Critic 必須同時接收 State 與 Action
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        # 將狀態與動作在維度 1 拼接起來 (Concatenate)
        q = torch.cat([state, action], 1)
        q = torch.relu(self.l1(q))
        q = torch.relu(self.l2(q))
        return self.l3(q)


# --- 3. OU 雜訊 (Ornstein-Uhlenbeck Noise) ---
# 確保機械手臂探索時的動作是平滑連續的，避免馬達瘋狂抽搐
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


# --- 4. 經驗回放緩衝區 (Replay Buffer) ---
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))

    def add(self, state, action, reward, next_state, done):
        # 儲存每一次的互動經驗
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 隨機抽取一個 Batch 的資料來訓練神經網路
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
