import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class AttentionActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, num_heads):
        super().__init__()
        self.v = obs_shape[0]
        self.f = obs_shape[1]
        self.embed_dim = 64
        self.num_heads = num_heads

        self.feature_proj = nn.Linear(self.f, self.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True
        )
        flattened_dim = self.v * self.embed_dim

        self.critic = nn.Sequential(
            nn.Linear(flattened_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(flattened_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def extract_features(self, x):
        if len(x.shape) == 2:
            x = x.view(-1, self.v, self.f)
        proj = torch.relu(self.feature_proj(x))
        attn_out, _ = self.attention(proj, proj, proj)
        return attn_out.reshape(-1, self.v * self.embed_dim)

    def get_value(self, x):
        return self.critic(self.extract_features(x))

    def get_action_and_value(self, x, action=None):
        features = self.extract_features(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


class MlpActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        input_dim = np.prod(obs_shape)
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 320),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(320, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(320, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def get_value(self, x):
        return self.critic(self.feature_extractor(x))

    def get_action_and_value(self, x, action=None):
        features = self.feature_extractor(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


class MlpQNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.init_args = (obs_shape, action_dim)  # 保存初始化參數供 Target Network 複製
        input_dim = np.prod(obs_shape)

        self.q_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        return self.q_net(x)
