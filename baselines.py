import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict


class RandomAgent:
    """隨機代理人：完全不學習，做為最低基準線"""

    def __init__(self, action_dim=4):
        self.action_dim = action_dim

    def get_action(self, state, noise_scale=0):
        # 無視 State，直接回傳隨機動作
        return np.random.uniform(-1, 1, size=(self.action_dim,))

    def update(self, replay_buffer, batch_size=64):
        # 隨機代理人不具備學習能力，直接 Pass
        pass


class QTableAgent:
    """Q-Table 代理人：展示離散演算法在連續金融環境中的侷限性"""

    def __init__(self, learning_rate=0.01, gamma=0.99):
        # Q-table 使用 defaultdict，遇到未見過的狀態預設 Q 值為 [0, 0, 0]
        self.q_table = defaultdict(lambda: np.zeros(3))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.995

    def _discretize_state(self, state):
        """
        核心痛點展示：為了解決維度災難，被迫捨棄 20 個訂單簿特徵，
        只保留現金、庫存、情緒，並進行粗略分箱。
        """
        cash = state[20]
        inventory = state[21]
        sentiment = state[22]

        # 粗略分箱 (Binning)
        cash_bin = 0 if cash < 50000 else (1 if cash < 150000 else 2)
        inv_bin = 0 if inventory < 500 else (1 if inventory < 1500 else 2)
        sent_bin = 0 if sentiment < -0.3 else (1 if sentiment < 0.3 else 2)

        return f"{cash_bin}_{inv_bin}_{sent_bin}"  # 組合為字串作為 Hash Key

    def get_action(self, state, noise_scale=None):
        state_key = self._discretize_state(state)

        # Epsilon-Greedy 探索策略
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(3)
        else:
            action_idx = np.argmax(self.q_table[state_key])

        # 衰減探索率
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

        # 將離散動作 (0, 1, 2) 轉換為環境需要的 4D 連續向量
        if action_idx == 0:
            # 離散動作 0: 強烈買進 (動能交易)
            return np.array([-0.9, 1.0, 0.0, 1.0])
        elif action_idx == 1:
            # 離散動作 1: 強烈賣出 (動能交易)
            return np.array([-0.9, -1.0, 0.0, 1.0])
        else:
            # 離散動作 2: 觀望
            return np.array([0.0, 0.0, 0.0, 0.0])

    def update(self, replay_buffer, batch_size=None):
        """提取 Buffer 中最新的一筆資料進行 Q-Learning 更新"""
        if len(replay_buffer) == 0:
            return

        # Q-Learning 是 Online/Off-policy 更新，此處簡化為只取最後一步經驗
        state, action, reward, next_state, done = replay_buffer.buffer[-1]

        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        # 反推當初執行的 action_idx
        if action[1] > 0.5:
            action_idx = 0
        elif action[1] < -0.5:
            action_idx = 1
        else:
            action_idx = 2

        # 傳統 Q-Learning 更新公式: $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max Q(s') - Q(s,a)]$
        best_next_q = np.max(self.q_table[next_state_key])
        td_target = reward + self.gamma * best_next_q * (1 - done)
        td_error = td_target - self.q_table[state_key][action_idx]

        self.q_table[state_key][action_idx] += self.lr * td_error


class DQNNetwork(nn.Module):
    """Ch 6: 基礎神經網路架構"""

    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),  # 輸出為各個離散動作的 Q 值
        )

    def forward(self, state):
        return self.net(state)


class DQNAgent:
    """Ch 7 & Ch 8: 結合 Deep Q-Learning 與 Soft Update 的代理人"""

    def __init__(
        self, state_dim=23, discrete_action_dim=3, lr=1e-3, gamma=0.99, tau=0.005
    ):
        self.discrete_action_dim = discrete_action_dim
        self.gamma = gamma
        self.tau = tau  # Ch 8: Soft Update 參數
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 實例化 Q 網路與 Target 網路
        self.q_net = DQNNetwork(state_dim, discrete_action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, discrete_action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 探索率設定
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, state, noise_scale=None):
        """Epsilon-Greedy 策略，並轉換為環境相容的 4D 向量"""
        # 探索
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.discrete_action_dim)
        # 利用 (神經網路決策)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
                action_idx = q_values.argmax().item()

        # 探索率衰減
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 將離散動作映射到連續 4D 向量 [策略, 方向, 價格, 數量]
        if action_idx == 0:
            # 離散動作 0: 買進 (動能交易，數量全滿)
            return np.array([-0.9, 1.0, 0.0, 1.0])
        elif action_idx == 1:
            # 離散動作 1: 賣出 (動能交易，數量全滿)
            return np.array([-0.9, -1.0, 0.0, 1.0])
        else:
            # 離散動作 2: 觀望
            return np.array([0.0, 0.0, 0.0, 0.0])

    def _map_continuous_to_discrete(self, action_vector):
        """將 Buffer 中的 4D 向量反推回離散 Index，供 Q-Learning 計算 Loss"""
        side_signal = action_vector[1]
        if side_signal > 0.5:
            return 0  # Buy
        elif side_signal < -0.5:
            return 1  # Sell
        else:
            return 2  # Hold

    def update(self, replay_buffer, batch_size=64):
        """Ch 7: DQN 經驗回放與梯度更新"""
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)

        # 將 4D 動作轉回離散 Index 列表
        action_indices = [self._map_continuous_to_discrete(a) for a in actions]
        action_indices = torch.LongTensor(action_indices).unsqueeze(1).to(self.device)

        # 計算 Current Q 值 (從網路輸出中挑出實際執行動作的 Q 值)
        current_q = self.q_net(states).gather(1, action_indices)

        # 計算 Target Q 值 (使用 Target 網路)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 計算 Loss 並更新網路
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Ch 8: Soft Update (緩慢更新 Target 網路，確保訓練穩定)
        for target_param, param in zip(
            self.target_net.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
