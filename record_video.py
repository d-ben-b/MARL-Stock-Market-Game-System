import os
import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

# ==========================================
# 0. 參數解析器設定
# ==========================================
parser = argparse.ArgumentParser(
    description="Record driving videos for trained RL models."
)
parser.add_argument(
    "--model_path", type=str, required=True, help="要載入的 .pth 模型路徑"
)
parser.add_argument(
    "--arch",
    type=str,
    default="attention",
    choices=["attention", "mlp"],
    help="模型架構 (attention 或 mlp)",
)
parser.add_argument(
    "--num_heads", type=int, default=4, help="Attention 頭數 (若為 attention 架構)"
)
parser.add_argument("--episodes", type=int, default=3, help="要錄製的回合數")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"準備錄製影片 - 模型: {args.model_path} | 架構: {args.arch}")


# ==========================================
# 1. 重新定義神經網路 (用於載入權重)
# ==========================================
class AttentionAgent(nn.Module):
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

    def get_action(self, x):
        features = self.extract_features(x)
        logits = self.actor(features)
        # 測試時直接取機率最大的動作 (Deterministic)
        return torch.argmax(logits, dim=1).item()


class MlpAgent(nn.Module):
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

    def get_action(self, x):
        features = self.feature_extractor(x)
        logits = self.actor(features)
        return torch.argmax(logits, dim=1).item()


# ==========================================
# 2. 環境設定與錄影 Wrapper
# ==========================================
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 40,
}

# 設定 render_mode 為 rgb_array 以供影片錄製
base_env = gym.make("highway-v0", render_mode="rgb_array")
base_env.unwrapped.configure(config)

# 從模型路徑提取檔名，作為影片資料夾名稱
model_name = os.path.basename(args.model_path).replace(".pth", "")
video_folder = f"./videos/{model_name}"
os.makedirs(video_folder, exist_ok=True)

# 包裝 RecordVideo
env = RecordVideo(base_env, video_folder=video_folder, episode_trigger=lambda x: True)
obs, info = env.reset()

# ==========================================
# 3. 載入模型權重
# ==========================================
if args.arch == "attention":
    agent = AttentionAgent(
        obs_shape=obs.shape, action_dim=env.action_space.n, num_heads=args.num_heads
    ).to(device)
else:
    agent = MlpAgent(obs_shape=obs.shape, action_dim=env.action_space.n).to(device)

try:
    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    agent.eval()
    print("✅ 成功載入模型權重！")
except FileNotFoundError:
    print(f"❌ 找不到檔案: {args.model_path}")
    exit()

# ==========================================
# 4. 開始錄影
# ==========================================
print(f"開始錄製 {args.episodes} 回合的影片...")

for ep in range(args.episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    score = 0

    while not (done or truncated):
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.get_action(obs_tensor)

        obs, reward, done, truncated, info = env.step(action)
        score += reward

    print(f"回合 {ep+1} 結束，獲得獎勵: {score:.2f}")

env.close()
print(f"🎥 錄影完成！影片已儲存於 {video_folder} 資料夾中。")
