import os
import csv
import time
import argparse
import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 0. 參數解析器 (Argparse) 設定
# ==========================================
parser = argparse.ArgumentParser(
    description="Train RL Agent with strict variable control for Ablation Studies."
)
# --- 網路架構與注意力控制 ---
parser.add_argument(
    "--arch",
    type=str,
    default="attention",
    choices=["attention", "mlp"],
    help="選擇神經網路架構: attention (注意力機制), mlp (傳統多層感知機)",
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=4,
    help="Multi-head Attention 的頭數 (預設: 4)。僅在 --arch attention 時有效。",
)
# --- 獎勵與優化控制 ---
parser.add_argument(
    "--style",
    type=str,
    default="base",
    choices=["base", "conservative", "aggressive"],
    help="選擇駕駛性格: base (基準), conservative (保守), aggressive (激進)",
)
parser.add_argument(
    "--minibatch_size",
    type=int,
    default=256,
    help="PPO 更新的批次大小，較大可平滑梯度 (預設: 256)",
)
parser.add_argument(
    "--ent_coef",
    type=float,
    default=0.005,
    help="熵獎勵權重，控制隨機探索程度 (預設: 0.005)",
)
parser.add_argument(
    "--disable_lr_decay",
    action="store_true",
    help="加入此參數則【關閉】學習率線性衰減 (預設為開啟)",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="",
    help="實驗名稱後綴，避免覆蓋舊檔案 (例如: v1_stable)",
)

args = parser.parse_args()

print(f"====================================")
print(f" 🧪 嚴謹控制變因 - 實驗設定確認:")
print(
    f" - 網路架構: {args.arch.upper()} (Heads: {args.num_heads if args.arch == 'attention' else 'N/A'})"
)
print(f" - 駕駛風格: {args.style.upper()}")
print(f" - 實驗後綴: {args.exp_name if args.exp_name else '無'}")
print(f" - Batch Size: {args.minibatch_size}")
print(f" - Entropy Coef: {args.ent_coef}")
print(f" - LR Decay: {'OFF' if args.disable_lr_decay else 'ON'}")
print(f"====================================")

# ==========================================
# 1. 超參數設定 (動態接收 Argparse)
# ==========================================
TOTAL_TIMESTEPS = 100000
NUM_STEPS = 1024
MINIBATCH_SIZE = args.minibatch_size
EPOCHS = 10
LEARNING_RATE = 5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = args.ent_coef
VF_COEF = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 2. 定義神經網路 (提供 Attention 與 MLP 雙架構)
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

    def get_value(self, x):
        features = self.extract_features(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.extract_features(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


class MlpAgent(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        input_dim = np.prod(obs_shape)  # 5x5 = 25

        # 控制參數體積與 Attention Agent 一致 (最終輸出 320 維)
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
        features = self.feature_extractor(x)
        return self.critic(features)

    def get_action_and_value(self, x, action=None):
        features = self.feature_extractor(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


# ==========================================
# 3. 環境與日誌初始化
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

if args.style == "conservative":
    config.update(
        {
            "collision_reward": -2.0,
            "reward_speed_range": [10, 20],
            "lane_change_reward": -0.5,
        }
    )
elif args.style == "aggressive":
    config.update(
        {
            "collision_reward": -0.5,
            "reward_speed_range": [30, 40],
            "high_speed_reward": 1.0,
        }
    )
else:
    config.update({"collision_reward": -1.0, "reward_speed_range": [20, 30]})

env = gym.make("highway-v0")
env.unwrapped.configure(config)
obs, info = env.reset()

# 動態實例化模型
if args.arch == "attention":
    agent = AttentionAgent(
        obs_shape=obs.shape, action_dim=env.action_space.n, num_heads=args.num_heads
    ).to(device)
else:
    agent = MlpAgent(obs_shape=obs.shape, action_dim=env.action_space.n).to(device)

optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

# 動態輸出檔案命名
suffix = f"_{args.exp_name}" if args.exp_name else ""
base_filename = f"{args.arch}_{args.style}{suffix}"

csv_filename = f"logs/training_log_{base_filename}.csv"
model_filename = f"custom_ppo_{base_filename}.pth"
plot_filename = f"logs/learning_curve_{base_filename}.png"

os.makedirs("logs", exist_ok=True)
csv_file = open(csv_filename, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["update", "step", "avg_reward", "value_loss", "policy_loss"])

# ==========================================
# 4. PPO 訓練主迴圈
# ==========================================
num_updates = TOTAL_TIMESTEPS // NUM_STEPS
global_step = 0
episode_rewards = []
recent_rewards = []

obs_buffer = torch.zeros((NUM_STEPS,) + obs.shape).to(device)
actions_buffer = torch.zeros((NUM_STEPS,)).to(device)
logprobs_buffer = torch.zeros((NUM_STEPS,)).to(device)
rewards_buffer = torch.zeros((NUM_STEPS,)).to(device)
dones_buffer = torch.zeros((NUM_STEPS,)).to(device)
values_buffer = torch.zeros((NUM_STEPS,)).to(device)

next_obs = torch.Tensor(obs).to(device)
next_done = torch.zeros(1).to(device)
current_ep_reward = 0

print(f"開始訓練 [{args.arch.upper()}] PPO ({args.style} 模式)...")
with tqdm(total=TOTAL_TIMESTEPS, desc=f"Training ({args.arch}|{args.style})") as pbar:
    for update in range(1, num_updates + 1):

        if not args.disable_lr_decay:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, NUM_STEPS):
            global_step += 1
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs.unsqueeze(0)
                )
                values_buffer[step] = value.flatten()

            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            next_obs_np, reward, terminated, truncated, info = env.step(
                action.cpu().numpy()[0]
            )
            current_ep_reward += reward
            done = terminated or truncated

            rewards_buffer[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor([done]).to(device)

            if done:
                episode_rewards.append(current_ep_reward)
                recent_rewards.append(current_ep_reward)
                if len(recent_rewards) > 20:
                    recent_rewards.pop(0)
                current_ep_reward = 0
                next_obs_np, info = env.reset()
                next_obs = torch.Tensor(next_obs_np).to(device)

            pbar.update(1)

        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).flatten()
            advantages = torch.zeros_like(rewards_buffer).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = (
                    rewards_buffer[t]
                    + GAMMA * nextvalues * nextnonterminal
                    - values_buffer[t]
                )
                advantages[t] = lastgaelam = (
                    delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buffer

        b_obs = obs_buffer.reshape((-1,) + obs.shape)
        b_actions = actions_buffer.reshape(-1)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(NUM_STEPS)
        for epoch in range(EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - CLIP_COEF, 1 + CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        avg_rew = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0
        csv_writer.writerow(
            [update, global_step, avg_rew, v_loss.item(), pg_loss.item()]
        )
        csv_file.flush()

        pbar.set_postfix({"Avg Reward": f"{avg_rew:.2f}"})

# ==========================================
# 5. 儲存模型與繪製學習曲線
# ==========================================
env.close()
csv_file.close()

torch.save(agent.state_dict(), model_filename)
print(f"\n模型已儲存為 {model_filename}")

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Episodic Reward", alpha=0.6, color="orange")
window_size = 50
if len(episode_rewards) >= window_size:
    moving_avg = np.convolve(
        episode_rewards, np.ones(window_size) / window_size, mode="valid"
    )
    plt.plot(
        range(window_size - 1, len(episode_rewards)),
        moving_avg,
        color="blue",
        label="Moving Average",
    )
plt.title(f"Training Curve ({args.arch.upper()} - {args.style.capitalize()}{suffix})")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.savefig(plot_filename)
print(f"學習曲線圖已儲存為 {plot_filename}")
