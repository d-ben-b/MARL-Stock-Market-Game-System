import os
import csv
import time
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
# 1. 超參數設定 (Hyperparameters)
# ==========================================
TOTAL_TIMESTEPS = 100000
NUM_STEPS = 1024  # 每次收集多少步才更新一次模型 (Rollout length)
MINIBATCH_SIZE = 64
EPOCHS = 10  # 每次資料重複訓練的次數
LEARNING_RATE = 5e-4
GAMMA = 0.99  # 折扣因子
GAE_LAMBDA = 0.95  # GAE 參數
CLIP_COEF = 0.2  # PPO 裁剪範圍限制
ENT_COEF = 0.01  # 熵獎勵權重 (鼓勵探索)
VF_COEF = 0.5  # 價值函數損失權重

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 2. 定義神經網路 (Actor-Critic)
# ==========================================
class Agent(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        # highway-env 回傳形狀為 (5, 5)，壓平後為 25 維
        input_dim = np.prod(obs_shape)

        # 評論家網路 (Critic): 評估當前狀態的價值
        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 演員網路 (Actor): 決定要採取的動作機率
        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


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
    "collision_reward": -1.0,
    "reward_speed_range": [20, 30],
}
env = gym.make("highway-v0")
env.unwrapped.configure(config)
obs, info = env.reset()

agent = Agent(obs_shape=obs.shape, action_dim=env.action_space.n).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

os.makedirs("logs", exist_ok=True)
csv_file = open("logs/training_log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["update", "step", "avg_reward", "value_loss", "policy_loss"])

# ==========================================
# 4. PPO 訓練主迴圈
# ==========================================
num_updates = TOTAL_TIMESTEPS // NUM_STEPS
global_step = 0
episode_rewards = []
recent_rewards = []  # 用於計算移動平均

# 儲存 Rollout 資料的容器
obs_buffer = torch.zeros((NUM_STEPS,) + obs.shape).to(device)
actions_buffer = torch.zeros((NUM_STEPS,)).to(device)
logprobs_buffer = torch.zeros((NUM_STEPS,)).to(device)
rewards_buffer = torch.zeros((NUM_STEPS,)).to(device)
dones_buffer = torch.zeros((NUM_STEPS,)).to(device)
values_buffer = torch.zeros((NUM_STEPS,)).to(device)

next_obs = torch.Tensor(obs).to(device)
next_done = torch.zeros(1).to(device)
current_ep_reward = 0

print("開始訓練手刻 PPO...")
with tqdm(total=TOTAL_TIMESTEPS, desc="Training Progress") as pbar:
    for update in range(1, num_updates + 1):
        # 階段 A: 收集資料 (Rollout)
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

            # 執行動作
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

        # 階段 B: 計算優勢函數 (GAE)
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

        # 階段 C: 更新神經網路 (PPO Clip)
        b_obs = obs_buffer.reshape((-1,) + obs.shape)
        b_actions = actions_buffer.reshape(-1)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # 批次更新
        b_inds = np.arange(NUM_STEPS)
        clipfracs = []
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

                # Advantage 正規化
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - CLIP_COEF, 1 + CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy Bonus
                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # 紀錄日誌
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

torch.save(agent.state_dict(), "custom_ppo_baseline.pth")
print("\n模型已儲存為 custom_ppo_baseline.pth")

# 繪圖
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label="Episodic Reward", alpha=0.6)
# 計算移動平均線讓圖表平滑
window_size = 50
if len(episode_rewards) >= window_size:
    moving_avg = np.convolve(
        episode_rewards, np.ones(window_size) / window_size, mode="valid"
    )
    plt.plot(
        range(window_size - 1, len(episode_rewards)),
        moving_avg,
        color="red",
        label="Moving Average",
    )
plt.title("Training Learning Curve (Custom PPO)")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.savefig("logs/learning_curve.png")
print("學習曲線圖已儲存為 logs/learning_curve.png")
