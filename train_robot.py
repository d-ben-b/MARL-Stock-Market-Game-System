import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import os
from tqdm import tqdm

from custom_env import ObstacleAvoidanceWrapper
from ddpg_core import Actor, Critic, OUNoise, ReplayBuffer

# --- 參數設定 ---
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.002
ACTOR_LR = 2e-4
CRITIC_LR = 1e-3
MAX_EPISODES = 3000  # 可以先維持 1000 看收斂速度
MAX_STEPS = 50
MEMORY_SIZE = 1e5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 建立環境 ---
base_env = gym.make("FetchReach-v4", render_mode=None, max_episode_steps=MAX_STEPS)
env = ObstacleAvoidanceWrapper(base_env)

obs, _ = env.reset()
state_dim = obs.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# --- 2. 初始化網路與優化器 ---
actor = Actor(state_dim, action_dim, max_action).to(device)
actor_target = Actor(state_dim, action_dim, max_action).to(device)
actor_target.load_state_dict(actor.state_dict())
actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)

critic = Critic(state_dim, action_dim).to(device)
critic_target = Critic(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

noise = OUNoise(action_dim)
replay_buffer = ReplayBuffer(max_size=MEMORY_SIZE)


def soft_update(target_net, source_net, tau):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def train_step():
    if len(replay_buffer) < BATCH_SIZE:
        return

    state, action, reward, next_state, done = replay_buffer.sample(BATCH_SIZE)

    state = torch.FloatTensor(state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    done = torch.FloatTensor(done).unsqueeze(1).to(device)

    with torch.no_grad():
        next_action = actor_target(next_state)
        target_Q = critic_target(next_state, next_action)
        target_Q = reward + (1 - done) * GAMMA * target_Q

    current_Q = critic(state, action)
    critic_loss = torch.nn.MSELoss()(current_Q, target_Q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = -critic(state, actor(state)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    soft_update(critic_target, critic, TAU)
    soft_update(actor_target, actor, TAU)


# --- 5. 主訓練迴圈 (加入 HER) ---
history_rewards = []
current_phase = 0

pbar = tqdm(range(MAX_EPISODES), desc="Phase 0 訓練中", unit="ep")

for episode in pbar:
    obs, _ = env.reset()
    if episode > MAX_EPISODES // 2:
        noise.sigma = 0.05
    else:
        noise.sigma = 0.2

    noise.reset()
    noise.reset()
    episode_reward = 0
    collision_count = 0

    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

        action = actor(state_tensor).cpu().data.numpy().flatten()
        action = np.clip(action + noise.sample(), -max_action, max_action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info.get("collision", False):
            collision_count += 1

        # 1. 存入真實的經驗 (目標是紅球)
        replay_buffer.add(obs, action, reward, next_obs, done)

        # -----------------------------------------------------------
        # ★ Hindsight Experience Replay (HER) 實作區塊 ★
        # -----------------------------------------------------------
        # 取得當前手臂夾爪的三維座標 (在打平的陣列中是前 3 個元素 0:3)
        achieved_goal = next_obs[0:3]

        # 複製狀態，準備竄改目標 (打平陣列中目標座標的索引是 10:13)
        her_obs = np.copy(obs)
        her_next_obs = np.copy(next_obs)
        her_obs[10:13] = achieved_goal
        her_next_obs[10:13] = achieved_goal

        # 重新計算這筆虛擬經驗的分數 (因為目標就是夾爪現在的位置，距離為 0)
        her_reward = 0.0
        # 依然要保留碰撞懲罰，讓 AI 在虛擬經驗中也記得避開桌子！
        if info.get("collision", False):
            her_reward -= 5.0

        # 2. 存入捏造的完美經驗
        replay_buffer.add(her_obs, action, her_reward, her_next_obs, done)
        # -----------------------------------------------------------

        train_step()

        obs = next_obs
        episode_reward += reward

        if done:
            break

    history_rewards.append(episode_reward)
    avg_reward = np.mean(history_rewards[-50:])

    pbar.set_postfix(
        {
            "Reward": f"{episode_reward:.1f}",
            "Avg(50)": f"{avg_reward:.1f}",
            "Crash": "Yes" if collision_count > 0 else "No",
        }
    )

    # 目標：平均獎勵突破 -5 的瓶頸！
    if avg_reward > -5 and current_phase == 0:
        pbar.write(
            f"\n🌟 回合 {episode+1}: 成功率達標！進入 Phase 1：啟動動態障礙物！\n"
        )
        current_phase = 1
        pbar.set_description("Phase 1 訓練中")

print("✅ 訓練完成！")
os.makedirs("models", exist_ok=True)
torch.save(actor.state_dict(), "models/actor_final.pth")
