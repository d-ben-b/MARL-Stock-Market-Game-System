import os
import csv
import argparse
import gymnasium as gym
import highway_env
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# 引入我們寫好的模組化組件
from agents.networks import AttentionActorCritic, MlpActorCritic, MlpQNetwork
from agents.ppo import PPO
from agents.dqn import DQNAgent
from envs.reward_shaper import get_highway_config

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default=".")
# 新增演算法選擇與時長設定
parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
parser.add_argument(
    "--duration", type=int, default=80, help="回合時長(步數)，預設拉長為80步"
)
parser.add_argument(
    "--arch", type=str, default="attention", choices=["attention", "mlp"]
)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument(
    "--style", type=str, default="base", choices=["base", "conservative", "aggressive"]
)
parser.add_argument("--minibatch_size", type=int, default=256)
parser.add_argument("--ent_coef", type=float, default=0.005)
parser.add_argument("--disable_lr_decay", action="store_true")
parser.add_argument("--exp_name", type=str, default="")
args = parser.parse_args()

print(f"=== 🧪 訓練啟動 ({args.algo.upper()}) ===")
print(f" - 架構: {args.arch} | 風格: {args.style} | 時長: {args.duration}步")
print(f"==============================")

TOTAL_TIMESTEPS = 100000
NUM_STEPS = 1024  # PPO rollout length
LEARNING_RATE = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 目錄與檔案命名
logs_dir = os.path.join(args.root_dir, "logs")
models_dir = os.path.join(args.root_dir, "models")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

suffix = (
    f"_dur{args.duration}_{args.exp_name}" if args.exp_name else f"_dur{args.duration}"
)
base_filename = f"{args.algo}_{args.arch}_{args.style}{suffix}"

csv_filename = os.path.join(logs_dir, f"training_log_{base_filename}.csv")
model_filename = os.path.join(models_dir, f"custom_model_{base_filename}.pth")
plot_filename = os.path.join(logs_dir, f"learning_curve_{base_filename}.png")

csv_file = open(csv_filename, mode="w", newline="")
csv_writer = csv.writer(csv_file)

# 環境初始化
config = get_highway_config(args.style, duration=args.duration)
env = gym.make("highway-v0")
env.unwrapped.configure(config)
obs, info = env.reset()

# ==========================================
# 訓練迴圈分支 (PPO vs DQN)
# ==========================================
episode_rewards = []
recent_rewards = []
current_ep_reward = 0
global_step = 0

if args.algo == "ppo":
    csv_writer.writerow(["update", "step", "avg_reward", "value_loss", "policy_loss"])

    if args.arch == "attention":
        network = AttentionActorCritic(
            obs.shape, env.action_space.n, args.num_heads
        ).to(device)
    else:
        network = MlpActorCritic(obs.shape, env.action_space.n).to(device)

    agent = PPO(
        network=network,
        learning_rate=LEARNING_RATE,
        minibatch_size=args.minibatch_size,
        ent_coef=args.ent_coef,
    )
    num_updates = TOTAL_TIMESTEPS // NUM_STEPS

    obs_buffer = torch.zeros((NUM_STEPS,) + obs.shape).to(device)
    actions_buffer = torch.zeros((NUM_STEPS,)).to(device)
    logprobs_buffer = torch.zeros((NUM_STEPS,)).to(device)
    rewards_buffer = torch.zeros((NUM_STEPS,)).to(device)
    dones_buffer = torch.zeros((NUM_STEPS,)).to(device)
    values_buffer = torch.zeros((NUM_STEPS,)).to(device)

    next_obs = torch.Tensor(obs).to(device)
    next_done = torch.zeros(1).to(device)

    with tqdm(total=TOTAL_TIMESTEPS, desc=f"PPO Training") as pbar:
        for update in range(1, num_updates + 1):
            if not args.disable_lr_decay:
                agent.update_learning_rate(update, num_updates, LEARNING_RATE)

            for step in range(0, NUM_STEPS):
                global_step += 1
                obs_buffer[step] = next_obs
                dones_buffer[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = network.get_action_and_value(
                        next_obs.unsqueeze(0)
                    )
                    values_buffer[step] = value.flatten()

                actions_buffer[step] = action
                logprobs_buffer[step] = logprob

                next_obs_np, reward, terminated, truncated, _ = env.step(
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
                    next_obs_np, _ = env.reset()
                    next_obs = torch.Tensor(next_obs_np).to(device)

                pbar.update(1)

            with torch.no_grad():
                next_value = network.get_value(next_obs.unsqueeze(0)).flatten()
                advantages, returns = agent.compute_gae(
                    rewards_buffer, values_buffer, dones_buffer, next_value, next_done
                )

            v_loss, pg_loss = agent.train_step(
                obs_buffer.reshape((-1,) + obs.shape),
                actions_buffer.reshape(-1),
                logprobs_buffer.reshape(-1),
                advantages.reshape(-1),
                returns.reshape(-1),
            )

            avg_rew = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0
            csv_writer.writerow([update, global_step, avg_rew, v_loss, pg_loss])
            csv_file.flush()
            pbar.set_postfix({"Avg Reward": f"{avg_rew:.2f}"})

elif args.algo == "dqn":
    csv_writer.writerow(["step", "avg_reward", "loss", "epsilon"])

    # DQN 僅實作 MLP 版本作為簡單 Baseline
    network = MlpQNetwork(obs.shape, env.action_space.n).to(device)
    agent = DQNAgent(network, learning_rate=1e-4)

    with tqdm(total=TOTAL_TIMESTEPS, desc=f"DQN Training") as pbar:
        while global_step < TOTAL_TIMESTEPS:
            global_step += 1
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            action = agent.select_action(obs_tensor, env.action_space)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            current_ep_reward += reward
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.train_step()

            obs = next_obs

            if done:
                episode_rewards.append(current_ep_reward)
                recent_rewards.append(current_ep_reward)
                if len(recent_rewards) > 20:
                    recent_rewards.pop(0)

                avg_rew = np.mean(recent_rewards)
                csv_writer.writerow([global_step, avg_rew, loss, agent.epsilon])
                csv_file.flush()
                pbar.set_postfix(
                    {"Avg Reward": f"{avg_rew:.2f}", "Eps": f"{agent.epsilon:.2f}"}
                )

                current_ep_reward = 0
                obs, _ = env.reset()

            pbar.update(1)

# ==========================================
# 5. 結束與儲存模型
# ==========================================
env.close()
csv_file.close()

torch.save(network.state_dict(), model_filename)
print(f"\n🎉 訓練完成！模型已儲存為 {model_filename}")

# 繪製與儲存學習曲線
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
plt.title(f"Training Curve ({args.algo.upper()} - {args.arch} - {args.style})")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.savefig(plot_filename)
print(f"📈 學習曲線圖已儲存為 {plot_filename}")
