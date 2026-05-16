import numpy as np
import torch
import os

# 匯入你手刻的環境與演算法模組
# 請依照你的檔案結構調整 import 路徑
from environment.trade_environment import MultiAgentMarketEnv
from maddpg import DDPGAgent, ReplayBuffer


def flatten_state(state_dict, portfolio, depth=5):
    """將訂單簿與個人資產攤平成 1D 向量 (維度: 22)"""
    features = []

    for i in range(depth):
        if i < len(state_dict["bids"]):
            features.extend(
                [float(state_dict["bids"][i][0]), float(state_dict["bids"][i][1])]
            )
        else:
            features.extend([0.0, 0.0])

    for i in range(depth):
        if i < len(state_dict["asks"]):
            features.extend(
                [float(state_dict["asks"][i][0]), float(state_dict["asks"][i][1])]
            )
        else:
            features.extend([0.0, 0.0])

    features.append(float(portfolio["cash"]))
    features.append(float(portfolio["inventory"]))

    return np.array(features, dtype=np.float32)


def train_maddpg():
    # 1. 參數設定
    num_agents = 3
    num_episodes = 1000  # 總訓練回合數
    max_steps = 200  # 每回合最大步數 (代表一天的交易時間)
    batch_size = 64

    # 2. 初始化環境、Agents 與 Buffers
    env = MultiAgentMarketEnv(num_agents=num_agents, heterogeneous_reward=True)

    agents = [DDPGAgent(state_dim=22, action_dim=4) for _ in range(num_agents)]
    buffers = [ReplayBuffer(capacity=50000) for _ in range(num_agents)]

    # 紀錄訓練過程的 Reward
    history_rewards = {i: [] for i in range(num_agents)}

    print("=== 開始 MADDPG 訓練 ===")

    # 3. 進入訓練迴圈
    for episode in range(num_episodes):
        # 隨著訓練進行，逐漸降低探索噪音 (讓 Agent 從亂試變成相信自己的策略)
        noise_scale = max(0.01, 1.0 - (episode / (num_episodes * 0.8)))

        raw_states = env.reset()
        # 取得初始資產狀態 (用於 flatten)
        infos = {i: env.portfolios[i] for i in range(num_agents)}

        # 將 Dict 狀態攤平為 Numpy Array
        states = {i: flatten_state(raw_states[i], infos[i]) for i in range(num_agents)}

        episode_rewards = np.zeros(num_agents)

        for step in range(max_steps):
            # (A) 收集每個 Agent 的動作
            actions = {}
            for i in range(num_agents):
                actions[i] = agents[i].get_action(states[i], noise_scale=noise_scale)

            # (B) 與環境互動
            next_raw_states, step_rewards, dones, next_infos = env.step(actions)

            # (C) 處理 Next State 並存入 Buffer
            next_states = {}
            for i in range(num_agents):
                next_states[i] = flatten_state(next_raw_states[i], next_infos[i])

                # 儲存經驗 (State, Action, Reward, Next_State, Done)
                buffers[i].push(
                    states[i], actions[i], step_rewards[i], next_states[i], dones[i]
                )

                episode_rewards[i] += step_rewards[i]

            # (D) 更新神經網路
            for i in range(num_agents):
                agents[i].update(buffers[i], batch_size)

            states = next_states

        # (E) 回合結算與日誌輸出
        for i in range(num_agents):
            history_rewards[i].append(episode_rewards[i])

        if (episode + 1) % 10 == 0:
            print(
                f"Episode: {episode + 1:4d} | Noise: {noise_scale:.2f} | "
                f"Rewards -> A0: {episode_rewards[0]:.2f}, A1: {episode_rewards[1]:.2f}, A2: {episode_rewards[2]:.2f}"
            )

    # 4. 儲存訓練好的模型權重
    os.makedirs("saved_models", exist_ok=True)
    for i in range(num_agents):
        torch.save(agents[i].actor.state_dict(), f"saved_models/actor_agent_{i}.pth")

    print("=== 訓練結束，模型已儲存 ===")
    plot_learning_curves(history_rewards, window=50)


import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(history_rewards, window=50):
    """
    繪製多代理人的 Reward 學習曲線
    包含原始數據 (淺色背景) 與移動平均線 (深色主線)
    """
    plt.figure(figsize=(12, 6))

    # 定義每個 Agent 的專屬顏色
    colors = {0: "blue", 1: "orange", 2: "green"}
    labels = {
        0: "Agent 0 (Market Maker)",
        1: "Agent 1 (Conservative)",
        2: "Agent 2 (Aggressive)",
    }

    for agent_id, rewards in history_rewards.items():
        episodes = np.arange(len(rewards))

        # 1. 繪製原始 Reward (低透明度，顯示震盪範圍)
        plt.plot(episodes, rewards, color=colors[agent_id], alpha=0.2, linewidth=1)

        # 2. 計算並繪製移動平均線 (MA)
        if len(rewards) >= window:
            # 使用卷積計算 Moving Average
            weights = np.ones(window) / window
            ma_rewards = np.convolve(rewards, weights, mode="valid")
            ma_episodes = np.arange(window - 1, len(rewards))

            plt.plot(
                ma_episodes,
                ma_rewards,
                color=colors[agent_id],
                alpha=1.0,
                linewidth=2,
                label=f"{labels.get(agent_id, f'Agent {agent_id}')} (MA-{window})",
            )
        else:
            # 如果訓練回合數少於 window，則直接標示原線
            plt.plot(
                episodes,
                rewards,
                color=colors[agent_id],
                alpha=1.0,
                linewidth=2,
                label=labels.get(agent_id, f"Agent {agent_id}"),
            )

    # 圖表美化與標籤設定
    plt.title(
        "MADDPG Agents Learning Curve (Reward over Episodes)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Cumulative Reward (PnL - Penalties)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axhline(
        y=0, color="red", linestyle="-", linewidth=1, alpha=0.5
    )  # 標示損益兩平線
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    # 儲存圖片並顯示
    plt.savefig("saved_models/maddpg_learning_curve.png", dpi=300)
    print("=== 學習曲線圖表已儲存至 saved_models/maddpg_learning_curve.png ===")
    plt.show()


if __name__ == "__main__":
    train_maddpg()
