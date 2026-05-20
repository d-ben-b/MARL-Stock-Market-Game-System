import numpy as np
import torch
import os
import csv
from datetime import datetime
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt

# 匯入手刻的環境與演算法模組
from environment.trade_environment import (
    MultiAgentMarketEnv,
    calculate_financial_metrics,
)
from maddpg import DDPGAgent, ReplayBuffer


def flatten_state(state_dict, portfolio, sentiment_score, depth=5):
    """將訂單簿與個人資產攤平成 1D 向量 (維度: 23)"""
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
    features.append(float(sentiment_score))

    return np.array(features, dtype=np.float32)


def train_maddpg():
    # 1. 參數設定
    num_agents = 3
    num_episodes = 1000
    max_steps = 200
    batch_size = 64

    # 2. 初始化時間戳記與日誌目錄結構
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H%M%S")

    # 建立 logs/yyyy-mm-dd/ 目錄
    log_dir = os.path.join("logs", current_date)
    os.makedirs(log_dir, exist_ok=True)

    # 定義 CSV 檔案路徑
    log_file_path = os.path.join(log_dir, f"training_log_{current_time}.csv")
    summary_file_path = os.path.join(log_dir, f"financial_summary_{current_time}.csv")

    # 3. 初始化環境、Agents 與 Buffers
    env = MultiAgentMarketEnv(num_agents=num_agents, heterogeneous_reward=True)
    agents = [DDPGAgent(state_dim=23, action_dim=4) for _ in range(num_agents)]
    buffers = [ReplayBuffer(capacity=50000) for _ in range(num_agents)]

    history_rewards = {i: [] for i in range(num_agents)}

    print("\n" + "=" * 50)
    print("🚀 開始 MADDPG 異質代理人市場訓練")
    print(f"詳細日誌將儲存至: {log_file_path}")
    print("=" * 50 + "\n")

    pbar = tqdm(range(num_episodes), desc="Training Progress", ncols=100, unit="ep")

    # 4. 開啟 CSV 檔案準備動態寫入每回合數據
    with open(log_file_path, mode="w", newline="", encoding="utf-8") as log_file:
        csv_writer = csv.writer(log_file)
        # 寫入 CSV 表頭
        csv_writer.writerow(
            [
                "Episode",
                "Noise_Scale",
                "Reward_MarketMaker",
                "Reward_Conservative",
                "Reward_Aggressive",
            ]
        )

        for episode in pbar:
            noise_scale = max(0.01, 1.0 - (episode / (num_episodes * 0.8)))

            raw_states = env.reset()
            infos = {i: env.portfolios[i] for i in range(num_agents)}
            states = {
                i: flatten_state(raw_states[i], infos[i], env.current_sentiment)
                for i in range(num_agents)
            }

            episode_rewards = np.zeros(num_agents)

            for step in range(max_steps):
                actions = {}
                for i in range(num_agents):
                    actions[i] = agents[i].get_action(
                        states[i], noise_scale=noise_scale
                    )

                next_raw_states, step_rewards, dones, next_infos = env.step(actions)

                next_states = {}
                for i in range(num_agents):
                    next_states[i] = flatten_state(
                        next_raw_states[i], next_infos[i], env.current_sentiment
                    )
                    buffers[i].push(
                        states[i], actions[i], step_rewards[i], next_states[i], dones[i]
                    )
                    episode_rewards[i] += step_rewards[i]

                for i in range(num_agents):
                    agents[i].update(buffers[i], batch_size)

                states = next_states

            # 紀錄內部陣列
            for i in range(num_agents):
                history_rewards[i].append(episode_rewards[i])

            # 新增：將本回合數據即時寫入 CSV 檔案
            csv_writer.writerow(
                [
                    episode + 1,
                    f"{noise_scale:.4f}",
                    f"{episode_rewards[0]:.2f}",
                    f"{episode_rewards[1]:.2f}",
                    f"{episode_rewards[2]:.2f}",
                ]
            )
            log_file.flush()  # 強制將緩衝區數據寫入磁碟，防止程式異常中斷遺失數據

            pbar.set_postfix(
                {
                    "Noise": f"{noise_scale:.2f}",
                    "R_MM": f"{episode_rewards[0]:.0f}",
                    "R_Con": f"{episode_rewards[1]:.0f}",
                    "R_Agg": f"{episode_rewards[2]:.0f}",
                }
            )

    # 5. 儲存模型權重
    os.makedirs("saved_models", exist_ok=True)
    for i in range(num_agents):
        torch.save(agents[i].actor.state_dict(), f"saved_models/actor_agent_{i}.pth")

    print("\n" + "=" * 50)
    print("✅ 訓練結束，模型權重已儲存")
    print("=" * 50 + "\n")

    # 6. 計算並輸出最終金融指標報告，同時導出總結 CSV
    print("📊 最終金融指標評估報告")
    metrics_table = []
    headers = ["Agent", "Role", "Final Net Worth", "Sharpe Ratio", "Max Drawdown (%)"]
    roles = {0: "Market Maker", 1: "Conservative", 2: "Aggressive"}

    # 開啟總結報告的 CSV 檔案
    with open(
        summary_file_path, mode="w", newline="", encoding="utf-8"
    ) as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(headers)  # 寫入表頭

        for i in range(num_agents):
            sharpe, mdd = calculate_financial_metrics(env.net_worth_history[i])
            final_wealth = env.net_worth_history[i][-1]

            # 準備終端機顯示的格式
            metrics_table.append(
                [
                    f"Agent {i}",
                    roles.get(i, "Unknown"),
                    f"${final_wealth:,.2f}",
                    f"{sharpe:.4f}",
                    f"{mdd*100:.2f}%",
                ]
            )

            # 寫入總結 CSV (純數字方便後續分析)
            summary_writer.writerow(
                [
                    f"Agent {i}",
                    roles.get(i, "Unknown"),
                    f"{final_wealth:.2f}",
                    f"{sharpe:.4f}",
                    f"{mdd*100:.2f}%",
                ]
            )

    print(
        tabulate(
            metrics_table, headers=headers, tablefmt="fancy_grid", stralign="center"
        )
    )
    print(f"ℹ️ 評估報告已導出至: {summary_file_path}\n")

    # 7. 繪製學習曲線
    plot_learning_curves(history_rewards, window=50)


def plot_learning_curves(history_rewards, window=50):
    """繪製多代理人的 Reward 學習曲線"""
    plt.figure(figsize=(12, 6))
    colors = {0: "blue", 1: "orange", 2: "green"}
    labels = {
        0: "Agent 0 (Market Maker)",
        1: "Agent 1 (Conservative)",
        2: "Agent 2 (Aggressive)",
    }

    for agent_id, rewards in history_rewards.items():
        episodes = np.arange(len(rewards))
        plt.plot(episodes, rewards, color=colors[agent_id], alpha=0.2, linewidth=1)

        if len(rewards) >= window:
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
            plt.plot(
                episodes,
                rewards,
                color=colors[agent_id],
                alpha=1.0,
                linewidth=2,
                label=labels.get(agent_id, f"Agent {agent_id}"),
            )

    plt.title(
        "MADDPG Agents Learning Curve (Reward over Episodes)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Cumulative Reward (PnL - Penalties)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axhline(y=0, color="red", linestyle="-", linewidth=1, alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig("saved_models/maddpg_learning_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    train_maddpg()
