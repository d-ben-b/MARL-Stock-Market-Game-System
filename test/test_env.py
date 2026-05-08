import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# 確保上方有 import 你的 MatchingEngine, Order, MultiAgentMarketEnv 類別
from environment.trade_environment import MatchingEngine, Order, MultiAgentMarketEnv


def run_environment_test():
    num_agents = 3
    initial_cash = 100000

    # 初始化環境
    print(
        f"=== 初始化市場環境 (Agents: {num_agents}, Initial Cash: {initial_cash}) ==="
    )
    env = MultiAgentMarketEnv(num_agents=num_agents, initial_cash=initial_cash)

    # 測試 reset 函數
    states = env.reset()
    print("初始狀態 (State):")
    for agent_id, state in states.items():
        print(f"  Agent {agent_id}: {state}")

    num_steps = 1
    print(f"\n=== 開始執行隨機交易測試 (共 {num_steps} 步) ===")

    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")

        # 產生隨機動作
        # 動作維度: [策略訊號, 方向訊號, 價格訊號, 數量訊號], 範圍 [-1, 1]
        actions = {}
        for i in range(num_agents):
            actions[i] = np.random.uniform(-1, 1, size=(4,))
            print(f"Agent {i} 的動作: {actions[i]}")

        # 執行 step
        next_states, rewards, dones, infos = env.step(actions)

        # 印出撮合結果與獎勵
        print("【Rewards】:")
        for i in range(num_agents):
            print(f"  Agent {i}: {rewards[i]:.2f}")

        print("【Portfolios】:")
        for i in range(num_agents):
            cash = infos[i]["cash"]
            inv = infos[i]["inventory"]
            print(f"  Agent {i}: 現金 {cash:.2f}, 庫存 {inv}")

        print("【LOB 快照 (Agent 0 看到的狀態)】:")
        print(f"  Bids: {next_states[0]['bids']}")
        print(f"  Asks: {next_states[0]['asks']}")


if __name__ == "__main__":
    run_environment_test()
