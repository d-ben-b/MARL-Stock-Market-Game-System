import gymnasium as gym
import gymnasium_robotics  # 必須加入這行，註冊機器人環境
import time


def main():
    # 載入 Fetch 機械手臂到達目標的基礎環境，並開啟視窗渲染
    # max_episode_steps=50 代表每 50 步為一個回合
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchReach-v4", render_mode="human", max_episode_steps=50)

    # 初始化環境，取得初始狀態
    observation, info = env.reset()

    print("=== 環境空間資訊 ===")
    print(f"動作空間 (Action Space): {env.action_space}")
    print(f"觀察空間 (Observation Space keys): {observation.keys()}")
    print("==================\n")

    for episode in range(3):  # 測試跑 3 個回合
        observation, info = env.reset()
        print(f"開始第 {episode + 1} 回合")

        for step in range(50):
            # 產生一個符合物理限制的隨機動作 (4 維連續向量)
            action = env.action_space.sample()

            # 將動作輸入環境，取得回傳值
            observation, reward, terminated, truncated, info = env.step(action)

            # 放慢迴圈速度，讓肉眼能看清楚手臂的動作
            time.sleep(0.05)

            if terminated or truncated:
                print(f"回合結束，共執行了 {step + 1} 步")
                break

    env.close()


if __name__ == "__main__":
    main()
