import gymnasium as gym
import torch
import numpy as np
import time

# 載入我們的自定義環境與 Actor 網路
from custom_env import ObstacleAvoidanceWrapper
from ddpg_core import Actor


def main():
    # 測試階段：開啟 3D 渲染畫面 (render_mode="human")
    base_env = gym.make("FetchReach-v4", render_mode="human", max_episode_steps=50)
    env = ObstacleAvoidanceWrapper(base_env)

    # 取得維度資訊
    obs, _ = env.reset()
    state_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 建立 Actor 網路並載入訓練好的權重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(state_dim, action_dim, max_action).to(device)

    # 讀取剛剛存下來的模型檔案
    try:
        actor.load_state_dict(torch.load("models/actor_final.pth", map_location=device))
        print("✅ 成功載入模型權重 models/actor_final.pth")
    except FileNotFoundError:
        print("❌ 找不到模型檔案，請確認 train_robot.py 有正確執行並存檔。")
        return

    actor.eval()  # 設定為推論模式

    print("\n🎬 開始播放測試畫面...")
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        collision_count = 0

        print(f"\n--- 測試回合 {episode + 1} ---")

        for step in range(50):
            # 轉換狀態格式
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            # 讓 Actor 決定動作 (測試階段：不加入 OU Noise 雜訊)
            with torch.no_grad():
                action = actor(state_tensor).cpu().data.numpy().flatten()

            # 與環境互動
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if info.get("collision", False):
                collision_count += 1

            time.sleep(0.05)  # 放慢畫面速度方便肉眼觀察

            if terminated or truncated:
                print(
                    f"回合結束。總步數: {step+1} | 總獎勵: {episode_reward:.2f} | 碰撞次數: {collision_count}"
                )
                break

    env.close()


if __name__ == "__main__":
    main()
