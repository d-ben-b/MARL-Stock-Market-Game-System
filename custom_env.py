import gymnasium as gym
import gymnasium_robotics
import numpy as np


class ObstacleAvoidanceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # 1. 設定虛擬障礙物 (假設放在桌子正上方中央)
        self.obstacle_pos = np.array([1.35, 0.60, 0.50])
        self.safe_radius = 0.1  # 安全半徑 10 公分，低於此距離算撞到

        # 2. 重新定義觀察空間 (Observation Space)
        # 原本 observation(10維) + desired_goal(3維) + obstacle(3維) = 16維
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)

        gripper_pos = obs_dict["achieved_goal"]
        target_pos = obs_dict["desired_goal"]

        dist_to_obstacle = np.linalg.norm(gripper_pos - self.obstacle_pos)
        dist_to_target = np.linalg.norm(gripper_pos - target_pos)

        # --- Reward Shaping (獎勵塑形) ---
        # 1. 給予靠近目標的獎勵 (距離越小，負分越少)
        shaped_reward = -dist_to_target

        # 2. 碰撞懲罰 (不要提早結束回合，讓它能在失敗中繼續嘗試走到目標)
        if dist_to_obstacle < self.safe_radius:
            shaped_reward -= 5.0  # 懲罰調小一點，避免遮蓋掉尋找目標的獎勵
            info["collision"] = True
        else:
            info["collision"] = False

        return self._flatten_obs(obs_dict), shaped_reward, terminated, truncated, info

    def _flatten_obs(self, obs_dict):
        # 將字典打平合併成一個一維陣列，讓後續的 DDPG 網路可以直接吃
        flattened = np.concatenate(
            [obs_dict["observation"], obs_dict["desired_goal"], self.obstacle_pos]
        )
        return flattened.astype(np.float32)


# 測試我們寫好的客製化環境
if __name__ == "__main__":
    # 建立基礎環境
    base_env = gym.make("FetchReach-v4", render_mode="human", max_episode_steps=50)

    # 套上我們的避障外衣
    env = ObstacleAvoidanceWrapper(base_env)

    obs, info = env.reset()
    print(f"✅ 成功打平狀態！新的狀態維度大小為: {obs.shape}")

    for episode in range(3):
        env.reset()
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    env.close()
