import gymnasium as gym
import highway_env
import time

# 1. 建立環境，開啟視覺化視窗
env = gym.make("highway-v0", render_mode="human")

# 2. 進行環境設定
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    # 【關鍵設定】：開啟環境內建的鍵盤攔截功能
    "manual_control": True,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 40,
}
env.unwrapped.configure(config)

# 初始化環境
obs, info = env.reset()

print("==============================")
print(" 🚗 手動控制模式已啟動 🚗")
print("請點擊彈出的 PyGame 遊戲視窗，並使用【方向鍵】控制：")
print(" ⬆️ (Up)    : 加速 (Action 3: FASTER)")
print(" ⬇️ (Down)  : 減速 (Action 4: SLOWER)")
print(" ⬅️ (Left)  : 左切車道 (Action 0: LANE_LEFT)")
print(" ➡️ (Right) : 右切車道 (Action 2: LANE_RIGHT)")
print("==============================")

done = False
truncated = False
score = 0

# 3. 遊戲主迴圈
while not (done or truncated):
    # 因為我們開啟了 manual_control，這裡傳遞的 action=1 (IDLE) 會作為預設值
    # 當你按下方向鍵時，環境底層會自動覆蓋掉這個預設值
    action = 1

    # 執行動作，獲取回饋
    obs, reward, done, truncated, info = env.step(action)
    score += reward

    # 加入微小的延遲，讓畫面幀率對人類視覺來說比較舒適 (可依你的電腦效能微調)
    time.sleep(0.05)

print(f"\n回合結束！")
print(f"🏆 你的總獲得獎勵 (Score): {score:.2f}")
print(f"💥 是否發生碰撞 (Terminated): {done}")

env.close()
