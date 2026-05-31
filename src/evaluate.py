import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import glob

# 引入模組化組件
from agents.networks import AttentionActorCritic, MlpActorCritic, MlpQNetwork
from envs.reward_shaper import get_env_config


def run_evaluation(model_path, algo, arch, num_heads, episodes, duration, device, env_id="highway-v0"):
    """執行單一模型的 100 局測試，並回傳指標字典"""
    # 1. 初始化環境
    config = get_env_config(env_id, "base", duration=duration)
    env = gym.make(env_id)
    env.unwrapped.configure(config)
    obs, info = env.reset()

    # 2. 實例化網路
    if algo == "ppo":
        if arch == "attention":
            network = AttentionActorCritic(obs.shape, env.action_space.n, num_heads).to(
                device
            )
        else:
            network = MlpActorCritic(obs.shape, env.action_space.n).to(device)
    elif algo == "dqn":
        network = MlpQNetwork(obs.shape, env.action_space.n).to(device)
    else:
        return None

    # 載入權重
    try:
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
    except Exception as e:
        print(f"❌ 載入模型失敗 ({model_path}): {e}")
        return None

    total_crashes = 0
    all_speeds = []
    total_lane_changes = 0
    all_survival_steps = []

    # 3. 開始測試
    for ep in tqdm(
        range(episodes), desc=f"評估中: {os.path.basename(model_path)}", leave=False
    ):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        ep_speeds = []
        ep_lane_changes = 0

        while not (done or truncated):
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                if algo == "ppo":
                    features = (
                        network.extract_features(obs_tensor)
                        if arch == "attention"
                        else network.feature_extractor(obs_tensor)
                    )
                    logits = network.actor(features)
                    action = torch.argmax(logits, dim=1).item()
                elif algo == "dqn":
                    q_values = network(obs_tensor)
                    action = torch.argmax(q_values, dim=1).item()

            obs, reward, done, truncated, info = env.step(action)
            steps += 1

            ep_speeds.append(info.get("speed", 0))
            if action in [0, 2]:
                ep_lane_changes += 1
            if info.get("crashed", False):
                total_crashes += 1
                break

        all_survival_steps.append(steps)
        if ep_speeds:
            all_speeds.append(np.mean(ep_speeds))
        total_lane_changes += ep_lane_changes

    env.close()

    # 4. 回傳字典
    return {
        "crash_rate": (total_crashes / episodes) * 100,
        "avg_survival": np.mean(all_survival_steps),
        "avg_speed": np.mean(all_speeds) if all_speeds else 0,
        "avg_lane_changes": total_lane_changes / episodes,
    }


def evaluate_model():
    parser = argparse.ArgumentParser(description="Evaluate RL Models")
    parser.add_argument(
        "--all", action="store_true", help="自動掃描 models/ 並評估所有 .pth 模型"
    )
    parser.add_argument(
        "--model_path", type=str, default="", help="單一評估: .pth 模型路徑"
    )
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument(
        "--arch", type=str, default="attention", choices=["attention", "mlp"]
    )
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=100, help="每組測試局數")
    parser.add_argument("--duration", type=int, default=80, help="環境最高存活步數")
    parser.add_argument(
        "--env", type=str, default="highway-v0",
        choices=["highway-v0", "merge-v0", "roundabout-v0", "intersection-v0"],
        help="highway-env environment ID",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_to_eval = []

    # 智慧型解析：尋找並拆解檔名
    if args.all:
        print(f"\n🔍 啟動批次評估模式 (自動掃描 models/ 目錄)...")
        pth_files = glob.glob(os.path.join("models", "*.pth"))

        for pth in pth_files:
            basename = os.path.basename(pth)
            parts = basename.replace(".pth", "").split("_")

            # 檔名格式: custom_model_{algo}_{arch}_{style}_...
            if len(parts) >= 5:
                algo = parts[2]
                arch = parts[3]
                style = parts[4]
                display_name = f"{arch.upper()}-{algo.upper()} ({style.capitalize()})"

                models_to_eval.append(
                    {
                        "path": pth,
                        "algo": algo,
                        "arch": arch,
                        "style": style,
                        "name": display_name,
                    }
                )
    elif args.model_path:
        models_to_eval.append(
            {
                "path": args.model_path,
                "algo": args.algo,
                "arch": args.arch,
                "style": "unknown",
                "name": os.path.basename(args.model_path),
            }
        )
    else:
        print("❌ 請指定 --model_path 或使用 --all 參數啟動批次處理！")
        return

    if not models_to_eval:
        print("⚠️ 找不到任何可以評估的模型！")
        return

    print(
        f"共找到 {len(models_to_eval)} 個模型即將進行評估 (每組 {args.episodes} 局)。\n"
    )

    # 執行迴圈測試
    results = []
    for m in models_to_eval:
        metrics = run_evaluation(
            m["path"],
            m["algo"],
            m["arch"],
            args.num_heads,
            args.episodes,
            args.duration,
            device,
            env_id=args.env,
        )
        if metrics:
            m.update(metrics)
            results.append(m)
            print(
                f"   ↳ 完成: 肇事率 {metrics['crash_rate']:.1f}% | 存活 {metrics['avg_survival']:.1f} 步 | 時速 {metrics['avg_speed']:.1f} | 換道 {metrics['avg_lane_changes']:.1f} 次\n"
            )

    # ===============================================
    # 印出可以直接貼入 Markdown/論文 的完美表格
    # ===============================================
    print(f"\n{'='*95}")
    print(
        f" 📊 最終客觀指標評估總表 (每組獨立測試 {args.episodes} 局, 環境上限 {args.duration} 步)"
    )
    print(f"{'='*95}")
    print(
        f"| {'模型配置 (架構-演算法-風格)':<35} | {'肇事率 %':<10} | {'平均存活步數':<15} | {'平均時速 (m/s)':<15} | {'平均換道次數':<12} |"
    )
    print(f"|{'-'*37}|{'-'*12}|{'-'*17}|{'-'*17}|{'-'*14}|")

    # 根據肇事率做排序，讓最安全的排在最上面 (可選)
    results = sorted(results, key=lambda x: x["crash_rate"])

    for r in results:
        name = r["name"]
        crash = f"{r['crash_rate']:.1f}%"
        surv = f"{r['avg_survival']:.1f}"
        spd = f"{r['avg_speed']:.2f}"
        lc = f"{r['avg_lane_changes']:.2f}"
        print(f"| {name:<35} | {crash:<10} | {surv:<15} | {spd:<15} | {lc:<12} |")
    print(f"{'='*95}\n")


if __name__ == "__main__":
    evaluate_model()
