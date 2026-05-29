import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(experiments, output_filename, title):
    """
    讀取多個 CSV 並將它們的 Learning Curve 畫在同一張圖上

    :param experiments: dict, 格式為 {"圖例名稱": "CSV檔案路徑"}
    :param output_filename: str, 輸出的圖片檔名
    :param title: str, 圖表標題
    """
    plt.figure(figsize=(10, 6))

    # 定義顏色清單，確保多條線顏色容易區分
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for (label, filepath), color in zip(experiments.items(), colors):
        if not os.path.exists(filepath):
            print(f"⚠️ 找不到檔案: {filepath}，跳過繪製 '{label}'")
            continue

        # 讀取 CSV 檔案
        df = pd.read_csv(filepath)

        # 提取步數 (step) 與 平均獎勵 (avg_reward)
        steps = df["step"]
        rewards = df["avg_reward"]

        # 繪製曲線
        plt.plot(steps, rewards, label=label, color=color, linewidth=2, alpha=0.8)

    # 圖表美化
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Average Episodic Reward (Window=20)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right", fontsize=11)

    # 儲存圖片
    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", output_filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ 圖表已成功儲存至: {out_path}")
    plt.close()


if __name__ == "__main__":
    # ==============================================================
    # 實驗一：優化演算法穩定性 (Ablation on Optimization)
    # ==============================================================
    exp1_optimization = {
        "Conservative (Base, High Variance)": "logs/training_log_attention_conservative.csv",
        "Conservative (No LR Decay)": "logs/training_log_attention_conservative_no_decay.csv",
        "Conservative (Stable, BS=256 + Decay)": "logs/training_log_attention_conservative_stable_v1.csv",
    }
    plot_learning_curves(
        experiments=exp1_optimization,
        output_filename="comparison_optimization.png",
        title="Ablation Study: Optimization Stability",
    )

    # ==============================================================
    # 實驗二：獎勵塑形的行為影響 (Impact of Reward Shaping)
    # ==============================================================
    exp2_driving_styles = {
        "Conservative Style (Safety First)": "logs/training_log_attention_conservative_stable_v1.csv",
        "Aggressive Style (Speed First)": "logs/training_log_attention_aggressive_fast_driver.csv",
        "Default Base Style": "logs/training_log_attention.csv",
    }
    plot_learning_curves(
        experiments=exp2_driving_styles,
        output_filename="comparison_driving_styles.png",
        title="Impact of Reward Shaping on Driving Styles",
    )

    # ==============================================================
    # 實驗三：網路架構消融 (Ablation on Architecture)
    # 目的：對比純 MLP 與 Attention 機制的學習表現
    # ==============================================================
    exp3_architecture = {
        "Attention (Stable)": "logs/training_log_attention_conservative_stable_v1.csv",
        "MLP (Baseline)": "logs/training_log_mlp_conservative_stable_v1.csv",
    }
    plot_learning_curves(
        experiments=exp3_architecture,
        output_filename="comparison_architecture.png",
        title="Ablation Study: Attention vs. MLP Architecture",
    )

    # ==============================================================
    # 實驗四：注意力頭數消融 (Ablation on Attention Heads)
    # 目的：對比 4 Heads 與 1 Head 的資訊捕捉能力
    # ==============================================================
    exp4_attention_heads = {
        "4 Heads (Default)": "logs/training_log_attention_conservative_stable_v1.csv",
        "1 Head": "logs/training_log_attention_conservative_1head_stable_v1.csv",
    }
    plot_learning_curves(
        experiments=exp4_attention_heads,
        output_filename="comparison_attention_heads.png",
        title="Ablation Study: Multi-Head vs. Single-Head Attention",
    )

    # ==============================================================
    # 實驗五：探索率消融 (Ablation on Entropy)
    # 目的：對比適度隨機探索與完全貪婪策略的差異
    # ==============================================================
    exp5_entropy = {
        "Entropy = 0.005 (Default)": "logs/training_log_attention_conservative_stable_v1.csv",
        "Entropy = 0.0 (Zero Exploration)": "logs/training_log_attention_conservative_zero_entropy.csv",
    }
    plot_learning_curves(
        experiments=exp5_entropy,
        output_filename="comparison_entropy.png",
        title="Ablation Study: Impact of Entropy Coefficient",
    )
