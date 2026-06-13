# RL_final — When Episodic Return Lies: A Behavioural Ablation of PPO vs DQN on highway-env

在 [highway-env](https://github.com/Farama-Foundation/HighwayEnv) 的離散動作自駕場景下，從零重現並消融 **PPO** 與 **Double DQN**，並以參數對齊（parameter-matched）的方式比較 **Self-Attention vs MLP** 特徵抽取器。

---

## 🔑 核心發現 (Key Findings)

> **在安全攸關的強化學習中，「平均回合獎勵 (episodic return)」是一個會騙人的指標。**

- **回合獎勵幾乎打平，行為卻天差地遠。** 在 `highway-v0`（aggressive 風格，n=5 seeds）下，PPO+Attention（235.7）、PPO+MLP（223.0）、DQN+MLP（234.1）三者最終獎勵差距 ≤ 5.7%；但在共同 base 測試配置下，**碰撞率分別是 60.7% / 80.0% / 11.3%**——相差一個數量級。
- **DQN 反而最穩、最早學會。** Double DQN 跨 seed 變異最低（σ=7.7，PPO 為 16.7 / 19.6），且最快到達 avg≥150（中位數 ≈7,721 步，比 PPO+Attention 早 ~25%）。歸因於 experience replay 平滑了早期高變異梯度。
- **結論泛化到第二個環境。** 在 `merge-v0`（n=3）三者訓練獎勵更緊密（58–59，σ≤0.3），但 **DQN 碰撞率 0%、PPO 兩個變體皆 100%（所有 seed 一致）**——印證「return parity 掩蓋行為發散」並非單一 benchmark 的偶然。
- **獎勵設計才是駕駛風格的主控桿**，而非演算法或架構選擇；但即使同一獎勵，不同 seed 仍可能收斂到相反行為（reward shaping 約束方向，不保證收斂到目標 basin）。

🎬 **成果影片**：[`demo_dqn_vs_ppo_merge.mp4`](demo_dqn_vs_ppo_merge.mp4) — merge-v0 上同場景並排：DQN 安全合流 vs PPO 全速衝撞（100% 撞車）。

📄 **論文**：[`paper.tex`](paper.tex) / `paper.pdf`（IEEE conference 格式，7 頁）。

---

## ⚠️ 公平性與範圍

- **架構消融是參數對齊的**：Attention（66.8k）vs MLP（72.3k），差距 7.5%，任何差異不能歸因於容量。此公平性僅在 **PPO 內部**成立。
- **PPO vs DQN 是「演算法家族」比較**，非容量受控：DQN（20.5k）刻意較小，作為輕量 off-policy baseline。
- **Self-Attention 是 highway-env 的標準做法**（Leurent & Mercat 2019），本專案採用它作為消融的一臂，重現而非宣稱方法創新。本專案的貢獻在於**實驗發現本身**：量化 return 與行為的脫節，並在兩個環境上驗證。

---

## 環境設定

| 項目 | 設定 |
|---|---|
| 環境 | `highway-v0`（主）/ `merge-v0`（泛化驗證）/ `roundabout-v0` / `intersection-v0` |
| 動作空間 | `DiscreteMetaAction`（5 個離散動作） |
| 觀測 | Kinematics，5 台車，features: presence / x / y / vx / vy，normalized |
| 總訓練步數 | 100,000 timesteps（PPO rollout = 1024） |
| 每局時長 | `--duration 80` = 80 模擬秒 = **最多 400 個決策步**（policy_frequency=5Hz） |
| 訓練風格 | `base` / `conservative` / `aggressive`（影響 reward shaping） |
| 評估協定 | 訓練 aggressive，於共同 **base** 測試配置下跑 30 局 deterministic，量測碰撞率 / 存活步數 / 平均時速 / 換道次數 |

---

## 模組結構

```
src/
├── agents/
│   ├── networks.py      # AttentionActorCritic, MlpActorCritic, MlpQNetwork
│   ├── ppo.py           # 手刻 PPO（GAE + clipped surrogate + linear lr decay）
│   └── dqn.py           # Double DQN + replay buffer
├── envs/
│   └── reward_shaper.py # get_env_config()：多環境 × 多風格 reward 設定
├── train_modular.py     # 主訓練腳本（PPO / DQN，--seed / --env / --style）
├── train_sb3.py         # SB3 PPO 對照組（驗證手刻實作）
├── eval_report.py       # 平行行為評估 → 輸出 CSV + Markdown 表（--env / --pattern）
├── evaluate.py          # 單一模型評估
├── plot_comparison.py   # mean±std 學習曲線（跨 seed 彙整）
├── verify_stats.py      # 從原始 log 重算表格數字（整數性自我檢查）
├── run_merge.py         # merge-v0 泛化實驗批次（n=3）
└── make_video.py        # 並排對比影片 (DQN vs PPO)
```

> **路徑慣例**：所有 model / log 一律放 repo root 的 `models/` 與 `logs/`。請從 **repo root** 執行腳本（`root_dir="."`）。

---

## 快速上手

### 安裝
```bash
pip install -r requirements.txt
```

### 訓練
```bash
# 三個核心配置（aggressive 風格，seed=0）
python src/train_modular.py --algo ppo --arch attention --style aggressive --seed 0
python src/train_modular.py --algo ppo --arch mlp       --style aggressive --seed 0
python src/train_modular.py --algo dqn --arch mlp       --style aggressive --seed 0

# 第二個環境（merge-v0 泛化驗證，一次跑 3 configs × 3 seeds）
python src/run_merge.py
```

### 評估行為（核心結果）
```bash
# highway 三配置，30 局，base 測試config，輸出 CSV
python src/eval_report.py --episodes 30 --workers 1 --pattern aggressive --env highway-v0 --out logs/eval_results_aggressive.csv

# merge-v0 泛化
python src/eval_report.py --episodes 30 --workers 1 --pattern merge --env merge-v0 --out logs/eval_results_merge.csv
```

### 重現圖表與整數性檢查
```bash
python src/verify_stats.py                                  # 重算論文 Table II 數字
python src/plot_comparison.py --pattern aggressive_highway  # mean±std 學習曲線
```

### 產生成果影片
```bash
python src/make_video.py --env merge-v0 --episodes 4 --out logs/demo_merge.mp4
# 壓縮為小檔（不是壓縮檔，是降解析度/位元率）
ffmpeg -y -i logs/demo_merge.mp4 -vf "scale=1000:-2:flags=lanczos" -r 10 -c:v libx264 -pix_fmt yuv420p -crf 26 -preset slow demo_dqn_vs_ppo_merge.mp4
```

---

## 主要結果

### highway-v0（aggressive，n=5 seeds，base 測試）
| 配置 | 訓練獎勵 (mean±σ) | 碰撞率 | 存活步數 | 跨seed σ |
|---|---|---|---|---|
| PPO + Attention | 235.7 ± 16.7 | 60.7% | 199.9 | 16.7 |
| PPO + MLP | 223.0 ± 19.6 | 80.0% | 152.6 | 19.6 |
| **DQN + MLP** | 234.1 ± 7.7 | **11.3%** | **368.5** | **7.7（最低）** |

### merge-v0（aggressive，n=3 seeds，base 測試）— 泛化驗證
| 配置 | 訓練獎勵 (mean±σ) | 碰撞率 | 存活步數 |
|---|---|---|---|
| PPO + Attention | 58.3 ± 0.2 | 100% | 34.0 |
| PPO + MLP | 59.1 ± 0.3 | 100% | 35.0 |
| **DQN + MLP** | 59.0 ± 0.3 | **0.0%** | **77.7** |

> 所有數字皆可由 `src/verify_stats.py` 與 `logs/eval_results_*.csv` 重現。標準差為 population std (ddof=0)。

---

## 繳交物件對照
| 項目 | 檔案 |
|---|---|
| 報告書（IEEE 7 頁） | `paper.tex` → `paper.pdf` |
| 程式碼 | `src/` |
| 資料集 | 由訓練產生（`logs/`, `models/`），無外部 dataset |
| 成果影片 | `demo_dqn_vs_ppo_merge.mp4`（450 KB） |

---

## 參考文獻
- Leurent, E., & Mercat, J. (2019). *Social Attention for Autonomous Decision-Making in Dense Traffic*. NeurIPS Workshop.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Schulman, J., et al. (2015). *High-Dimensional Continuous Control Using GAE*. arXiv:1506.02438.
- van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep RL with Double Q-learning*. AAAI.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- [highway-env](https://github.com/Farama-Foundation/HighwayEnv) · [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) · [CleanRL](https://github.com/vwxyzjn/cleanrl)（PPO 實作參考）
