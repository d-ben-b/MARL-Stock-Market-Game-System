# RL_final — PPO / DQN Ablation Study on highway-env

本專案在 [highway-env](https://github.com/Farama-Foundation/HighwayEnv) 的離散動作場景下，重現 PPO 與 DQN 演算法，並針對網路架構（Self-Attention vs MLP）進行消融分析。

---

## 專案定位

本專案的核心貢獻是**重現與消融分析**，而非提出新演算法：

- 手刻實現 PPO（含 GAE、CleanRL-style dones indexing、linear lr decay）與 Double DQN
- 以 Self-Attention 與 MLP 兩種 feature extractor 進行架構消融
- 以 stable-baselines3 PPO 為對照組，驗證手刻實作的正確性
- 支援多 seed 訓練，並以 mean ± std 陰影區間呈現學習曲線

**關於 Self-Attention 的定位：**
將 Self-Attention 應用於 ego + N 台 NPC 的 Kinematics 觀測，是 highway-env 設計時的標準做法，參見 Leurent & Mercat, *Social Attention for Autonomous Decision-Making in Dense Traffic* (2019)。本專案採用此架構並與 MLP baseline 進行對照實驗，目的是量化其在本實驗條件下的效果差異，而非宣稱方法上的創新。

---

## 環境設定

| 項目 | 設定 |
|---|---|
| 環境 | `highway-v0`（預設）/ `merge-v0` / `roundabout-v0` / `intersection-v0` |
| 動作空間 | `DiscreteMetaAction`（5 個離散動作） |
| 觀測 | Kinematics，5 台車，features: presence / x / y / vx / vy，normalized |
| 總訓練步數 | 100,000 timesteps |
| 每局最長步數 | 80 步（`--duration 80`） |
| 訓練風格 | `base` / `conservative` / `aggressive`（影響 reward shaping） |

> 注意：動作空間為**離散**。PPO 在此場景的適用性來自其 on-policy 穩定性與 clip 機制，而非連續動作控制能力。

---

## 模組結構

```
src/
├── agents/
│   ├── networks.py      # AttentionActorCritic, MlpActorCritic, MlpQNetwork
│   ├── ppo.py           # 手刻 PPO（GAE + clipped surrogate）
│   └── dqn.py           # Double DQN + replay buffer
├── envs/
│   └── reward_shaper.py # get_env_config()：多環境 reward 設定
├── train_modular.py     # 主訓練腳本（PPO / DQN，支援 --seed / --env）
├── train_sb3.py         # SB3 PPO 對照組
├── evaluate.py          # 100 局測試，輸出指標表格
└── plot_comparison.py   # mean±std 學習曲線（跨 seed 彙整）
```

---

## 網路架構參數量

| 架構 | 總參數量 | 說明 |
|---|---|---|
| AttentionActorCritic | ~66,822 | MultiheadAttention + actor/critic heads |
| MlpActorCritic | ~72,262 | Flatten → 64 → 320 → actor/critic heads |

兩者參數量相近（差距 < 10%），消融比較具公平性。

---

## 快速上手

### 安裝

```bash
pip install highway-env stable-baselines3 torch gymnasium tqdm matplotlib pandas
```

### 訓練

```bash
# PPO + Attention，aggressive 風格，seed=0
python src/train_modular.py --algo ppo --arch attention --style aggressive --seed 0

# PPO + MLP，同設定
python src/train_modular.py --algo ppo --arch mlp --style aggressive --seed 0

# DQN baseline
python src/train_modular.py --algo dqn --arch mlp --style aggressive --seed 0

# SB3 PPO 對照組
python src/train_sb3.py --style aggressive --seed 0

# 不同環境
python src/train_modular.py --env roundabout-v0 --algo ppo --arch attention --style base --seed 0
```

### 多 seed 批次（Windows PowerShell）

```powershell
foreach ($seed in 0,1,2) {
    python src/train_modular.py --algo ppo --arch attention --style aggressive --seed $seed
}
```

### 繪製學習曲線

```bash
# 所有 aggressive 配置的 mean±std 曲線
python src/plot_comparison.py --pattern "aggressive" --title "Aggressive Style Comparison"

# 所有配置
python src/plot_comparison.py
```

### 評估

```bash
# 評估單一模型
python src/evaluate.py --model_path models/custom_model_ppo_attention_aggressive_highway_dur80_seed0.pth --algo ppo --arch attention

# 掃描並評估所有模型
python src/evaluate.py --all --env highway-v0
```

---

## 實驗觀察

以下結果均在 `highway-v0 / aggressive / dur=80` 設定下取得，僅代表本實驗條件下的觀察，不作為一般性結論：

- **在本實驗條件下觀察到**：PPO + Attention 的平均回合獎勵高於 PPO + MLP，差距在 training 後期較為明顯
- **在本實驗條件下觀察到**：PPO 系列收斂速度優於 DQN MLP baseline（可能與 on-policy sample efficiency 有關）
- 手刻 PPO 與 SB3 PPO 的學習曲線趨勢相近，顯示實作無明顯系統性 bug

如需更強的統計結論，建議增加 seed 數（≥5）並在多個環境上重複驗證。

---

## 參考文獻

- Leurent, E., & Mercat, J. (2019). *Social Attention for Autonomous Decision-Making in Dense Traffic*. NeurIPS Workshop.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- [highway-env](https://github.com/Farama-Foundation/HighwayEnv) — Farama Foundation
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — PPO 實作參考
