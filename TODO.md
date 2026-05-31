# RL_final 改進待辦清單 (供 Claude Code 使用)

> **怎麼用:** 把這個檔放在專案根目錄,進到資料夾打 `claude`,然後說
> 「照 TODO.md 從 P0 開始一項一項做,每做完一項先給我 diff 跟說明,等我確認再繼續」。
> 規則:**一次只做一項**;改大東西前**先給計畫不要動手**;改完**跑起來驗證不報錯**;
> 只改被要求的部分,不要順手亂改其他東西。

---

## P0 — 先理解(不要改任何檔案)

- [ ] **盤點專案**
  - 目標:讓 Claude Code 建立正確的專案理解,後面所有任務都依賴這步。
  - 做什麼:讀過所有檔案,說明每支檔案職責、模組間如何互動,並列出任何明顯 bug、寫死的數值、或設計疑慮。
  - 完成標準:產出一份口頭摘要,**這階段不修改任何檔案**。

---

## P1 — 正確性與關鍵釐清(最優先,影響所有結論可信度)

- [ ] **驗證 PPO 數學實作**

  - 檔案:`agents/ppo.py`
  - 檢查:GAE 的 advantage 計算、advantage normalization、ratio 計算、clipped surrogate objective、value function 有無 clip、entropy bonus 正負號。
  - 完成標準:逐項指出正確/有疑慮處並解釋理由;有錯先提修法,**確認後再改**。

- [ ] **釐清訓練長度(「80 步」之謎)**

  - 目標:確認 README 寫的「統一測試長度為 80 步」到底是**訓練長度**還是**每局 episode horizon**。
  - 做什麼:找出 timestep / episode / horizon 的設定在哪,算出實際總訓練量。
  - 完成標準:明確回報「總共訓練 N timesteps / M episodes,每局最長 K 步」,並判斷對 highway-v0 而言是否足夠收斂。

- [ ] **釐清動作空間(離散 vs 連續)**

  - 目標:確認環境用的是 `DiscreteMetaAction` 還是連續動作。
  - 為什麼:README 寫「PPO 在連續環境的優勢」,但 highway-v0 預設是離散動作,措辭可能有誤。
  - 完成標準:回報實際動作空間設定,並指出 README 該如何修正用詞。

- [ ] **檢查 Attention vs MLP 參數量是否公平**
  - 檔案:`agents/networks.py`
  - 做什麼:印出 `AttentionActorCritic` 與 `MlpActorCritic` 的總參數量。
  - 完成標準:給出兩者參數量數字;若差距大,建議如何調整使消融比較公平。

---

## P2 — 統計嚴謹度(最容易補、CP 值最高)

- [ ] **加入 random seed 控制**

  - 檔案:`train_modular.py`
  - 做什麼:用 argparse 加 `--seed`,確保 numpy、torch(含 cudnn deterministic 視需要)、以及 highway 環境的 seed 都有設到。
  - 完成標準:同一 seed 跑兩次結果可重現。

- [ ] **支援多 seed 批次訓練與彙整**
  - 做什麼:讓訓練/繪圖能跑多個 seed(例如 3–5 個),log 能依 seed 分開存。
  - 檔案:`train_modular.py`、`plot_comparison.py`
  - 完成標準:`plot_comparison.py` 能畫出 **mean ± std**(陰影區間)的 learning curve,而不是單條線。

---

## P3 — 提升研究強度(想往論文/研討會推再做)

- [ ] **加一個有公信力的對照組驗證手刻實作**

  - 做什麼:用 stable-baselines3 的 PPO 在相同環境設定下跑一組,跟你手刻版比性能。
  - 完成標準:證明手刻 PPO 性能與 SB3 相當(代表實作沒有隱藏 bug)。

- [ ] **擴充到多個環境測泛化**
  - 做什麼:用同一套程式碼在 `merge-v0` / `roundabout-v0` / `intersection-v0` 至少多跑 1–2 個。
  - 檔案:`envs/reward_shaper.py`(確認 config 生成器支援多環境)
  - 完成標準:能用參數切換環境並完成訓練,產出跨環境比較表。

---

## P4 — 文件與定位(誠實化,避免被審稿人打臉)

- [ ] **修正 README 過度宣稱的用詞**
  - 做什麼:把「創新性地引入 Attention」改為誠實定位——attention 用於此類 ego + N 台 NPC 觀測,是 highway-env 設計時的標準做法(參考 Leurent & Mercat, _Social Attention for Autonomous Decision-Making in Dense Traffic_, 2019)。重點放在「重現並消融分析」而非「發明」。
  - 同時:把「遠勝」「證明」等過強用詞換成「在本實驗條件下觀察到」之類站得住腳的學術措辭。
  - 完成標準:README 定位誠實、措辭中性,但仍清楚呈現你的工程與實驗貢獻。

---

## 備註

- 每完成一項就把 `[ ]` 改成 `[x]`,方便追蹤。
- P1 是地基,務必先做完再進 P2 之後。
- P3 屬於「加分項」,時間不夠可先跳過,但 P1、P2 是讓結論可信的底線。
