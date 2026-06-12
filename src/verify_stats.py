"""Recompute Table II numbers from the raw training logs to verify integrity.
For each (config) it loads all 5 seed CSVs, computes per-seed final reward
(mean of last 10% of logged rows) and steps-to-avg>=150, then mean/std across
seeds. Prints per-seed values so claims in the paper can be checked one by one.
"""
import glob, os
import numpy as np
import pandas as pd

CONFIGS = {
    "PPO+Attention": "ppo_attention_aggressive",
    "PPO+MLP":       "ppo_mlp_aggressive",
    "DQN+MLP":       "dqn_mlp_aggressive",
}

def steps_to_threshold(df, thr=150.0):
    """First env-step where rolling avg_reward >= thr."""
    hit = df[df["avg_reward"] >= thr]
    if hit.empty:
        return None
    return int(hit.iloc[0]["step"])

for name, key in CONFIGS.items():
    print("=" * 64)
    print(name)
    print("-" * 64)
    finals, steps = [], []
    for seed in range(5):
        f = f"logs/training_log_{key}_highway_dur80_seed{seed}.csv"
        if not os.path.exists(f):
            print(f"  seed{seed}: MISSING {f}")
            continue
        df = pd.read_csv(f)
        n = len(df)
        tail = df["avg_reward"].iloc[max(0, int(n * 0.9)):]
        final = tail.mean()
        peak = df["avg_reward"].max()
        s150 = steps_to_threshold(df)
        finals.append(final)
        if s150 is not None:
            steps.append(s150)
        print(f"  seed{seed}: final(last10%)={final:7.2f}  peak={peak:7.2f}  "
              f"steps->150={s150}  rows={n}")
    finals = np.array(finals)
    print(f"  => MEAN final = {finals.mean():.2f}  STD = {finals.std(ddof=0):.2f} "
          f"(ddof=1: {finals.std(ddof=1):.2f})")
    if steps:
        print(f"  => steps->150: median={int(np.median(steps))}  "
              f"mean={int(np.mean(steps))}  all={sorted(steps)}")

# steps-to-150 percentage deltas
print("=" * 64)
print("Sample-efficiency deltas (using paper's medians):")
attn, mlp, dqn = 10240, 9216, 7721
print(f"  DQN vs Attention: {(attn-dqn)/attn*100:.1f}% fewer")
print(f"  DQN vs MLP:       {(mlp-dqn)/mlp*100:.1f}% fewer")
print(f"  Reward gap Attn-MLP: {(235.7-223.0)/235.7*100:.2f}% (of 235.7) | "
      f"{(235.7-223.0)/223.0*100:.2f}% (of 223.0)")
