"""Summarize training logs into report-ready statistics.

Outputs, per config group (seeds aggregated):
  - final avg reward (mean of last 10 PPO updates / last 10% rows), mean +/- std across seeds
  - peak rolling avg reward
  - sample efficiency: first env-step at which rolling avg reward >= THRESHOLD

Usage:  python analyze_logs.py
"""
import glob, os, re
import numpy as np
import pandas as pd

LOGS = "logs"
THRESHOLD = 150.0  # rolling-avg reward threshold for "sample efficiency" (aggressive scale)


def reward_col(df):
    return df["avg_reward"].values.astype(float)


def step_col(df):
    return df["step"].values.astype(float)


def summarize_file(path):
    df = pd.read_csv(path)
    if "avg_reward" not in df.columns:
        return None
    r = reward_col(df)
    s = step_col(df)
    k = max(1, len(r) // 10)
    final = float(np.mean(r[-k:]))
    peak = float(np.max(r))
    hit = np.where(r >= THRESHOLD)[0]
    steps_to_thr = float(s[hit[0]]) if len(hit) else float("nan")
    return dict(final=final, peak=peak, steps_to_thr=steps_to_thr, n_rows=len(r), last_step=float(s[-1]))


def group_key(name):
    # strip leading 'training_log_' and trailing '.csv' and _seed\d+
    base = name[len("training_log_"):-len(".csv")]
    return re.sub(r"_seed\d+", "", base)


def main():
    files = sorted(glob.glob(os.path.join(LOGS, "training_log_*.csv")))
    groups = {}
    for f in files:
        name = os.path.basename(f)
        if "_seed" not in name:
            continue  # skip old un-seeded 'final' logs
        if os.path.getsize(f) < 10:
            continue  # skip empty files (aborted runs)
        st = summarize_file(f)
        if st is None:
            continue
        groups.setdefault(group_key(name), []).append((name, st))

    print(f"{'config':<48} {'n':>2} {'final(mean+/-std)':>20} {'peak':>8} {'steps>=%g':>10}".replace('%g', str(int(THRESHOLD))))
    print("-" * 92)
    for g in sorted(groups):
        items = groups[g]
        finals = np.array([s["final"] for _, s in items])
        peaks = np.array([s["peak"] for _, s in items])
        thr = np.array([s["steps_to_thr"] for _, s in items])
        complete = [s for _, s in items if s["n_rows"] >= 90 or "dqn" in g]
        n = len(items)
        mean, std = finals.mean(), finals.std()
        thr_mean = np.nanmean(thr)
        print(f"{g:<48} {n:>2} {mean:>11.1f} +/- {std:>5.1f} {peaks.max():>8.1f} {thr_mean:>10.0f}")
        for name, s in items:
            flag = "" if (s["n_rows"] >= 90 or "dqn" in g) else "  <-- INCOMPLETE"
            print(f"    {name:<44} rows={s['n_rows']:>3} last_step={s['last_step']:>7.0f} "
                  f"final={s['final']:.1f} peak={s['peak']:.1f}{flag}")
    print("-" * 92)


if __name__ == "__main__":
    main()
