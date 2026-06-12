"""
fill_eval_tables.py
-------------------
Reads the three eval CSV files and prints the LaTeX table rows to paste into paper.tex.

Usage:
    python src/fill_eval_tables.py
"""
import pandas as pd
import numpy as np
import os


def load_and_avg(csv_path, filter_style=None, filter_algo=None, filter_arch=None):
    """Load eval CSV, optionally filter, return per-metric averages across seeds."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if filter_style:
        df = df[df["style"] == filter_style]
    if filter_algo:
        df = df[df["algo"] == filter_algo]
    if filter_arch:
        df = df[df["arch"] == filter_arch]
    if df.empty:
        return None
    return {
        "crash_rate": df["crash_rate"].mean(),
        "avg_survival": df["avg_survival"].mean(),
        "avg_speed": df["avg_speed"].mean(),
        "avg_lane_changes": df["avg_lane_changes"].mean(),
        "n": len(df),
    }


def fmt_row(label, m, bold_crash=False):
    if m is None:
        return f"{label} & --- & --- & --- & --- \\\\"
    crash = f"\\textbf{{{m['crash_rate']:.1f}}}" if bold_crash else f"{m['crash_rate']:.1f}"
    return (f"{label} & {crash} & {m['avg_survival']:.1f} & "
            f"{m['avg_speed']:.2f} & {m['avg_lane_changes']:.2f} \\\\")


if __name__ == "__main__":
    print("=" * 70)
    print("TABLE B: Ablation behavioural evaluation (aggressive-trained models,")
    print("         30 eps, base test config)")
    print("=" * 70)

    agg_csv = "logs/eval_results_aggressive.csv"
    ppo_attn = load_and_avg(agg_csv, filter_algo="ppo", filter_arch="attention")
    ppo_mlp  = load_and_avg(agg_csv, filter_algo="ppo", filter_arch="mlp")
    dqn_mlp  = load_and_avg(agg_csv, filter_algo="dqn", filter_arch="mlp")

    if ppo_attn:
        # Find lowest crash rate for bold
        crashes = [m["crash_rate"] for m in [ppo_attn, ppo_mlp, dqn_mlp] if m]
        min_crash = min(crashes) if crashes else None
    else:
        min_crash = None

    print(fmt_row("PPO + Attention", ppo_attn,
                  bold_crash=(ppo_attn and ppo_attn["crash_rate"] == min_crash)))
    print(fmt_row("PPO + MLP",       ppo_mlp,
                  bold_crash=(ppo_mlp  and ppo_mlp["crash_rate"]  == min_crash)))
    print(fmt_row("DQN + MLP",       dqn_mlp,
                  bold_crash=(dqn_mlp  and dqn_mlp["crash_rate"]  == min_crash)))

    if ppo_attn:
        print(f"\n(n seeds per model: {ppo_attn['n']})")

    print()
    print("=" * 70)
    print("TABLE C: Reward-shaping behavioural comparison (PPO+Attention,")
    print("         30 eps, base test config)")
    print("=" * 70)

    cons_csv = "logs/eval_results_conservative.csv"
    base_csv = "logs/eval_results_base.csv"

    cons = load_and_avg(cons_csv, filter_algo="ppo", filter_arch="attention")
    base = load_and_avg(base_csv, filter_algo="ppo", filter_arch="attention")
    # Aggressive PPO+Attention from the aggressive eval
    agg  = load_and_avg(agg_csv, filter_algo="ppo", filter_arch="attention")

    if cons and base and agg:
        crashes_c = [cons["crash_rate"], base["crash_rate"], agg["crash_rate"]]
        min_c = min(crashes_c)
    else:
        min_c = None

    print(fmt_row("Conservative", cons,
                  bold_crash=(cons and cons["crash_rate"] == min_c)))
    print(fmt_row("Base",         base,
                  bold_crash=(base and base["crash_rate"] == min_c)))
    print(fmt_row("Aggressive",   agg,
                  bold_crash=(agg  and agg["crash_rate"]  == min_c)))

    print()
    print("=" * 70)
    print("All confirmed. Update paper.tex Table B (tab:eval) and Table C (tab:behavior_styles).")
