import os
import argparse
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path):
    df = pd.read_csv(path)
    return df["step"].values.astype(float), df["avg_reward"].values.astype(float)


def interpolate_to_common_x(all_x, all_y, n_points=300):
    x_min = max(x[0] for x in all_x)
    x_max = min(x[-1] for x in all_x)
    x_common = np.linspace(x_min, x_max, n_points)
    y_interp = [np.interp(x_common, x, y) for x, y in zip(all_x, all_y)]
    return x_common, np.array(y_interp)


def main():
    parser = argparse.ArgumentParser(description="Plot mean±std learning curves across seeds")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument(
        "--pattern", type=str, default="*",
        help="glob pattern for config name, e.g. 'ppo_attention_aggressive'",
    )
    parser.add_argument("--output", type=str, default="", help="output image path")
    parser.add_argument("--title", type=str, default="Learning Curves (mean ± std)")
    args = parser.parse_args()

    csv_files = glob.glob(
        os.path.join(args.logs_dir, f"training_log_{args.pattern}*_seed*.csv")
    )

    if not csv_files:
        print(
            f"No CSVs found in '{args.logs_dir}' matching pattern '{args.pattern}' with _seed* suffix.\n"
            "Make sure you trained with --seed so filenames contain '_seed<N>'."
        )
        return

    # Group files by base config (strip _seed\d+ from name)
    groups: dict[str, list[str]] = {}
    for f in sorted(csv_files):
        basename = os.path.basename(f)
        base = re.sub(
            r"_seed\d+", "",
            basename.removeprefix("training_log_").removesuffix(".csv"),
        )
        groups.setdefault(base, []).append(f)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for i, (base_name, files) in enumerate(sorted(groups.items())):
        color = colors[i % len(colors)]
        all_x, all_y = [], []

        for f in files:
            try:
                x, y = load_csv(f)
                all_x.append(x)
                all_y.append(y)
            except Exception as e:
                print(f"Warning: skipping {f}: {e}")

        if not all_x:
            continue

        n = len(all_x)
        label = f"{base_name} (n={n})"

        if n == 1:
            ax.plot(all_x[0], all_y[0], label=label, color=color, alpha=0.8)
        else:
            x_common, y_matrix = interpolate_to_common_x(all_x, all_y)
            mean = y_matrix.mean(axis=0)
            std = y_matrix.std(axis=0)
            ax.plot(x_common, mean, label=label, color=color, linewidth=1.8)
            ax.fill_between(x_common, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Avg Reward (rolling 20 eps)")
    ax.set_title(args.title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    output = args.output or os.path.join(
        args.logs_dir, f"comparison_{args.pattern}.png"
    )
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved: {output}")
    plt.close()


if __name__ == "__main__":
    main()
