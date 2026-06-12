"""Parallel behaviour evaluation -> report-ready table.

Reuses run_evaluation() from evaluate.py but:
  - runs several models concurrently (ProcessPoolExecutor),
  - forces CPU (network is tiny; env is the bottleneck, so CPU avoids
    multi-process CUDA overhead),
  - writes logs/eval_results.csv and prints a Markdown table.

Usage:
    python eval_report.py --episodes 40 --workers 3
    python eval_report.py --episodes 40 --workers 3 --pattern aggressive
"""
import os, glob, argparse, csv
from concurrent.futures import ProcessPoolExecutor, as_completed


def _parse_name(pth):
    """custom_model_{algo}_{arch}_{style}_..._seed{N}.pth -> dict"""
    base = os.path.basename(pth).replace(".pth", "")
    parts = base.split("_")
    algo, arch, style = parts[2], parts[3], parts[4]
    seed = ""
    for p in parts:
        if p.startswith("seed"):
            seed = p.replace("seed", "")
    return dict(path=pth, algo=algo, arch=arch, style=style, seed=seed,
                label=f"{arch.upper()}-{algo.upper()} ({style},s{seed or '?'})")


def _worker(job, episodes, duration, env_id):
    import torch
    from evaluate import run_evaluation
    m = run_evaluation(job["path"], job["algo"], job["arch"], 4,
                       episodes, duration, torch.device("cpu"), env_id=env_id)
    return job, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--duration", type=int, default=80)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--env", type=str, default="highway-v0")
    ap.add_argument("--pattern", type=str, default="",
                    help="only evaluate models whose filename contains this")
    ap.add_argument("--out", type=str, default="logs/eval_results.csv")
    ap.add_argument("--model_dir", type=str, action="append", default=None,
                    help="model directory to search (can be specified multiple times)")
    args = ap.parse_args()

    model_dirs = args.model_dir if args.model_dir else ["models"]
    pths = []
    for d in model_dirs:
        pths.extend(glob.glob(os.path.join(d, "*seed*.pth")))
    pths = sorted(set(pths))
    if args.pattern:
        pths = [p for p in pths if args.pattern in os.path.basename(p)]
    jobs = [_parse_name(p) for p in pths]
    print(f"Evaluating {len(jobs)} models x {args.episodes} eps "
          f"on {args.env} with {args.workers} workers...\n")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, j, args.episodes, args.duration, args.env): j
                for j in jobs}
        for fut in as_completed(futs):
            job, m = fut.result()
            if m is None:
                print(f"  FAILED: {job['label']}")
                continue
            row = dict(label=job["label"], algo=job["algo"], arch=job["arch"],
                       style=job["style"], seed=job["seed"], **m)
            rows.append(row)
            print(f"  done: {job['label']:<28} crash={m['crash_rate']:5.1f}%  "
                  f"surv={m['avg_survival']:6.1f}  spd={m['avg_speed']:5.2f}  "
                  f"lc={m['avg_lane_changes']:5.2f}")

    rows.sort(key=lambda r: (r["style"], r["algo"], r["arch"], r["seed"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "algo", "arch", "style",
                                          "seed", "crash_rate", "avg_survival",
                                          "avg_speed", "avg_lane_changes"])
        w.writeheader()
        w.writerows(rows)

    # Markdown table for direct paste into the report
    print("\n\n| Config (arch-algo, style, seed) | Crash % | Survival | Speed (m/s) | Lane chg. |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['crash_rate']:.1f} | {r['avg_survival']:.1f} "
              f"| {r['avg_speed']:.2f} | {r['avg_lane_changes']:.2f} |")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
