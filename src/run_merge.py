"""Train the 3 core configs on merge-v0 (aggressive style) for a generalization
probe. Mirrors run_missing.py: ThreadPoolExecutor with a concurrency cap.

Run from the REPO ROOT so outputs land in ./models and ./logs (consistent with
the existing highway-v0 data):
    python src/run_merge.py
"""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

python_exe = sys.executable

# 3 core configs x seeds 0,1,2  (n=3 generalization probe)
CONFIGS = [
    ("ppo", "attention"),
    ("ppo", "mlp"),
    ("dqn", "mlp"),
]
SEEDS = [0, 1, 2]

commands = []
for algo, arch in CONFIGS:
    for seed in SEEDS:
        commands.append([
            python_exe, "./src/train_modular.py",
            "--algo", algo, "--arch", arch,
            "--style", "aggressive", "--env", "merge-v0",
            "--seed", str(seed),
        ])

MAX_CONCURRENT = 3


def run_cmd(cmd):
    cmd_str = " ".join(cmd)
    print(f"啟動: {cmd_str}", flush=True)
    try:
        process = subprocess.Popen(cmd)
        process.wait()
        if process.returncode == 0:
            print(f"完成: {cmd_str}", flush=True)
        else:
            print(f"錯誤 (Return Code {process.returncode}): {cmd_str}", flush=True)
    except Exception as e:
        print(f"例外錯誤 {cmd_str}: {e}", flush=True)


if __name__ == "__main__":
    print(f"準備執行 {len(commands)} 個 merge-v0 訓練任務 (n={len(SEEDS)} seeds)...")
    print(f"最大並行數限制為: {MAX_CONCURRENT}")
    print("-" * 50)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        list(executor.map(run_cmd, commands))
    print("-" * 50)
    print("所有 merge-v0 實驗已執行完畢。")
