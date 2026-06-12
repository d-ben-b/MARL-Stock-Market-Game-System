import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# 獲取當前虛擬環境的 Python 直譯器絕對路徑
python_exe = sys.executable

commands = [
    # 核心消融實驗 (補齊 seed 3, 4)
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "dqn",
        "--arch",
        "mlp",
        "--style",
        "aggressive",
        "--seed",
        "3",
    ],
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "dqn",
        "--arch",
        "mlp",
        "--style",
        "aggressive",
        "--seed",
        "4",
    ],
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "ppo",
        "--arch",
        "attention",
        "--style",
        "aggressive",
        "--seed",
        "3",
    ],
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "ppo",
        "--arch",
        "attention",
        "--style",
        "aggressive",
        "--seed",
        "4",
    ],
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "ppo",
        "--arch",
        "mlp",
        "--style",
        "aggressive",
        "--seed",
        "3",
    ],
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "ppo",
        "--arch",
        "mlp",
        "--style",
        "aggressive",
        "--seed",
        "4",
    ],
    # 獎勵塑形展示 (補齊 seed 1, 2，僅使用 PPO+Attention)
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "ppo",
        "--arch",
        "attention",
        "--style",
        "base",
        "--seed",
        "1",
    ],
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "ppo",
        "--arch",
        "attention",
        "--style",
        "base",
        "--seed",
        "2",
    ],
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "ppo",
        "--arch",
        "attention",
        "--style",
        "conservative",
        "--seed",
        "1",
    ],
    [
        python_exe,
        "./src/train_modular.py",
        "--algo",
        "ppo",
        "--arch",
        "attention",
        "--style",
        "conservative",
        "--seed",
        "2",
    ],
]

MAX_CONCURRENT = 3


def run_cmd(cmd):
    cmd_str = " ".join(cmd)
    print(f"啟動: {cmd_str}")
    try:
        process = subprocess.Popen(cmd)
        process.wait()

        if process.returncode == 0:
            print(f"完成: {cmd_str}")
        else:
            print(f"錯誤 (Return Code {process.returncode}): {cmd_str}")
    except Exception as e:
        print(f"例外錯誤 {cmd_str}: {e}")


if __name__ == "__main__":
    print(f"準備執行 {len(commands)} 個缺失的訓練任務...")
    print(f"最大並行數限制為: {MAX_CONCURRENT}")
    print("-" * 50)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        executor.map(run_cmd, commands)

    print("-" * 50)
    print("所有實驗排程已執行完畢。")
