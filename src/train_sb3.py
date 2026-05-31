"""
SB3 PPO baseline — runs on the same highway-v0 config as train_modular.py.
Purpose: verify custom PPO implementation is in the correct performance range.

Usage:
    python train_sb3.py --style aggressive --seed 0
    python train_sb3.py --style aggressive --seed 1
    python train_sb3.py --style conservative --seed 0
"""
import os
import csv
import argparse
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 — registers highway envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from envs.reward_shaper import get_highway_config


class CsvLoggerCallback(BaseCallback):
    """Logs (step, avg_reward) to CSV after every rollout."""

    def __init__(self, csv_writer, csv_file, window: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.csv_writer = csv_writer
        self.csv_file = csv_file
        self.window = window

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.ep_info_buffer
        if len(buf) == 0:
            return
        recent = list(buf)[-self.window:]
        avg = float(np.mean([ep["r"] for ep in recent]))
        self.csv_writer.writerow([self.num_timesteps, avg])
        self.csv_file.flush()


def main():
    parser = argparse.ArgumentParser(description="Train SB3 PPO baseline on highway-v0")
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument(
        "--style", type=str, default="aggressive",
        choices=["base", "conservative", "aggressive"],
    )
    parser.add_argument("--duration", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    args = parser.parse_args()

    logs_dir = os.path.join(args.root_dir, "logs")
    models_dir = os.path.join(args.root_dir, "models")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    base_filename = f"ppo_sb3_{args.style}_dur{args.duration}_seed{args.seed}"
    csv_path = os.path.join(logs_dir, f"training_log_{base_filename}.csv")
    model_path = os.path.join(models_dir, f"sb3_{base_filename}.zip")

    config = get_highway_config(args.style, duration=args.duration)
    env = gym.make("highway-v0")
    env.unwrapped.configure(config)
    env = Monitor(env)

    # Hyperparameters matched to custom PPO in train_modular.py
    # Note: SB3 MlpPolicy default net_arch=[64,64]; architecture differs from
    # our custom networks but the algorithm itself is identical — that is what
    # we are validating here.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=0,
    )

    print(f"=== SB3 PPO Baseline ({args.style}, seed={args.seed}) ===")

    with open(csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "avg_reward"])
        callback = CsvLoggerCallback(csv_writer, csv_file)
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
        )

    model.save(model_path)
    env.close()

    print(f"Model saved : {model_path}")
    print(f"Log saved   : {csv_path}")


if __name__ == "__main__":
    main()
