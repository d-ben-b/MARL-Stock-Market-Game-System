#!/usr/bin/env bash
# Reproducible experiment driver for the RL_final ablation study.
# Completes the multi-seed core comparison (seeds 1-2; seed 0 already exists)
# and the stable-baselines3 PPO validation baseline (seeds 0-2).
#
# Run from the src/ directory:  bash run_experiments.sh
# Logs/models land in src/logs and src/models (same as the existing seed-0 runs).
#
# PYTHONUTF8=1 avoids the Windows cp950 console crash on the emoji prints.

set -u
PY="../.venv/Scripts/python.exe"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

run() {  # run <logtag> <command...>
  echo "[$(date +%H:%M:%S)] START $1"
  "${@:2}"
  echo "[$(date +%H:%M:%S)] END   $1 (exit $?)"
}

# --- Core multi-seed ablation: aggressive / highway-v0 / dur80 (seed 0 done) ---
batch() {  # run up to 3 jobs in parallel, then wait
  "$@" &
}

for SEED in 1 2; do
  echo "===== SEED $SEED batch (PPO-Attn, PPO-MLP, DQN) ====="
  PYTHONUTF8=1 $PY train_modular.py --algo ppo --arch attention --style aggressive --seed $SEED &
  PYTHONUTF8=1 $PY train_modular.py --algo ppo --arch mlp       --style aggressive --seed $SEED &
  PYTHONUTF8=1 $PY train_modular.py --algo dqn --arch mlp       --style aggressive --seed $SEED &
  wait
  echo "===== SEED $SEED batch done ====="
done

# --- SB3 PPO validation baseline: aggressive / dur80, seeds 0-2 ---
echo "===== SB3 baseline batch (seeds 0,1,2) ====="
PYTHONUTF8=1 $PY train_sb3.py --style aggressive --seed 0 &
PYTHONUTF8=1 $PY train_sb3.py --style aggressive --seed 1 &
PYTHONUTF8=1 $PY train_sb3.py --style aggressive --seed 2 &
wait
echo "===== ALL EXPERIMENTS DONE [$(date +%H:%M:%S)] ====="
