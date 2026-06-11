#!/usr/bin/env bash
# Lean continuation (time-boxed): completes a 2-seed core ablation
# (seed 0 already trained; this adds seed 1) plus one SB3 validation seed.
# Runs the seed-1 trio and SB3 seed-0 concurrently (4 processes).
#
# Run from src/:  bash run_remaining.sh
set -u
PY="../.venv/Scripts/python.exe"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "[$(date +%H:%M:%S)] START seed-1 trio + SB3 seed-0 (4 parallel)"
PYTHONUTF8=1 $PY train_modular.py --algo ppo --arch attention --style aggressive --seed 1 &
PYTHONUTF8=1 $PY train_modular.py --algo ppo --arch mlp       --style aggressive --seed 1 &
PYTHONUTF8=1 $PY train_modular.py --algo dqn --arch mlp       --style aggressive --seed 1 &
PYTHONUTF8=1 $PY train_sb3.py --style aggressive --seed 0 &
wait
echo "[$(date +%H:%M:%S)] ===== ALL REMAINING EXPERIMENTS DONE ====="
