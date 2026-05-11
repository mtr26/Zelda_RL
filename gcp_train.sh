#!/bin/bash
# ============================================================
# GCP training launch script — Zelda RL "Get the Sword" run
# Server: 30 vCPU / 120 GB RAM (e.g. c2-standard-30 Spot VM)
#
# Timestep math (Train.py: total_timesteps = max_steps * num_episodes):
#   max_steps=16384  num_episodes=3000  -> ~49.1M steps  (~4-6h @ 28 envs)
# ============================================================

set -e

RUN_NAME="run_hybrid_v2"
CHECKPOINT=""

# Check checkpoint exists
if [ -n "${CHECKPOINT}" ] && ([ -f "${CHECKPOINT}" ] || [ -d "${CHECKPOINT}" ]); then
    echo "[INFO] Fine-tuning from: ${CHECKPOINT}"
    CHECKPOINT_FLAG="--checkpoint ${CHECKPOINT}"
else
    echo "[INFO] No checkpoint — training from scratch."
    CHECKPOINT_FLAG=""
fi

echo "[INFO] Starting run: ${RUN_NAME}"
echo "[INFO] Target: 16384 * 1250 = ~20.4M timesteps"
echo "[INFO] Curriculum: 100% init.state (Phase 1 — from scratch)"

.venv/bin/python Train.py \
    --num_cpu 48 \
    --log_dir "${RUN_NAME}" \
    --max_steps 16384 \
    --num_episodes 1250 \
    --ent_coef 0.02 \
    --debug_freq 2000 \
    --coverage_freq 10000 \
    --text_map_freq 10000 \
    --checkpoint_freq 100000 \
    --curriculum init.state:1.0 \
    ${CHECKPOINT_FLAG}
