#!/bin/bash
# ============================================================
# GCP training launch script — Zelda RL "Get the Sword" run
# Server: 30 vCPU / 120 GB RAM (e.g. c2-standard-30 Spot VM)
#
# Timestep math (Train.py: total_timesteps = max_steps * num_episodes):
#   max_steps=16384  num_episodes=3000  -> ~49.1M steps  (~4-6h @ 28 envs)
# ============================================================

set -e

RUN_NAME="run_sword_v1"
CHECKPOINT="run_20m_long/best_model.zip"

# Check checkpoint exists
if [ ! -f "${CHECKPOINT}.zip" ] && [ ! -d "${CHECKPOINT}" ]; then
    echo "[WARN] Checkpoint '${CHECKPOINT}' not found — starting from scratch."
    CHECKPOINT_FLAG=""
else
    echo "[INFO] Fine-tuning from: ${CHECKPOINT}"
    CHECKPOINT_FLAG="--checkpoint ${CHECKPOINT}"
fi

echo "[INFO] Starting run: ${RUN_NAME}"
echo "[INFO] Target: 16384 * 3000 = ~49.1M timesteps"
echo "[INFO] Curriculum: 70% init.state / 30% saved.state"

.venv/bin/python Train.py \
    --num_cpu 28 \
    --log_dir "${RUN_NAME}" \
    --max_steps 16384 \
    --num_episodes 3000 \
    --ent_coef 0.02 \
    --debug_freq 2000 \
    --coverage_freq 10000 \
    --text_map_freq 10000 \
    --checkpoint_freq 100000 \
    --curriculum init.state:0.7 saved.state:0.3 \
    ${CHECKPOINT_FLAG}
