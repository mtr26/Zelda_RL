#!/bin/bash
# ============================================================
# GCP training launch script — Zelda RL "Get the Sword" run
# Server: 8 vCPU / 32 GB RAM
#
# Timestep math (Train.py: total_timesteps = max_steps * num_episodes):
#   max_steps=16384  num_episodes=1500  -> 24.6M steps  (~21h @ 7 envs)
#   (previous run: max_steps=8192 * num_episodes=3000 = 24.6M in 15h @ 10 envs)
# ============================================================

set -e

RUN_NAME="run_sword_v1"
CHECKPOINT="run_20m_long/best_model"

# Check checkpoint exists
if [ ! -f "${CHECKPOINT}.zip" ] && [ ! -d "${CHECKPOINT}" ]; then
    echo "[WARN] Checkpoint '${CHECKPOINT}' not found — starting from scratch."
    CHECKPOINT_FLAG=""
else
    echo "[INFO] Fine-tuning from: ${CHECKPOINT}"
    CHECKPOINT_FLAG="--checkpoint ${CHECKPOINT}"
fi

echo "[INFO] Starting run: ${RUN_NAME}"
echo "[INFO] Target: 16384 * 1500 = ~24.6M timesteps"
echo "[INFO] Curriculum: 70% init.state / 30% saved.state"

.venv/bin/python Train.py \
    --num_cpu 7 \
    --log_dir "${RUN_NAME}" \
    --max_steps 16384 \
    --num_episodes 1500 \
    --ent_coef 0.02 \
    --debug_freq 2000 \
    --coverage_freq 10000 \
    --text_map_freq 10000 \
    --checkpoint_freq 100000 \
    --curriculum init.state:0.7 saved.state:0.3 \
    ${CHECKPOINT_FLAG}
