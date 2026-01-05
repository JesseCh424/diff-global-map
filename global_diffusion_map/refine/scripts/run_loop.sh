#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=4 bash global_diffusion_map/refine/scripts/run_loop.sh \
#     --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
#     --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
#     --scene-file /tmp/clean10.txt \
#     --out-root global_diffusion_map/refine/work_dirs/train_gpu_fast_persistent \
#     --epochs 2000 --batch 16 --save-every 50

MAX_RETRIES=${MAX_RETRIES:-100}
SLEEP_SECS=${SLEEP_SECS:-5}

ARGS=("$@")

count=0
while [[ $count -lt $MAX_RETRIES ]]; do
  echo "=================================================="
  echo "[run_loop] Training session $count starting..."
  echo "=================================================="

  # Launch training; rely on auto-resume inside the script
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u global_diffusion_map/refine/clean/train_full_gpu_fast.py "${ARGS[@]}"

  EXIT_CODE=$?
  if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[run_loop] Training finished successfully."
    exit 0
  fi

  echo "[run_loop] Training exited with code $EXIT_CODE (OOM or planned restart)."
  echo "[run_loop] Cleaning and restarting in ${SLEEP_SECS}s..."

  # Forcefully clear any stray python processes of the current user (optional: comment out if multi-user)
  pkill -u "$(whoami)" -9 python || true
  sleep "$SLEEP_SECS"

  ((count++))
done

echo "[run_loop] Reached MAX_RETRIES=$MAX_RETRIES; exiting."
exit 1

