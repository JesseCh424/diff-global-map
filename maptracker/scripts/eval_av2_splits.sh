#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/eval_av2_splits.sh [gpus]
# Example: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_av2_splits.sh 4

GPUS=${1:-4}
# Avoid port conflicts if multiple jobs run
export PORT=${PORT:-$(( 29500 + (RANDOM % 1000) ))}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using GPUS=$GPUS, PORT=$PORT"

# Ensure dataset PKLs and raw data are reachable
echo "Dataset check:"
ls -l datasets/av2 | sed -n '1,50p' || true

run_eval() {
  local CONFIG=$1
  local CKPT=$2
  local TAG=$3

  local OUTDIR="work_dirs/${TAG}"
  mkdir -p "$OUTDIR"
  echo "\n=== Running $TAG ==="
  echo "Config:    $CONFIG"
  echo "Checkpoint:$CKPT"
  echo "Outdir:    $OUTDIR"

  # Distributed test with evaluation + save_semantic
  bash tools/dist_test.sh "$CONFIG" "$CKPT" "$GPUS" --eval --eval-options save_semantic=True | tee -a "$OUTDIR/test_log.txt"
}

# AV2 oldsplit
run_eval \
  plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py \
  work_dirs/pretrained_ckpts/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/latest.pth \
  maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune

# AV2 newsplit
run_eval \
  plugin/configs/maptracker/av2_newsplit/maptracker_av2_newsplit_5frame_span10_stage3_joint_finetune.py \
  work_dirs/pretrained_ckpts/maptracker_av2_newsplit_5frame_span10_stage3_joint_finetune/latest.pth \
  maptracker_av2_newsplit_5frame_span10_stage3_joint_finetune

echo "All evaluations submitted. Logs in work_dirs/*/test_log.txt"

