#!/usr/bin/env bash
set -euo pipefail
# Ensure Python prints unbuffered so logs appear promptly under torchrun
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
# Avoid torchrun default warning and reduce oversubscription
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MASTER_PORT=${MASTER_PORT:-29500}

# Full-data EDM refine training (multi-scene, 70/30 sim/real mix)
# Example (AV2 old split, train):
#   bash -x global_diffusion_map/refine/scripts/train_full.sh \
#     maptracker/work_dirs/static_gt_vector/av2_oldsplit/train \
#     maptracker/work_dirs/rendered_gt/av2_oldsplit/train \
#     maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train \
#     global_diffusion_map/refine/work_dirs/train_edm_full_av2_old

STATIC_ROOT=${1:?static root with <scene>.pkl}
RENDERED_ROOT=${2:?rendered 10 root}
AGG_PRED_ROOT=${3:?aggregated predictions root}
OUT_ROOT=${4:?output root}

## Optimized defaults for 4x48GB
# Global batch size (DataParallel 按 GPU 数自动切分)
BATCH=${BATCH:-48}
# 每 GPU 预期样本（仅显示用途，实际由 DP 自动划分）
BATCHGPU=${BATCHGPU:-12}
# 学习率默认（根据近期稳定值）
LR=${LR:-0.0002}
# 训练时长：更长 EPOCHS，适度降低 ITERS
EPOCHS=${EPOCHS:-100}
ITERS=${ITERS:-300}
MIX_REAL=${MIX_REAL:-0.3}
STEPS=${STEPS:-18}
SIGMA_MIN=${SIGMA_MIN:-0.002}
SIGMA_MAX=${SIGMA_MAX:-1.5}
ALPHA=${ALPHA:-0.03}
SMOOTH_WEIGHT=${SMOOTH_WEIGHT:-}
REG_LEN_EXP=${REG_LEN_EXP:-}
SMOOTH_INV_LEN_EXP=${SMOOTH_INV_LEN_EXP:-}
SAVE_VIZ=${SAVE_VIZ:-0}
AMP=${AMP:-1}
# Control Heun (second-order) correction; default on to match prior behavior
SECOND_ORDER=${SECOND_ORDER:-1}
# Optional: increase inner-loop print cadence (e.g., LOG_EVERY=1)
LOG_EVERY=${LOG_EVERY:-}
WORKERS=${WORKERS:-12}
FIND_UNUSED=${FIND_UNUSED:-0}
ACCUM_STEPS=${ACCUM_STEPS:-}
PREFETCH=${PREFETCH:-}
# Optional loss weighting
STEP_LOSS_WEIGHT=${STEP_LOSS_WEIGHT:-}
RESUME_CKPT=${RESUME_CKPT:-}
CKPT_EVERY=${CKPT_EVERY:-}
CKPT_ITERS=${CKPT_ITERS:-}

# Collect optional CLI flags consistently for both DDP and non-DDP branches
EXTRA_ARGS=()
if [ "${SAVE_VIZ}" = "1" ]; then EXTRA_ARGS+=(--save-viz); fi
if [ "${AMP}" = "1" ]; then EXTRA_ARGS+=(--amp); fi
if [ "${SECOND_ORDER}" = "1" ]; then EXTRA_ARGS+=(--second-order); fi
if [ -n "${LOG_EVERY}" ]; then EXTRA_ARGS+=(--log-every "${LOG_EVERY}"); fi
if [ "${FIND_UNUSED}" = "1" ]; then EXTRA_ARGS+=(--find-unused); fi
if [ -n "${ACCUM_STEPS}" ]; then EXTRA_ARGS+=(--accum-steps "${ACCUM_STEPS}"); fi
if [ -n "${PREFETCH}" ]; then EXTRA_ARGS+=(--prefetch "${PREFETCH}"); fi
if [ -n "${STEP_LOSS_WEIGHT}" ]; then EXTRA_ARGS+=(--step-loss-weight "${STEP_LOSS_WEIGHT}"); fi
if [ -n "${CKPT_EVERY}" ]; then EXTRA_ARGS+=(--ckpt-every "${CKPT_EVERY}"); fi
if [ -n "${CKPT_ITERS}" ]; then EXTRA_ARGS+=(--ckpt-iters "${CKPT_ITERS}"); fi
if [ -n "${SMOOTH_WEIGHT}" ]; then EXTRA_ARGS+=(--smooth-weight "${SMOOTH_WEIGHT}"); fi
if [ -n "${REG_LEN_EXP}" ]; then EXTRA_ARGS+=(--reg-len-exp "${REG_LEN_EXP}"); fi
if [ -n "${SMOOTH_INV_LEN_EXP}" ]; then EXTRA_ARGS+=(--smooth-inv-len-exp "${SMOOTH_INV_LEN_EXP}"); fi
if [ -n "${RESUME_CKPT}" ]; then EXTRA_ARGS+=(--resume-ckpt "${RESUME_CKPT}"); fi

# Show detected GPUs and per-GPU batch split (informational)
NGPUS=$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PY
)
echo "[info] GPUs=$NGPUS  GlobalBatch=$BATCH  ~PerGPU=$(python - <<PY
import math
ng=int('$NGPUS'); bs=int('$BATCH');
print(max(1, math.ceil(bs/max(1,ng))))
PY
)  LR=$LR  EPOCHS=$EPOCHS  ITERS=$ITERS  MIX_REAL=$MIX_REAL  SMOOTH=$SMOOTH_WEIGHT PORT=$MASTER_PORT"

mkdir -p "$OUT_ROOT"
if [ "${DDP:-0}" != "0" ]; then
  # Use torchrun to launch DDP across visible GPUs
  torchrun --nproc_per_node "$NGPUS" --master_port "$MASTER_PORT" global_diffusion_map/refine/train_edm_full.py \
    --ddp \
    --static-root   "$STATIC_ROOT" \
    --rendered-root "$RENDERED_ROOT" \
    --agg-pred-root "$AGG_PRED_ROOT" \
    --epochs "$EPOCHS" --iters-per-epoch "$ITERS" --batch "$BATCH" \
    --mix-real-prob "$MIX_REAL" \
    --steps "$STEPS" --sigma-min "$SIGMA_MIN" --sigma-max "$SIGMA_MAX" \
    --alpha "$ALPHA" \
    --lr "$LR" \
    --workers "$WORKERS" \
    --out-root "$OUT_ROOT" \
    "${EXTRA_ARGS[@]}"
else
  python -u global_diffusion_map/refine/train_edm_full.py \
    --static-root   "$STATIC_ROOT" \
    --rendered-root "$RENDERED_ROOT" \
    --agg-pred-root "$AGG_PRED_ROOT" \
    --epochs "$EPOCHS" --iters-per-epoch "$ITERS" --batch "$BATCH" \
    --mix-real-prob "$MIX_REAL" \
    --steps "$STEPS" --sigma-min "$SIGMA_MIN" --sigma-max "$SIGMA_MAX" \
    --alpha "$ALPHA" \
    --lr "$LR" \
    --workers "$WORKERS" \
    --out-root "$OUT_ROOT" \
    "${EXTRA_ARGS[@]}"
fi

echo "[ok] training finished. Output: $OUT_ROOT"
