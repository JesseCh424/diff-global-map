#!/usr/bin/env bash
set -euo pipefail

# One-scene refine overfit + two-mode inference (GT+noise, Proposal)
# Usage:
#   SCENE=<scene_id> \
#   STATIC_ROOT=maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
#   RENDERED_ROOT=maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
#   AGG_PRED_ROOT=maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid \
#   EPOCHS=50 STEPS=18 SIGMA_MIN=0.002 SIGMA_MAX=1.5 \
#   bash -x global_diffusion_map/refine/scripts/one_scene_overfit_and_infer.sh

: "${SCENE:?SCENE is required}"
: "${STATIC_ROOT:?STATIC_ROOT is required}"
: "${RENDERED_ROOT:?RENDERED_ROOT is required}"
: "${AGG_PRED_ROOT:?AGG_PRED_ROOT is required}"

: "${EPOCHS:=50}"
: "${STEPS:=18}"
: "${SIGMA_MIN:=0.002}"
: "${SIGMA_MAX:=1.5}"
: "${SMOOTH_WEIGHT:=0.05}"

OUT_TRAIN="global_diffusion_map/refine/work_dirs/train_edm_one_scene_${SCENE}"

echo "[train] scene=${SCENE} epochs=${EPOCHS} steps=${STEPS} sigma=[${SIGMA_MIN},${SIGMA_MAX}]"
python -u global_diffusion_map/refine/train_edm_one_scene.py \
  --static-root   "${STATIC_ROOT}" \
  --rendered-root "${RENDERED_ROOT}" \
  --agg-pred-root "${AGG_PRED_ROOT}" \
  --scene "${SCENE}" \
  --epochs "${EPOCHS}" \
  --steps  "${STEPS}" \
  --sigma-min "${SIGMA_MIN}" --sigma-max "${SIGMA_MAX}" --second-order \
  --alpha 0.03 \
  --smooth-weight "${SMOOTH_WEIGHT}" \
  --out-root "${OUT_TRAIN}"

CKPT_DIR="${OUT_TRAIN}/${SCENE}"
# Pick the latest checkpoint by filename order; fall back to ep_0001 if missing
if compgen -G "${CKPT_DIR}/ckpt_ep_*.pth" > /dev/null; then
  CKPT=$(ls -1 "${CKPT_DIR}"/ckpt_ep_*.pth | sort | tail -n 1)
else
  CKPT="${CKPT_DIR}/ckpt_ep_0001.pth"
fi
echo "[infer] using ckpt=${CKPT}"

# Inference A: GT+noise
OUT_GT="global_diffusion_map/refine/work_dirs/infer_${SCENE}_gtnoise"
mkdir -p "${OUT_GT}"
python -u global_diffusion_map/refine/infer_refine.py \
  --agg-pred-root "${AGG_PRED_ROOT}" \
  --static-root   "${STATIC_ROOT}" \
  --rendered-root "${RENDERED_ROOT}" \
  --scene "${SCENE}" \
  --ckpt  "${CKPT}" \
  --start gt_noise --alpha 0.6 \
  --steps "${STEPS}" --sigma-min "${SIGMA_MIN}" --sigma-max "${SIGMA_MAX}" --second-order \
  --thr 0.2 \
  --out-root "${OUT_GT}"

# Inference B: Proposal (no noise)
OUT_PR="global_diffusion_map/refine/work_dirs/infer_${SCENE}_proposal"
mkdir -p "${OUT_PR}"
python -u global_diffusion_map/refine/infer_refine.py \
  --agg-pred-root "${AGG_PRED_ROOT}" \
  --static-root   "${STATIC_ROOT}" \
  --rendered-root "${RENDERED_ROOT}" \
  --scene "${SCENE}" \
  --ckpt  "${CKPT}" \
  --start proposal \
  --steps "${STEPS}" --sigma-min "${SIGMA_MIN}" --sigma-max "${SIGMA_MAX}" --second-order \
  --thr 0.2 \
  --out-root "${OUT_PR}"

echo "[ok] one-scene overfit + two-mode inference finished: ${SCENE}"
