#!/usr/bin/env bash
set -euo pipefail

# Single-scene inference launcher for two inputs:
#   1) GT+Noise (training-like)  2) Real Proposal (no noise)
# Produces: overlays, matched-class view, points overlay, and points_*.txt

# Defaults (override via env)
: "${SCENE:=0b5142c1-420b-3fea-9e98-b87327ae22c6}"
: "${STATIC_ROOT:=maptracker/work_dirs/static_gt_vector/av2_oldsplit/val}"
: "${RENDERED_ROOT:=maptracker/work_dirs/rendered_gt/av2_oldsplit/val}"
: "${AGG_PRED_ROOT:=maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid}"
: "${OUT_ROOT_INFER:=global_diffusion_map/refine/work_dirs/infer_dynamic_scratch}"

# Denoising strengths / filtering
: "${ALPHA_NOISE:=0.02}"
: "${THR:=0.7}"
: "${NMS:=1.5}"

# Checkpoint: explicit or auto-pick latest under training scratch dir
: "${CKPT:=}"
if [[ -z "${CKPT}" ]]; then
  CAND_DIR="global_diffusion_map/refine/work_dirs/train_dynamic_overfit_scratch_1000/${SCENE}"
  if compgen -G "${CAND_DIR}/ckpt_ep_*.pth" > /dev/null; then
    CKPT=$(ls -t "${CAND_DIR}"/ckpt_ep_*.pth | head -n1)
  else
    echo "[ERR] CKPT not set and no ckpt_ep_*.pth found under ${CAND_DIR}" >&2
    exit 1
  fi
fi

echo "Scene=${SCENE}  CKPT=${CKPT}"

# 1) GT+Noise
OUT_GT="${OUT_ROOT_INFER}_gtnoise"
python -u global_diffusion_map/refine/infer_refine.py \
  --agg-pred-root "${AGG_PRED_ROOT}" \
  --static-root   "${STATIC_ROOT}" \
  --rendered-root "${RENDERED_ROOT}" \
  --scene "${SCENE}" \
  --ckpt  "${CKPT}" \
  --start gt_noise --alpha "${ALPHA_NOISE}" \
  --thr "${THR}" --nms-meters "${NMS}" \
  --out-root "${OUT_GT}"

# 2) Proposal (no noise)
OUT_PROP="${OUT_ROOT_INFER}_prop"
python -u global_diffusion_map/refine/infer_refine.py \
  --agg-pred-root "${AGG_PRED_ROOT}" \
  --static-root   "${STATIC_ROOT}" \
  --rendered-root "${RENDERED_ROOT}" \
  --scene "${SCENE}" \
  --ckpt  "${CKPT}" \
  --start proposal --alpha 0.0 \
  --thr "${THR}" --nms-meters "${NMS}" \
  --out-root "${OUT_PROP}"

echo "[OK] Inference done. Outputs:"
echo "  GT+Noise: ${OUT_GT}/${SCENE}"
echo "  Proposal: ${OUT_PROP}/${SCENE}"

