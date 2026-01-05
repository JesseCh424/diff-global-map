#!/usr/bin/env bash
set -euo pipefail

# Quick condition alignment check for a scene.
# Draws a letterboxed 10_render_gt.png and overlays vectors normalized by static pkl bounds.
# Usage:
#   SCENE=<scene_id> \
#   STATIC_ROOT=maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
#   RENDERED_ROOT=maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
#   AGG_PRED_ROOT=maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid \
#   bash -x maptracker/tools/visualization/viz_check_condition_alignment.sh

: "${SCENE:?SCENE is required}"
: "${STATIC_ROOT:?STATIC_ROOT is required}"
: "${RENDERED_ROOT:?RENDERED_ROOT is required}"
: "${AGG_PRED_ROOT:?AGG_PRED_ROOT is required}"

OUT="global_diffusion_map/refine/work_dirs/align_check/${SCENE}"
mkdir -p "${OUT}"

python -u global_diffusion_map/refine/infer_refine.py \
  --agg-pred-root "${AGG_PRED_ROOT}" \
  --static-root   "${STATIC_ROOT}" \
  --rendered-root "${RENDERED_ROOT}" \
  --scene "${SCENE}" \
  --ckpt global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth \
  --start proposal --steps 1 --sigma 0.5 \
  --thr 0.3 --steps-cls-color --steps-match-cls \
  --out-root "${OUT}" --save-steps-dir "${OUT}/steps" --overlay-condition

echo "[ok] alignment check saved under ${OUT} (condition_letterbox.png + input overlays)"
