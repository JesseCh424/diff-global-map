#!/usr/bin/env bash
set -euo pipefail

CONFIG=plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py
PRED_PKL=work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl
STATIC_OUT=work_dirs/static_gt_vector/av2_oldsplit
SCENE="$1"

python tools/tracking/crop_static_gt_from_predictions.py \
  "$CONFIG" \
  --pred-pkl "$PRED_PKL" \
  --scene-id "$SCENE" \
  --out-dir "$STATIC_OUT"

python tools/visualization/viz_scene_compare.py \
  --aggregated-pred work_dirs/aggregated_scene_vectors/av2_oldsplit \
  --aggregated-gt work_dirs/agg_gt_vector/av2_oldsplit \
  --static-gt "$STATIC_OUT" \
  --out-root viz/av2_old \
  --scene-list /workspace/mrt/maptracker/tmp_scene_$SCENE.txt \
  --bounds-pkl work_dirs/aggregated_scene_vectors/av2_oldsplit/{scene}.pkl work_dirs/agg_gt_vector/av2_oldsplit/{scene}.pkl \
  --dpi 40 --simplify 0.5 --line-opacity 0.75
