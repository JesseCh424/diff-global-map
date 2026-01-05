#!/usr/bin/env bash
set -euo pipefail

# Batch prepare per-scene bounds for AV2 old split (train + valid)
# 1) Export aggregated predictions per scene
# 2) Crop static GT with union ROI (writes canonical bounds into each static pkl)
# 3) (Optional) Export aggregated GT per scene for 04 viz
# 4) Render 10 aligned to 05/08
#
# Edit the variables below to your environment, or export them before running.

CFG=${CFG:-maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py}
MAP_INFO=${MAP_INFO:-datasets/av2/av2_map_infos_val.pkl}

# pos_predictions_5.pkl for val/train
POS_VAL=${POS_VAL:-maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/pos_predictions_5.pkl}
POS_TRAIN=${POS_TRAIN:-maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/pos_predictions_5.pkl}

AGG_DIR=${AGG_DIR:-maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit}
STATIC_DIR=${STATIC_DIR:-maptracker/work_dirs/static_gt_vector/av2_oldsplit}
AGG_GT_DIR=${AGG_GT_DIR:-maptracker/work_dirs/agg_gt_vector/av2_oldsplit}
RENDER10_DIR=${RENDER10_DIR:-maptracker/work_dirs/rendered_gt/av2_oldsplit}

function do_split() {
  local SPLIT=$1
  local POS=$2
  echo "[split=${SPLIT}] export aggregated predictions"
  python -u maptracker/tools/tracking/export_scene_grouped.py \
    "${CFG}" --pred-pkl "${POS}" \
    --out-dir "${AGG_DIR}/${SPLIT}" --simplify 0.5 --overwrite

  echo "[split=${SPLIT}] crop static GT with union ROI (writes bounds)"
  SCENES=$(ls "${AGG_DIR}/${SPLIT}"/*.pkl | xargs -n1 basename | sed 's/.pkl$//')
  for s in ${SCENES}; do
    python -u maptracker/tools/tracking/crop_static_gt_from_predictions.py \
      "${CFG}" --pred-pkl "${POS}" --scene-id "${s}" \
      --out-dir "${STATIC_DIR}/${SPLIT}" --aggregated-pred-dir "${AGG_DIR}/${SPLIT}" \
      --map-info "${MAP_INFO}" || true
  done

  echo "[split=${SPLIT}] (optional) export aggregated GT per scene"
  if [[ -f datasets/av2/av2_map_infos_val_gt_tracks.pkl || -f maptracker/datasets/av2/av2_map_infos_val_gt_tracks.pkl ]]; then
    GT_TRACKS=${GT_TRACKS:-datasets/av2/av2_map_infos_val_gt_tracks.pkl}
    python -u maptracker/tools/tracking/export_gt_scene_grouped.py \
      "${CFG}" --gt-tracks "${GT_TRACKS}" \
      --out-dir "${AGG_GT_DIR}/${SPLIT}" --simplify 0.5 --overwrite || true
  fi

  echo "[split=${SPLIT}] render 10 aligned to 05/08"
  python -u maptracker/tools/tracking/render_gt_to_10.py \
    --static-root     "${STATIC_DIR}/${SPLIT}" \
    --aggregated-pred "${AGG_DIR}/${SPLIT}" \
    --aggregated-gt   "${AGG_GT_DIR}/${SPLIT}" \
    --semantic-root   "maptracker/work_dirs/semantic/${SPLIT}" \
    --out-root        "${RENDER10_DIR}/${SPLIT}" \
    --scenes ${SCENES} || true
}

echo "[prep] VALID split"
do_split valid "${POS_VAL}"

echo "[prep] TRAIN split"
do_split train "${POS_TRAIN}"

echo "[ok] bounds prepared under ${STATIC_DIR}/{train|valid}; 10 renders under ${RENDER10_DIR}/{train|valid}"
