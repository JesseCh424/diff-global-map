#!/usr/bin/env bash
set -euo pipefail

# Batch pipeline for AV2 old split (TRAIN):
# 1) Wait for train inference submission_vector.json
# 2) Prepare pos_predictions_5.pkl (global IDs)
# 3) Aggregate predictions per scene under aggregated_scene_vectors/av2_oldsplit/train
# 4) Crop static GT per scene into static_gt_vector/av2_oldsplit/train

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MAPTRACKER="$ROOT/maptracker"

CONFIG=${CONFIG:-plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py}
WORK_TRAIN=${WORK_TRAIN:-$MAPTRACKER/work_dirs/maptracker_av2_oldsplit_train_infer}
PRED_JSON=${PRED_JSON:-$WORK_TRAIN/submission_vector.json}
POS_PKL=${POS_PKL:-$WORK_TRAIN/pos_predictions_5.pkl}

AGG_TRAIN=${AGG_TRAIN:-$MAPTRACKER/work_dirs/aggregated_scene_vectors/av2_oldsplit/train}
STATIC_TRAIN=${STATIC_TRAIN:-$MAPTRACKER/work_dirs/static_gt_vector/av2_oldsplit/train}

mkdir -p "$AGG_TRAIN" "$STATIC_TRAIN"

LOG=${LOG:-$STATIC_TRAIN/batch_crop_train.log}
echo "[info] Batch train static GT crop started" > "$LOG" 2>&1
echo "CONFIG=$CONFIG" >> "$LOG"
echo "WORK_TRAIN=$WORK_TRAIN" >> "$LOG"

# Build a temporary config in the SAME directory as the original (keeps _base_ relative paths valid)
SRC_CFG_PATH="$MAPTRACKER/$CONFIG"
CFG_DIR="$(dirname "$SRC_CFG_PATH")"
TMP_CFG_BASENAME="tmp_train_$(basename "$CONFIG")"
TMP_CFG="$CFG_DIR/$TMP_CFG_BASENAME"
cp -f "$SRC_CFG_PATH" "$TMP_CFG"
# Swap val -> train ann_file occurrences inside the tmp config
sed -i -E "s#av2_map_infos_val\.pkl#av2_map_infos_train.pkl#g" "$TMP_CFG"
echo "[info] Using tmp config: $TMP_CFG" >> "$LOG"

# Wait for submission_vector.json
if [ ! -f "$PRED_JSON" ]; then
  echo "[wait] Waiting for $PRED_JSON ..." >> "$LOG"
  while [ ! -f "$PRED_JSON" ]; do sleep 30; done
fi
echo "[ok] Found $PRED_JSON" >> "$LOG"

# Step 1: Global ID assignment (pos_predictions_5.pkl)
if [ ! -f "$POS_PKL" ]; then
  echo "[run] prepare_pred_tracks.py -> $POS_PKL" >> "$LOG"
  ( cd "$MAPTRACKER" && \
    python tools/tracking/prepare_pred_tracks.py \
      "$TMP_CFG" --result_path "$PRED_JSON" --cons_frames 5 --thr 0.4 ) >> "$LOG" 2>&1
else
  echo "[skip] $POS_PKL exists" >> "$LOG"
fi

# Step 2: Aggregate predictions per scene (train)
echo "[run] export_scene_grouped.py -> $AGG_TRAIN" >> "$LOG"
( cd "$MAPTRACKER" && \
  python tools/tracking/export_scene_grouped.py \
    "$TMP_CFG" --pred-pkl "$POS_PKL" --out-dir "$AGG_TRAIN" --simplify 0.5 --overwrite ) >> "$LOG" 2>&1

# Step 3: Crop static GT per scene using union ROI (per-frame ROIs + aggregated vectors)
echo "[run] crop_static_gt_from_predictions.py -> $STATIC_TRAIN" >> "$LOG"
scenes=$(ls -1 "$AGG_TRAIN"/*.pkl | xargs -n1 basename | sed 's/.pkl$//')
total=0
for scene in $scenes; do
  echo "[crop] $scene" >> "$LOG"
  ( cd "$MAPTRACKER" && \
    python tools/tracking/crop_static_gt_from_predictions.py \
      "$TMP_CFG" --pred-pkl "$POS_PKL" --scene-id "$scene" \
      --out-dir "$STATIC_TRAIN" --aggregated-pred-dir "$AGG_TRAIN" --map-info ./datasets/av2/av2_map_infos_train.pkl ) >> "$LOG" 2>&1 || true
  total=$((total+1))
done
echo "[done] Cropped $total scenes into $STATIC_TRAIN" >> "$LOG"
