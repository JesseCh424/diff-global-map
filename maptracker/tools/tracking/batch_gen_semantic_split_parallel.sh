#!/usr/bin/env bash
set -euo pipefail

# Parallel batch generator for semantic 08 with correct alignment & cyan overlap.
# Usage: SPLIT=train|valid WORKERS=8 nohup bash maptracker/tools/tracking/batch_gen_semantic_split_parallel.sh > maptracker/work_dirs/semantic/batch_${SPLIT}.log 2>&1 &

SPLIT=${SPLIT:-valid}
WORKERS=${WORKERS:-8}

if [ "$SPLIT" = "valid" ]; then
  STATIC_ROOT=maptracker/work_dirs/static_gt_vector/av2_oldsplit/val
  OUT_ROOT=maptracker/work_dirs/semantic/valid
  # Use canonical 05 location for val
  VIZ_ROOT=maptracker/viz/av2_old/val
  BOUNDS_TPL_A=maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid/{scene}.pkl
  BOUNDS_TPL_B=maptracker/work_dirs/static_gt_vector/av2_oldsplit/val/{scene}.pkl
  CONFIG=${CONFIG:-maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py}
  SUB=${SUB:-maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/submission_vector.json}
  POS=${POS:-maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl}
elif [ "$SPLIT" = "train" ]; then
  STATIC_ROOT=maptracker/work_dirs/static_gt_vector/av2_oldsplit/train
  OUT_ROOT=maptracker/work_dirs/semantic/train
  # Use canonical 05 location for train
  VIZ_ROOT=maptracker/viz/av2_old/train
  BOUNDS_TPL_A=maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/train/{scene}.pkl
  BOUNDS_TPL_B=maptracker/work_dirs/static_gt_vector/av2_oldsplit/train/{scene}.pkl
  CONFIG=${CONFIG:-maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py}
  SUB=${SUB:-maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/submission_vector.json}
  POS=${POS:-maptracker/work_dirs/maptracker_av2_oldsplit_train_infer/pos_predictions_5.pkl}
else
  echo "Unknown SPLIT=$SPLIT" >&2; exit 1
fi

THK06=${THK06:-2}
SPLAT08_M=${SPLAT08_M:-0.4}
MIN_VOTES=${MIN_VOTES:-1}
MIN_AREA=${MIN_AREA:-8}
DENOISE_FLAG=${DENOISE_FLAG:---no-denoise}

mkdir -p "$OUT_ROOT"
SCENES_FILE=$(mktemp)
ls -1 "$STATIC_ROOT"/*.pkl | xargs -n1 basename | sed 's/.pkl$//' > "$SCENES_FILE"
TOTAL=$(wc -l < "$SCENES_FILE" | tr -dc '0-9')
echo "[start] split=$SPLIT total_scenes=$TOTAL workers=$WORKERS" >&2

# Split into shards
LINES=$TOTAL
PER=$(( (LINES + WORKERS - 1) / WORKERS ))
split -d -l "$PER" "$SCENES_FILE" "$SCENES_FILE.shard."

idx=0
for shard in "$SCENES_FILE".shard.*; do
  LOG=maptracker/work_dirs/semantic/batch_${SPLIT}_shard_${idx}.log
  echo "[shard $idx] scenes=$(wc -l < "$shard") -> $LOG" >&2
  (
    python -u maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
      "$CONFIG" --submission-json "$SUB" --pos-pkl "$POS" \
      --out-dir "$OUT_ROOT" --scene-list "$shard" \
      --png --match-05 --min-votes "$MIN_VOTES" --min-area "$MIN_AREA" $DENOISE_FLAG \
      --thickness06-px "$THK06" \
      --overlap-ring-mode inner --overlap-edge-m 0.6 \
      --overlap-prox-m 1.0 --overlap-density-k 2 --overlap-density-min 5 \
      --overlap-close-px 1 --overlap-boost-m 0.4 \
      --overlap-08-from-06 --overlap08-close-px 1 --overlap08-dilate-px 0 \
      --overlap08-connect-m 2.0 --overlap08-thin-px 0 \
      --splat08-m "$SPLAT08_M" --splat08-block 1 \
      --bounds-pkl "$BOUNDS_TPL_A" "$BOUNDS_TPL_B" \
      --viz-root "$VIZ_ROOT"
  ) > "$LOG" 2>&1 &
  idx=$((idx+1))
done

echo "[launched] $idx shards. Logs: maptracker/work_dirs/semantic/batch_${SPLIT}_shard_*.log" >&2
