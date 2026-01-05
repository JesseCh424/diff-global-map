#!/usr/bin/env bash
set -euo pipefail

# Generate 08 agg semantic PNGs for all scenes listed under
# static GT val split. Saves under maptracker/work_dirs/semantic/<scene>/.
#
# Preset (08 with 2.0m connect, reusing 06 overlap):
#   --png --match-05 --min-votes 1 --min-area 8 --no-denoise \
#   --thickness06-px 2 \
#   --overlap-ring-mode inner --overlap-edge-m 0.6 \
#   --overlap-prox-m 1.0 --overlap-density-k 2 --overlap-density-min 5 \
#   --overlap-close-px 1 --overlap-boost-m 0.4 \
#   --overlap-08-from-06 --overlap08-close-px 1 --overlap08-dilate-px 0 \
#   --overlap08-connect-m 2.0 --overlap08-thin-px 1 \
#   --splat08-m 0.5 --splat08-block 1

CONFIG=${CONFIG:-maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py}
SUB=${SUB:-maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/submission_vector.json}
POS=${POS:-maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl}
STATIC_VAL=${STATIC_VAL:-maptracker/work_dirs/static_gt_vector/av2_oldsplit/val}
OUT=${OUT:-maptracker/work_dirs/semantic}
VIZ_ROOT=${VIZ_ROOT:-maptracker/viz/av2_old/val}

THK06=${THK06:-2}
SPLAT08_M=${SPLAT08_M:-0.5}
SPLAT08_BLOCK=${SPLAT08_BLOCK:-1}
MIN_VOTES=${MIN_VOTES:-1}
MIN_AREA=${MIN_AREA:-8}
DENOISE_FLAG=${DENOISE_FLAG:---no-denoise}
OVERWRITE=${OVERWRITE:-1}

scenes_list=("$STATIC_VAL"/*.pkl)
total=${#scenes_list[@]}
echo "[info] Generating semantic 08 for ${total} scenes from ${STATIC_VAL} into ${OUT}" >&2

# Build a scene list file to pass in one invocation (amortize JSON parse)
SCENE_FILE=$(mktemp)
for p in "${scenes_list[@]}"; do
  s=$(basename "$p" .pkl)
  if [ "$OVERWRITE" = "1" ]; then
    echo "$s" >> "$SCENE_FILE"
  else
    if [ -f "$OUT/$s/08_agg_semantic.png" ]; then
      echo "[skip] $s already has 08_agg_semantic.png" >&2
    else
      echo "$s" >> "$SCENE_FILE"
    fi
  fi
done

if [ -s "$SCENE_FILE" ]; then
  echo "CMD: aggregate_semantic_scene_sparse.py --scene-list $SCENE_FILE --png --match-05 --min-votes $MIN_VOTES --min-area $MIN_AREA $DENOISE_FLAG \
    --thickness06-px $THK06 \
    --overlap-ring-mode inner --overlap-edge-m 0.6 \
    --overlap-prox-m 1.0 --overlap-density-k 2 --overlap-density-min 5 \
    --overlap-close-px 1 --overlap-boost-m 0.4 \
    --overlap-08-from-06 --overlap08-close-px 1 --overlap08-dilate-px 0 \
    --overlap08-connect-m 2.0 --overlap08-thin-px 1 \
    --splat08-m $SPLAT08_M --splat08-block $SPLAT08_BLOCK" >&2
  python -u maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
    "$CONFIG" --submission-json "$SUB" --pos-pkl "$POS" \
    --out-dir "$OUT" --scene-list "$SCENE_FILE" \
    --png --match-05 --min-votes "$MIN_VOTES" --min-area "$MIN_AREA" $DENOISE_FLAG \
    --thickness06-px "$THK06" \
    --overlap-ring-mode inner --overlap-edge-m 0.6 \
    --overlap-prox-m 1.0 --overlap-density-k 2 --overlap-density-min 5 \
    --overlap-close-px 1 --overlap-boost-m 0.4 \
    --overlap-08-from-06 --overlap08-close-px 1 --overlap08-dilate-px 0 \
    --overlap08-connect-m 2.0 --overlap08-thin-px 0 \
    --splat08-m "$SPLAT08_M" --splat08-block "$SPLAT08_BLOCK" \
    --bounds-pkl maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid/{scene}.pkl maptracker/work_dirs/static_gt_vector/av2_oldsplit/val/{scene}.pkl \
    --viz-root "$VIZ_ROOT" || true
else
  echo "[info] All scenes already have 08_agg_semantic.png" >&2
fi

echo "[done] Semantic 08 generated under ${OUT}" >&2
