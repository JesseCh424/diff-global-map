#!/usr/bin/env bash
set -euo pipefail

# Generate 08 agg semantic PNGs for scenes under rendered_gt val folder.
# This uses the same method as semantic 08 (continuity-aware overlap from 06).
#
# Defaults (can be overridden via env):
#   CONNECT_M=4.0    # 08 post-connect circle radius (meters)
#   Use inner ring, prox 1.0 m, density k=2/min=5, close 1, boost 0.4,
#   overlap from 06, no extra 08 dilate, thin 1 px.

CONFIG=${CONFIG:-maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py}
SUB=${SUB:-maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/submission_vector.json}
POS=${POS:-maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl}
STATIC_VAL=${STATIC_VAL:-maptracker/work_dirs/static_gt_vector/av2_oldsplit/val}
# Optional subset: SCENE_FILE points to a file with one scene per line
SCENE_FILE=${SCENE_FILE:-}
CONNECT_M=${CONNECT_M:-2.0}
OUT=${OUT:-maptracker/work_dirs/rendered_gt/av2_oldsplit/val}
VIZ_ROOT=${VIZ_ROOT:-maptracker/viz/av2_old}
OVERWRITE=${OVERWRITE:-1}

THK06=${THK06:-2}
SPLAT08_M=${SPLAT08_M:-0.5}
SPLAT08_BLOCK=${SPLAT08_BLOCK:-1}
MIN_VOTES=${MIN_VOTES:-1}
MIN_AREA=${MIN_AREA:-8}
DENOISE_FLAG=${DENOISE_FLAG:---no-denoise}

if [ -n "$SCENE_FILE" ] && [ -f "$SCENE_FILE" ]; then
  mapfile -t SCENES < "$SCENE_FILE"
else
  mapfile -t SCENES < <(find "$STATIC_VAL" -maxdepth 1 -name '*.pkl' -printf '%f\n' | sed 's/\.pkl$//' | sort)
fi
total=${#SCENES[@]}
echo "[info] Generating 08 for ${total} scenes into ${OUT}" >&2

SCENE_TMP=$(mktemp)
> "$SCENE_TMP"
skipped=0
for s in "${SCENES[@]}"; do
  if [ "$OVERWRITE" = "1" ]; then
    echo "$s" >> "$SCENE_TMP"
  else
    if [ -f "$OUT/$s/08_agg_semantic.png" ]; then
      echo "[skip] $s already has 08" >&2
      skipped=$((skipped+1))
    else
      echo "$s" >> "$SCENE_TMP"
    fi
  fi
done

if [ -s "$SCENE_TMP" ]; then
  echo "CMD: aggregate_semantic_scene_sparse.py --scene-list $SCENE_TMP --png --match-05 --min-votes $MIN_VOTES --min-area $MIN_AREA $DENOISE_FLAG \
    --thickness06-px $THK06 \
    --overlap-ring-mode inner --overlap-edge-m 0.6 \
    --overlap-prox-m 1.0 --overlap-density-k 2 --overlap-density-min 5 \
    --overlap-close-px 1 --overlap-boost-m 0.4 \
    --overlap-08-from-06 --overlap08-close-px 1 --overlap08-dilate-px 0 \
    --overlap08-connect-m $CONNECT_M --overlap08-thin-px 1 \
    --splat08-m $SPLAT08_M --splat08-block $SPLAT08_BLOCK" >&2
  python -u maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
    "$CONFIG" --submission-json "$SUB" --pos-pkl "$POS" \
    --out-dir "$OUT" --scene-list "$SCENE_TMP" \
    --png --match-05 --min-votes "$MIN_VOTES" --min-area "$MIN_AREA" $DENOISE_FLAG \
    --thickness06-px "$THK06" \
    --overlap-ring-mode inner --overlap-edge-m 0.6 \
    --overlap-prox-m 1.0 --overlap-density-k 2 --overlap-density-min 5 \
    --overlap-close-px 1 --overlap-boost-m 0.4 \
    --overlap-08-from-06 --overlap08-close-px 1 --overlap08-dilate-px 0 \
    --overlap08-connect-m "$CONNECT_M" --overlap08-thin-px 1 \
    --splat08-m "$SPLAT08_M" --splat08-block "$SPLAT08_BLOCK" \
    --viz-root "$VIZ_ROOT" || true
else
  echo "[info] All ${total} scenes already have 08 (skipped: ${skipped})." >&2
fi

echo "[done] 08 generated under ${OUT}" >&2
