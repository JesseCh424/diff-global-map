#!/usr/bin/env bash
set -euo pipefail

# Batch regenerate 06/07/08 semantic previews for all scenes under a root.
# Uses the latest aggregate_semantic_scene_sparse.py (overlap-aware coloring).

CONFIG=${CONFIG:-maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py}
SUB=${SUB:-maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/submission_vector.json}
POS=${POS:-maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl}
ROOT=${ROOT:-maptracker/work_dirs/semantic}

THK06=${THK06:-2}
THK07=${THK07:-5}
SPLAT08_M=${SPLAT08_M:-0.5}
SPLAT08_BLOCK=${SPLAT08_BLOCK:-1}
MIN_VOTES=${MIN_VOTES:-1}
MIN_AREA=${MIN_AREA:-8}
DENOISE_FLAG=${DENOISE_FLAG:---no-denoise}

echo "[info] Regenerating 06/07/08 for scenes under ${ROOT}" >&2
mapfile -t SCENES < <(find "$ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
total=${#SCENES[@]}
if [ "$total" -eq 0 ]; then
  echo "[warn] No scene folders found under ${ROOT}" >&2
  exit 0
fi

idx=0
for s in "${SCENES[@]}"; do
  ((idx++))
  echo "[run] ($idx/$total) scene=$s" >&2
  python maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
    "$CONFIG" --submission-json "$SUB" --pos-pkl "$POS" \
    --out-dir "$ROOT" --scenes "$s" \
    --png --match-05 --min-votes "$MIN_VOTES" --min-area "$MIN_AREA" $DENOISE_FLAG \
    --thickness06-px "$THK06" --thickness07-px "$THK07" \
    --splat08-m "$SPLAT08_M" --splat08-block "$SPLAT08_BLOCK" || true
done

echo "[done] Regenerated 06/07/08 for ${total} scenes under ${ROOT}" >&2

