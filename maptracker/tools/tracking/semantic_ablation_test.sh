#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash maptracker/tools/tracking/semantic_ablation_test.sh \
#     <CONFIG> <SUBMISSION_JSON> <POS_PKL> <OUT_DIR> <SCENE_1> [SCENE_2 ...]
#
# Env knobs:
#   THICKNESS   (default: 3)  # thicker variant for 07
#   BASE_THK    (default: 2)  # base thickness for 06/08
#   MIN_VOTES   (default: 1)
#   MIN_AREA    (default: 8)
#   NO_DENOISE  (default: 1 -> pass --no-denoise)
#
# Produces per-scene:
#   - 07_agg_semantic.png (thicker votes, canvas matched to 05)
#   (08 variant removed per request)

CONFIG=${1:?"need CONFIG"}
SUB=${2:?"need SUBMISSION_JSON"}
POS=${3:?"need POS_PKL"}
OUT=${4:?"need OUT_DIR"}
shift 4 || true

if [ $# -eq 0 ]; then
  echo "[warn] No scenes provided; exiting." >&2
  exit 1
fi

THICKNESS=${THICKNESS:-3}
BASE_THK=${BASE_THK:-2}
MIN_VOTES=${MIN_VOTES:-1}
MIN_AREA=${MIN_AREA:-8}
NO_DENOISE=${NO_DENOISE:-1}

ND_FLAG=""
if [ "${NO_DENOISE}" = "1" ]; then
  ND_FLAG="--no-denoise"
fi

for SCENE in "$@"; do
  # 06: base (match-05, base thickness)
  echo "[06] base thickness=${BASE_THK}, match-05: ${SCENE}"
  python maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
    "${CONFIG}" \
    --submission-json "${SUB}" \
    --pos-pkl "${POS}" \
    --out-dir "${OUT}" \
    --scenes "${SCENE}" \
    --png --match-05 \
    --min-votes "${MIN_VOTES}" --min-area "${MIN_AREA}" ${ND_FLAG} \
    --thickness-px "${BASE_THK}"

  # 07: thicker (match-05, THICKNESS); restore 06 afterward
  echo "[07] thickness=${THICKNESS}, match-05: ${SCENE}"
  python maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
    "${CONFIG}" \
    --submission-json "${SUB}" \
    --pos-pkl "${POS}" \
    --out-dir "${OUT}" \
    --scenes "${SCENE}" \
    --png --match-05 \
    --min-votes "${MIN_VOTES}" --min-area "${MIN_AREA}" ${ND_FLAG} \
    --thickness-px "${THICKNESS}"
  if [ -f "${OUT}/${SCENE}/06_agg_semantic.png" ]; then
    cp -f "${OUT}/${SCENE}/06_agg_semantic.png" "${OUT}/${SCENE}/07_agg_semantic.png"
  else
    echo "[warn] 06_agg_semantic.png missing for ${SCENE} (07 copy skipped)" >&2
  fi
  # restore 06 (base)
  python maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
    "${CONFIG}" \
    --submission-json "${SUB}" \
    --pos-pkl "${POS}" \
    --out-dir "${OUT}" \
    --scenes "${SCENE}" \
    --png --match-05 \
    --min-votes "${MIN_VOTES}" --min-area "${MIN_AREA}" ${ND_FLAG} \
    --thickness-px "${BASE_THK}"

done

echo "[done] Wrote 07_ agg semantic PNGs under ${OUT}/<scene>/"
