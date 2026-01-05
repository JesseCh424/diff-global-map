#!/usr/bin/env bash
set -euo pipefail

# Fix incorrectly named scene folders (with ripgrep line numbers like '10:scene')
# and generate 01/05/06/07 under maptracker/work_dirs/semantic/<scene>/.

ROOT="maptracker/work_dirs/semantic"
PRED_DIR="maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit"
GT_DIR="maptracker/work_dirs/agg_gt_vector/av2_oldsplit"
STATIC_DIR="maptracker/work_dirs/static_gt_vector/av2_oldsplit"
VIZ_ROOT="maptracker/viz/av2_old"

CONFIG="maptracker/plugin/configs/maptracker/av2_oldsplit/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune.py"
SUB="maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/submission_vector.json"
POS="maptracker/work_dirs/maptracker_av2_oldsplit_5frame_span10_stage3_joint_finetune/pos_predictions_5.pkl"

BASE_THK=${BASE_THK:-2}   # thickness for 06
THICK_07=${THICK_07:-5}   # thickness for 07

fix_names() {
  shopt -s nullglob
  for d in "${ROOT}"/*:*; do
    bn=$(basename "$d")
    scene=${bn#*:}
    target="${ROOT}/${scene}"
    if [ "$d" != "$target" ]; then
      echo "[fix] rename $bn -> $scene"
      if [ -e "$target" ]; then
        rmdir "$d" 2>/dev/null || true
      else
        mv "$d" "$target"
      fi
    fi
  done
}

gen_scene() {
  local scene="$1"
  local out_dir="${ROOT}/${scene}"
  mkdir -p "$out_dir"

  # Ensure 01/05 exist; if not, render and copy
  if [ ! -f "${VIZ_ROOT}/${scene}/01_agg_pred_direct.png" ] || [ ! -f "${VIZ_ROOT}/${scene}/05_static_gt.png" ]; then
    printf "%s\n" "$scene" > /tmp/one_scene.txt
    python maptracker/tools/visualization/viz_scene_compare.py \
      --aggregated-pred "$PRED_DIR" \
      --aggregated-gt   "$GT_DIR" \
      --static-gt       "$STATIC_DIR" \
      --out-root        "$VIZ_ROOT" \
      --scene-list      /tmp/one_scene.txt \
      --bounds-pkl      ${PRED_DIR}/{scene}.pkl ${GT_DIR}/{scene}.pkl \
      --dpi 60 || true
  fi
  cp -f "${VIZ_ROOT}/${scene}/01_agg_pred_direct.png" "${out_dir}/01_agg_pred_direct.png" || true
  cp -f "${VIZ_ROOT}/${scene}/05_static_gt.png"       "${out_dir}/05_static_gt.png" || true

  # 06 base (match-05)
  python maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
    "$CONFIG" --submission-json "$SUB" --pos-pkl "$POS" \
    --out-dir "$ROOT" --scenes "$scene" \
    --png --match-05 --min-votes 1 --min-area 8 --no-denoise \
    --thickness-px "$BASE_THK"

  # 07 thicker (match-05), then restore 06
  python maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
    "$CONFIG" --submission-json "$SUB" --pos-pkl "$POS" \
    --out-dir "$ROOT" --scenes "$scene" \
    --png --match-05 --min-votes 1 --min-area 8 --no-denoise \
    --thickness-px "$THICK_07"
  cp -f "${out_dir}/06_agg_semantic.png" "${out_dir}/07_agg_semantic.png" || true
  # restore 06
  python maptracker/tools/tracking/aggregate_semantic_scene_sparse.py \
    "$CONFIG" --submission-json "$SUB" --pos-pkl "$POS" \
    --out-dir "$ROOT" --scenes "$scene" \
    --png --match-05 --min-votes 1 --min-area 8 --no-denoise \
    --thickness-px "$BASE_THK"

  echo "[ok] $scene"
}

main() {
  fix_names
  # Build scene list: use existing folders under ROOT after fixing names
  SCENES=()
  while IFS= read -r d; do
    bn=$(basename "$d")
    SCENES+=("$bn")
  done < <(find "$ROOT" -maxdepth 1 -mindepth 1 -type d ! -name "*:*" -printf '%p\n')

  if [ ${#SCENES[@]} -eq 0 ]; then
    echo "[warn] No scene folders found under ${ROOT}." >&2
    exit 0
  fi

  for s in "${SCENES[@]}"; do
    gen_scene "$s"
  done
}

main "$@"

