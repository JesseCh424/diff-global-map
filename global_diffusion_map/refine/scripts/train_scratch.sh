#!/usr/bin/env bash
set -euo pipefail

# Single-scene dynamic overfit training launcher (scratch, with defaults).
# All parameters are set as defaults here; override via env vars if needed.

# Defaults (override via env, e.g., SCENE=... EPOCHS=200 bash -x ...)
: "${SCENE:=0b5142c1-420b-3fea-9e98-b87327ae22c6}"
: "${EPOCHS:=1000}"
: "${STATIC_ROOT:=maptracker/work_dirs/static_gt_vector/av2_oldsplit/val}"
: "${RENDERED_ROOT:=maptracker/work_dirs/rendered_gt/av2_oldsplit/val}"
: "${AGG_PRED_ROOT:=maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid}"

# Training knobs
: "${HYBRID:=1}"               # 1=use hybrid batch (easy+dynamic)
: "${ALPHA:=0.03}"             # SDEdit strength for training
: "${CLS_WEIGHT:=2.0}"
: "${SEM_WEIGHT:=1.0}"

# Geometric regularizers
# pure denoising: no geometric/background regularizers in default

# Diffusion sampler (multi-step) defaults (must be defined before building cmd)
: "${SAMPLER_STEPS:=1}"
: "${SIGMA_MIN:=0.01}"
: "${SIGMA_MAX:=1.0}"
: "${SAMPLER_SCHEDULE:=log}"
: "${SAMPLER_MODE:=sdedit}"

# Output dir
: "${OUT_ROOT:=global_diffusion_map/refine/work_dirs/train_dynamic_overfit_scratch_1000}"

SCENE_DIR="${OUT_ROOT}/${SCENE}"
mkdir -p "${SCENE_DIR}"

echo "Launching scratch training for scene=${SCENE} -> ${SCENE_DIR}"

cmd=(
  python -u global_diffusion_map/refine/train_dynamic_overfit.py
  --static-root "${STATIC_ROOT}"
  --rendered-root "${RENDERED_ROOT}"
  --agg-pred-root "${AGG_PRED_ROOT}"
  --scene "${SCENE}"
  --epochs "${EPOCHS}"
  --alpha "${ALPHA}"
  --use-focal
  --cls-weight "${CLS_WEIGHT}"
  --sem-weight "${SEM_WEIGHT}"
  --sampler-steps "${SAMPLER_STEPS}"
  --sigma-min "${SIGMA_MIN}"
  --sigma-max "${SIGMA_MAX}"
  --sampler-schedule "${SAMPLER_SCHEDULE}"
  --sampler-mode "${SAMPLER_MODE}"
  --out-root "${OUT_ROOT}"
)

if [[ "${HYBRID}" == "1" ]]; then
  cmd+=(--hybrid)
fi

LOG_FILE="${SCENE_DIR}/run.log"
echo "Command: ${cmd[*]}" | tee "${SCENE_DIR}/launch_cmd.txt"
nohup "${cmd[@]}" >"${LOG_FILE}" 2>&1 &
echo $! > "${SCENE_DIR}/pid.txt"
echo "Started. PID=$(cat "${SCENE_DIR}/pid.txt"). Tail log: tail -f ${LOG_FILE}"
