#!/usr/bin/env bash
set -euo pipefail

# Auto-wait for a checkpoint under a one-scene run dir and launch inference (proposal + gt_noise)
# Usage:
#   bash global_diffusion_map/refine/scripts/auto_infer_one_scene.sh \
#     --run-dir global_diffusion_map/refine/work_dirs/train_edm_one_scene_<scene>/<scene> \
#     --scene <scene_id> \
#     --agg maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid \
#     --static maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
#     --rendered maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
#     [--wait-ep 200] [--gpu 0]

RUN_DIR=""
SCENE=""
AGG=""
STATIC=""
RENDERED=""
WAIT_EP="200"
GPU="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2;;
    --scene) SCENE="$2"; shift 2;;
    --agg) AGG="$2"; shift 2;;
    --static) STATIC="$2"; shift 2;;
    --rendered) RENDERED="$2"; shift 2;;
    --wait-ep) WAIT_EP="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    *) echo "[err] unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$RUN_DIR" || -z "$SCENE" || -z "$AGG" || -z "$STATIC" || -z "$RENDERED" ]]; then
  echo "Usage: $0 --run-dir <dir>/<scene> --scene <id> --agg <dir> --static <dir> --rendered <dir> [--wait-ep 200] [--gpu 0]" >&2
  exit 2
fi

printf "[auto-infer] run=%s scene=%s wait_ep=%s gpu=%s\n" "$RUN_DIR" "$SCENE" "$WAIT_EP" "$GPU"

ckpt_path=""
target=""
if [[ "$WAIT_EP" =~ ^[0-9]+$ ]]; then
  ep=$(printf "%04d" "$WAIT_EP")
  target="ckpt_ep_${ep}.pth"
  ckpt_path="${RUN_DIR}/${target}"
  while [[ ! -f "$ckpt_path" ]]; do
    echo "[wait] $ckpt_path"; sleep 15;
  done
else
  # wait for any ckpt then pick numerically largest
  while true; do
    mapfile -t arr < <(ls -1 "$RUN_DIR"/ckpt_ep_*.pth 2>/dev/null || true)
    if [[ ${#arr[@]} -gt 0 ]]; then
      ckpt_path=$(ls -1 "$RUN_DIR"/ckpt_ep_*.pth | sed 's/.*ckpt_ep_\([0-9][0-9][0-9][0-9]\)\.pth/\1 \0/' | sort -nr | head -n1 | awk '{print $2}')
      break
    fi
    echo "[wait] any ckpt under $RUN_DIR"; sleep 15;
  done
fi

echo "[auto-infer] ckpt=$ckpt_path"
TS=$(date +%Y%m%d_%H%M%S)
OUTP="global_diffusion_map/refine/work_dirs/infer_${SCENE}_proposal_states_${TS}"
OUTG="global_diffusion_map/refine/work_dirs/infer_${SCENE}_gtnoise_states_${TS}"
mkdir -p "$OUTP" "$OUTG"

CUDA_VISIBLE_DEVICES="$GPU" python -u global_diffusion_map/refine/infer_refine.py \
  --agg-pred-root "$AGG" \
  --static-root    "$STATIC" \
  --rendered-root  "$RENDERED" \
  --scene "$SCENE" \
  --ckpt  "$ckpt_path" \
  --start proposal \
  --steps 18 \
  --sigma-min 0.002 \
  --sigma-max 0.5 \
  --steps-thr 0.2 \
  --steps-all \
  --thr 0.2 \
  --save-steps-dir "$OUTP/steps" \
  --out-root "$OUTP"

CUDA_VISIBLE_DEVICES="$GPU" python -u global_diffusion_map/refine/infer_refine.py \
  --agg-pred-root "$AGG" \
  --static-root    "$STATIC" \
  --rendered-root  "$RENDERED" \
  --scene "$SCENE" \
  --ckpt  "$ckpt_path" \
  --start gt_noise \
  --steps 18 \
  --sigma-min 0.002 \
  --sigma-max 0.5 \
  --steps-thr 0.2 \
  --steps-all \
  --thr 0.2 \
  --save-steps-dir "$OUTG/steps" \
  --out-root "$OUTG"

echo "[ok] auto-infer done: $OUTP and $OUTG"
