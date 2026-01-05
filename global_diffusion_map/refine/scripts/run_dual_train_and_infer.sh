#!/usr/bin/env bash
set -euo pipefail

# Dual training launcher (streaming + stable-match) with auto-infer (proposal + gt_noise)
# Usage:
#   SCENE=<scene_id> GPU_STREAM=5 GPU_STABLE=4 bash -x global_diffusion_map/refine/scripts/run_dual_train_and_infer.sh

SCENE=${SCENE:-02a00399-3857-444e-8db3-a8f58489c394}
GPU_STREAM=${GPU_STREAM:-5}
GPU_STABLE=${GPU_STABLE:-4}

STATIC_ROOT=${STATIC_ROOT:-maptracker/work_dirs/static_gt_vector/av2_oldsplit/val}
RENDERED_ROOT=${RENDERED_ROOT:-maptracker/work_dirs/rendered_gt/av2_oldsplit/val}
AGG_PRED_ROOT=${AGG_PRED_ROOT:-maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid}
STATS_JSON=${STATS_JSON:-global_diffusion_map/refine/work_dirs/av2_stats.json}
POLY_CFG=${POLY_CFG:-official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py}
POLY_CKPT=${POLY_CKPT:-global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth}

OUT_STREAM=${OUT_STREAM:-global_diffusion_map/refine/work_dirs/retrain_poly_streaming_final}
OUT_STABLE=${OUT_STABLE:-global_diffusion_map/refine/work_dirs/retrain_poly_stable_matching_new}
OUT_INFER=${OUT_INFER:-global_diffusion_map/refine/work_dirs/infer_dual_stream_stable}

mkdir -p "$OUT_STREAM/$SCENE" "$OUT_STABLE/$SCENE" "$OUT_INFER"

echo "[launch] streaming trainer -> GPU=${GPU_STREAM} out=$OUT_STREAM/$SCENE"
CUDA_VISIBLE_DEVICES=${GPU_STREAM} \
PYTORCH_CUDA_ALLOC_CONF='backend:cudaMalloc,max_split_size_mb:8,garbage_collection_threshold:0.0' \
nohup python -u global_diffusion_map/refine/clean/train_one_scene_polydiffuse_encoder_streaming.py \
  --static-root "$STATIC_ROOT" \
  --rendered-root "$RENDERED_ROOT" \
  --scene "$SCENE" \
  --stats-json "$STATS_JSON" \
  --epochs 800 --batch-size 1 --accum-steps 1 \
  --steps 8 --sigma-min 0.002 --sigma-max 0.4 --rho 7.0 \
  --alpha 0.03 --matcher greedy \
  --l1-weight 20 --cls-weight 5 --sem-weight 1.0 \
  --prior-lr-mult 3.0 --save-every 0 \
  --polydiff-cfg "$POLY_CFG" \
  --pretrained-maptr-ckpt "$POLY_CKPT" \
  --out-root "$OUT_STREAM" > "$OUT_STREAM/$SCENE/train_gpu${GPU_STREAM}_stream.log" 2>&1 &
echo $! > "$OUT_STREAM/$SCENE/pid_gpu${GPU_STREAM}_stream.txt"

echo "[launch] stable-match trainer -> GPU=${GPU_STABLE} out=$OUT_STABLE/$SCENE"
CUDA_VISIBLE_DEVICES=${GPU_STABLE} \
nohup python -u global_diffusion_map/refine/clean/train_one_scene_polydiffuse_encoder.py \
  --static-root "$STATIC_ROOT" \
  --rendered-root "$RENDERED_ROOT" \
  --scene "$SCENE" \
  --stats-json "$STATS_JSON" \
  --epochs 800 --batch-size 1 --accum-steps 1 \
  --steps 8 --sigma-min 0.002 --sigma-max 0.4 --rho 7.0 \
  --alpha 0.03 --matcher greedy \
  --l1-weight 20 --cls-weight 5 --sem-weight 1.0 \
  --polydiff-cfg "$POLY_CFG" \
  --pretrained-maptr-ckpt "$POLY_CKPT" \
  --out-root "$OUT_STABLE" > "$OUT_STABLE/$SCENE/train_gpu${GPU_STABLE}_stable.log" 2>&1 &
echo $! > "$OUT_STABLE/$SCENE/pid_gpu${GPU_STABLE}_stable.txt"

STREAM_CKPT="$OUT_STREAM/$SCENE/ckpt_ep_0800.pth"
STABLE_CKPT="$OUT_STABLE/$SCENE/ckpt_ep_0800.pth"

wait_for_ckpt() {
  local path="$1"; local tag="$2"
  echo "[wait] $tag -> $path"
  while [ ! -f "$path" ]; do
    sleep 60
  done
  echo "[ok] $tag ready: $path"
}

# background waiters with infer
(
  wait_for_ckpt "$STREAM_CKPT" "streaming ckpt"
  echo "[infer] streaming proposal"
  python -u global_diffusion_map/refine/clean/infer_polydiffuse_aligned.py \
    --static-root "$STATIC_ROOT" --rendered-root "$RENDERED_ROOT" \
    --agg-pred-root "$AGG_PRED_ROOT" --stats-json "$STATS_JSON" \
    --scene "$SCENE" --ckpt "$STREAM_CKPT" \
    --steps 10 --sigma-min 0.002 --sigma-max 0.4 --rho 7.0 \
    --start proposal --out-root "$OUT_INFER/streaming_proposal" --viz-mode denoised --thr 0.2 --nms-meters 0.0 || true
  echo "[infer] streaming gt_noise"
  python -u global_diffusion_map/refine/clean/infer_polydiffuse_aligned.py \
    --static-root "$STATIC_ROOT" --rendered-root "$RENDERED_ROOT" \
    --stats-json "$STATS_JSON" --scene "$SCENE" --ckpt "$STREAM_CKPT" \
    --steps 10 --sigma-min 0.002 --sigma-max 0.4 --rho 7.0 \
    --start gt_noise --out-root "$OUT_INFER/streaming_gtnoise" --viz-mode denoised --thr 0.2 || true
) &

(
  wait_for_ckpt "$STABLE_CKPT" "stable ckpt"
  echo "[infer] stable proposal"
  python -u global_diffusion_map/refine/clean/infer_polydiffuse_aligned.py \
    --static-root "$STATIC_ROOT" --rendered-root "$RENDERED_ROOT" \
    --agg-pred-root "$AGG_PRED_ROOT" --stats-json "$STATS_JSON" \
    --scene "$SCENE" --ckpt "$STABLE_CKPT" \
    --steps 10 --sigma-min 0.002 --sigma-max 0.4 --rho 7.0 \
    --start proposal --out-root "$OUT_INFER/stable_proposal" --viz-mode denoised --thr 0.2 --nms-meters 0.0 || true
  echo "[infer] stable gt_noise"
  python -u global_diffusion_map/refine/clean/infer_polydiffuse_aligned.py \
    --static-root "$STATIC_ROOT" --rendered-root "$RENDERED_ROOT" \
    --stats-json "$STATS_JSON" --scene "$SCENE" --ckpt "$STABLE_CKPT" \
    --steps 10 --sigma-min 0.002 --sigma-max 0.4 --rho 7.0 \
    --start gt_noise --out-root "$OUT_INFER/stable_gtnoise" --viz-mode denoised --thr 0.2 || true
) &

echo "[ok] launched trainers and infer waiters."

