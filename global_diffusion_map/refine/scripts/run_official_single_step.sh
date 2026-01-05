#!/usr/bin/env bash
set -euo pipefail

# Official‑aligned single‑step training launcher for PolyDiffuse encoder.
# - Kills any existing trainer
# - Starts training on the chosen GPU with: steps=1, no step-loss, final loss only
# - Logs to OUTDIR/train_gpu${GPU}.log and writes PID to OUTDIR/train_gpu${GPU}.pid

GPU=${GPU:-4}
SCENE=${SCENE:-02a00399-3857-444e-8db3-a8f58489c394}
STATIC_ROOT=${STATIC_ROOT:-maptracker/work_dirs/static_gt_vector/av2_oldsplit/val}
RENDERED_ROOT=${RENDERED_ROOT:-maptracker/work_dirs/rendered_gt/av2_oldsplit/val}
STATS_JSON=${STATS_JSON:-global_diffusion_map/work_dirs/av2_stats.json}
POLY_CFG=${POLY_CFG:-official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py}
POLY_CKPT=${POLY_CKPT:-global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth}
RESUME_CKPT=${RESUME_CKPT:-global_diffusion_map/refine/work_dirs/retrain_poly_800/${SCENE}/ckpt_ep_0250.pth}
RESUME_EPOCH=${RESUME_EPOCH:-250}
OUTDIR=${OUTDIR:-global_diffusion_map/refine/work_dirs/retrain_poly_official}

mkdir -p "${OUTDIR}"

echo "[kill] stopping any running poly trainers"
ps -eo pid,cmd | awk '/train_one_scene_polydiffuse_encoder.py/{print $1}' | xargs -r -n1 kill -9 || true

echo "[launch] GPU=${GPU} SCENE=${SCENE} OUTDIR=${OUTDIR}"
export CUDA_VISIBLE_DEVICES=${GPU}
export REFINE_VEC_LOSS=1
export PYTHONUNBUFFERED=1

CMD=(
  python -u global_diffusion_map/refine/clean/train_one_scene_polydiffuse_encoder.py
  --static-root "${STATIC_ROOT}"
  --rendered-root "${RENDERED_ROOT}"
  --stats-json "${STATS_JSON}"
  --scene "${SCENE}"
  --epochs 800 --lr 2e-4 --sched cosine --lr-min 1e-5
  --steps 1 --sigma-min 0.002 --sigma-max 0.6 --rho 7.0
  --shift-sigma 0.10 --point-sigma 0.02 --drop-frac 0.15 --ghosts 2
  --l1-weight 20 --cls-weight 5 --sem-weight 1.0
  --step-loss-weight 0.0 --final-loss-weight 1.0
  --anchor-max-center-dist 0.05
  --batch-size 4 --accum-steps 1
  --polydiff-cfg "${POLY_CFG}"
  --pretrained-maptr-ckpt "${POLY_CKPT}"
  --out-root "${OUTDIR}"
  --resume-ckpt "${RESUME_CKPT}" --resume-epoch "${RESUME_EPOCH}"
)

setsid bash -lc "${CMD[*]}" > "${OUTDIR}/train_gpu${GPU}.log" 2>&1 &
echo $! > "${OUTDIR}/train_gpu${GPU}.pid"
echo "[ok] started. pid=$(cat "${OUTDIR}/train_gpu${GPU}.pid") log=${OUTDIR}/train_gpu${GPU}.log"

