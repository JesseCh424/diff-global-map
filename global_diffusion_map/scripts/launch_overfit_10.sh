#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OUTDIR=${OUTDIR:-global_diffusion_map/work_dirs/denoise_overfit_10_S}
export GUIDE_CKPT=${GUIDE_CKPT:-global_diffusion_map/ckpts/guide/network-snapshot_m30q64.pth}
export CONFIG=${CONFIG:-global_diffusion_map/plugin/configs/global_diffusion/av2_polydiffuse_overfit_10.py}
export BATCH=${BATCH:-32}
export BATCHGPU=${BATCHGPU:-8}
export WORKERS=${WORKERS:-4}
export DURATION=${DURATION:-1}
export TICK=${TICK:-1}
export SNAP=${SNAP:-5}
export DUMP=${DUMP:-50}
export P_MEAN=${P_MEAN:--0.5}
export P_STD=${P_STD:-1.5}
export SIG_DATA=${SIG_DATA:-1.0}
export LAMBDA_DIR=${LAMBDA_DIR:-2e-3}
export VAL_EVERY_TICKS=${VAL_EVERY_TICKS:-0}

exec bash -x global_diffusion_map/scripts/train.sh

