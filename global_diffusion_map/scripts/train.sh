#!/usr/bin/env bash
set -euo pipefail

# Aligns with PolyDiffuse scripts/train.sh, adapted for AV2 config + wrapper.
# Usage: bash mrt/global_diffusion_map/scripts/train.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Python interpreter (override with PYTHON env)
PY=${PYTHON:-python}

OUTDIR=${OUTDIR:-global_diffusion_map/work_dirs/denoise}
CONFIG=${CONFIG:-global_diffusion_map/plugin/configs/global_diffusion/av2_polydiffuse_official_base.py}
# Default to ckpts folder (symlinked to MapTracker AV2 Stage-3 by setup)
# Default to MapTR tiny R50 checkpoint for baseline training
PRETRAINED_CKPT=${PRETRAINED_CKPT:-global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth}
GUIDE_CKPT=${GUIDE_CKPT:-global_diffusion_map/ckpts/guide/network-snapshot.pth}
GUIDE_CKPT=${GUIDE_CKPT:-}

if [ -z "$GUIDE_CKPT" ] || [ ! -f "$GUIDE_CKPT" ]; then
  echo "Error: GUIDE_CKPT missing. Expected: global_diffusion_map/ckpts/guide/network-snapshot.pth (copy your guide snapshot here), or set GUIDE_CKPT explicitly." >&2
  exit 1
fi
if [ ! -f "$PRETRAINED_CKPT" ]; then
  echo "Error: PRETRAINED_CKPT not found at $PRETRAINED_CKPT. Create a symlink under global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth (or set PRETRAINED_CKPT explicitly)." >&2
  exit 1
fi

BATCH=${BATCH:-40}
BATCHGPU=${BATCHGPU:-10}
LR=${LR:-6e-4}
TICK=${TICK:-1}
SNAP=${SNAP:-10}
DUMP=${DUMP:-100}
WORKERS=${WORKERS:-16}
DURATION=${DURATION:-5}
EXP_SUFFIX=${EXP_SUFFIX:-maptr_r50_samples_5M}
P_MEAN=${P_MEAN:--0.5}
P_STD=${P_STD:-1.5}
SIG_DATA=${SIG_DATA:-1.0}
LAMBDA_DIR=${LAMBDA_DIR:-2e-3}

mkdir -p "$OUTDIR"
date -Is > "$OUTDIR/STARTED"

STATS_JSON=${AV2_STATS_JSON:-global_diffusion_map/work_dirs/av2_stats.json}
if [ ! -f "$STATS_JSON" ]; then
  echo "Computing AV2 stats to choose M and num_queries..." >&2
  "$PY" global_diffusion_map/tools/scan_av2_stats.py \
    --static-root maptracker/work_dirs/static_gt_vector_simp/av2_oldsplit/train \
    --out-json "$STATS_JSON"
fi
export AV2_STATS_JSON="$STATS_JSON"

export LD_LIBRARY_PATH=$($PY -c "import torch,os,pathlib; print(os.path.join(pathlib.Path(torch.__file__).parent,'lib'))"):${LD_LIBRARY_PATH:-}

# Multi-GPU identical usage pattern
NPROC=$(python - << 'PY'
import os
g=os.environ.get('CUDA_VISIBLE_DEVICES','')
print(len(g.split(',')) if g else 1)
PY
)
# Optional resume from training-state-*.pth
RESUME=${RESUME:-}

"$PY" -m torch.distributed.run --standalone --nproc_per_node=$NPROC global_diffusion_map/tools/run_train.py \
  --outdir        "$OUTDIR" \
  --config_path   "$CONFIG" \
  --exp_name_suffix "$EXP_SUFFIX" \
  --train_mode    denoise \
  --precond       edm \
  --duration      "$DURATION" \
  --batch         "$BATCH" \
  --batch-gpu     "$BATCHGPU" \
  --lr            "$LR" \
  --workers       "$WORKERS" \
  --tick          "$TICK" \
  --snap          "$SNAP" \
  --dump          "$DUMP" \
  --p_mean        "$P_MEAN" \
  --p_std         "$P_STD" \
  --sig_data      "$SIG_DATA" \
  --lambda_dir    "$LAMBDA_DIR" \
  --pretrained_model_ckpt "$PRETRAINED_CKPT" \
  --guide_ckpt    "$GUIDE_CKPT" \
  ${RESUME:+--resume "$RESUME"}
