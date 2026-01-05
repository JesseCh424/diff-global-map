#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU guidance training aligned to PolyDiffuse's train_guide.sh
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash global_diffusion_map/scripts/dist_train_guide.sh
# Tunables via env: OUTDIR, CONFIG, BATCH, BATCHGPU, LR, TICK, SNAP, WORKERS, P_MEAN, P_STD, SIG_DATA, DURATION, EXP_SUFFIX, AV2_STATS_JSON, GUIDE_VERBOSE_EVERY

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

OUTDIR=${OUTDIR:-global_diffusion_map/work_dirs/guide}
CONFIG=${CONFIG:-global_diffusion_map/plugin/configs/global_diffusion/av2_polydiffuse_official_guide.py}
BATCH=${BATCH:-140}
BATCHGPU=${BATCHGPU:-35}
LR=${LR:-2e-4}
TICK=${TICK:-1}
SNAP=${SNAP:-5}
WORKERS=${WORKERS:-8}
P_MEAN=${P_MEAN:-1.0}
P_STD=${P_STD:-4.0}
SIG_DATA=${SIG_DATA:-1.0}
DURATION=${DURATION:-0.1}
EXP_SUFFIX=${EXP_SUFFIX:-reg_0.1}
GUIDE_VERBOSE_EVERY=${GUIDE_VERBOSE_EVERY:-20}
# Disable guide visualization by default; set to 0 to enable
GUIDE_DISABLE_VIZ=${GUIDE_DISABLE_VIZ:-1}

mkdir -p "$OUTDIR"
date -Is > "$OUTDIR/STARTED"

# Python interpreter (override with PYTHON env)
PY=${PYTHON:-python}

# Ensure stats exist for AV2 caps
STATS_JSON=${AV2_STATS_JSON:-global_diffusion_map/work_dirs/av2_stats.json}
if [ ! -f "$STATS_JSON" ]; then
  echo "Computing AV2 stats to choose M and num_queries..." >&2
  "$PY" global_diffusion_map/tools/scan_av2_stats.py \
    --static-root maptracker/work_dirs/static_gt_vector_simp/av2_oldsplit/train \
    --out-json "$STATS_JSON"
fi
export AV2_STATS_JSON="$STATS_JSON"

# Torch libs for compiled ops
export LD_LIBRARY_PATH=$($PY -c "import torch,os,pathlib; print(os.path.join(pathlib.Path(torch.__file__).parent,'lib'))"):${LD_LIBRARY_PATH:-}
export GUIDE_VERBOSE_EVERY
export GUIDE_DISABLE_VIZ

# Derive nproc from CUDA_VISIBLE_DEVICES
NPROC=$(python - << 'PY'
import os
g=os.environ.get('CUDA_VISIBLE_DEVICES','')
print(len(g.split(',')) if g else 1)
PY
)

exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=$NPROC global_diffusion_map/tools/run_train.py \
  --outdir          "$OUTDIR" \
  --config_path     "$CONFIG" \
  --exp_name_suffix "$EXP_SUFFIX" \
  --train_mode      guide \
  --precond         edm \
  --duration        "$DURATION" \
  --batch           "$BATCH" \
  --batch-gpu       "$BATCHGPU" \
  --lr              "$LR" \
  --workers         "$WORKERS" \
  --tick            "$TICK" \
  --snap            "$SNAP" \
  --p_mean          "$P_MEAN" \
  --p_std           "$P_STD" \
  --sig_data        "$SIG_DATA"
