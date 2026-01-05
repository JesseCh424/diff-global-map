#!/usr/bin/env bash
set -euo pipefail

# Lightweight GPU monitor. Writes one line every INTERVAL seconds:
#   YYYY-MM-DD HH:MM:SS <gpu>, <used MiB>, <util %>

GPU=${GPU:-4}
INTERVAL=${INTERVAL:-5}
OUT=${OUT:-global_diffusion_map/refine/work_dirs/retrain_poly_official/gpu${GPU}_mem.log}

mkdir -p "$(dirname "$OUT")"

for i in {1..3600}; do
  ts=$(date +%F" "%T)
  line=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n "$((GPU+1))p" || true)
  echo "$ts $line" | tee -a "$OUT"
  sleep "$INTERVAL"
done

