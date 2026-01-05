#!/usr/bin/env bash
set -euo pipefail

# Simple GPU/CPU monitor for the current training job.
# Usage:
#   bash -x global_diffusion_map/refine/scripts/monitor_training.sh [interval_sec] [samples]
# Example (sample every 10s, 120 samples):
#   bash -x global_diffusion_map/refine/scripts/monitor_training.sh 10 120 | tee work_dirs/monitor.log

INTERVAL=${1:-10}
SAMPLES=${2:-60}

echo "[monitor] interval=${INTERVAL}s samples=${SAMPLES}  (Ctrl-C to stop)"
echo "timestamp, gpu_idx, name, util_sm%, util_mem%, mem_used(MiB), mem_total(MiB), temp(C)"
for ((i=1; i<=SAMPLES; i++)); do
  TS=$(date +"%F %T")
  # GPU summary line-per-GPU
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null | awk -v ts="$TS" -F, '{gsub(/^ +| +$/,"",$0); printf "%s, %s\n", ts, $0}'
  # Active GPU processes (one-shot)
  echo "# procs: gpu pid sm% mem% cmd"; nvidia-smi pmon -c 1 2>/dev/null | sed '1,2d' | awk '{print $0}'
  # CPU load/mem
  echo "# host: $(hostname)  load: $(uptime | awk -F'load average:' '{print $2}')"
  free -h | sed -n '1,2p' | sed "s/^/$TS /"
  echo "---"
  sleep "$INTERVAL"
done

