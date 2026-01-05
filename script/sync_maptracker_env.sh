#!/usr/bin/env bash
set -euo pipefail

# Sync the maptracker conda env to match requirements_maptracker.txt
# Usage: bash script/sync_maptracker_env.sh

REQ_FILE="requirements_maptracker.txt"
ENV_NAME="maptracker"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found in PATH. Please load conda and retry." >&2
  exit 2
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' not found. Create it first (see maptracker_env.yml)." >&2
  exit 2
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Missing $REQ_FILE at repo root." >&2
  exit 2
fi

timestamp() { date +%Y%m%d_%H%M%S; }

echo "Backing up current pip freeze from '$ENV_NAME'..."
conda run -n "$ENV_NAME" python -m pip freeze > "maptracker_freeze_$(timestamp).txt"

echo "Preparing sanitized requirements (skip ROS/apt-only entries)..."
SAN_REQ="/tmp/req_maptracker_sanitized_$(timestamp).txt"
grep -Ev "^(#|\s*$|ros-|gazebo|cv-bridge$|sensor-msgs$|geometry-msgs$|tf$|tf2|smach|diagnostic-|interactive-markers|angles$|catkin$|actionlib$|laser_geometry$|joint-state-publisher|camera-calibration$|camera-calibration-parsers$|bondpy$|jsk-|gps_common$|image-geometry$|openni2_launch$|lanelet2-python$)" "$REQ_FILE" > "$SAN_REQ"

echo "Installing/upgrading pinned packages into '$ENV_NAME' (this may take a while)..."
conda run -n "$ENV_NAME" python -m pip install --no-input --no-cache-dir -r "$SAN_REQ"

echo "Re-checking environment after install..."
conda run -n "$ENV_NAME" python tools/check_env.py || true

echo "Done. Review the summary above. ROS/apt packages (if any) must be installed via your OS/ROS setup."

