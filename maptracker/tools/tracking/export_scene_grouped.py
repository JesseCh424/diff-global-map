#!/usr/bin/env python
"""
Export per-scene aggregated vectors using MapTracker's built-in utilities.

Inputs:
- config: MapTracker config (for roi_size/pc_range)
- --pred-pkl: pos_predictions_*.pkl from tools/tracking/prepare_pred_tracks.py

Outputs:
- --out-dir/<scene>.pkl per scene with merged vectors:
  {0: [Nx2 arrays], 1: [...], 2: [...]} in meters under the last frame ego frame.

Note: This is a thin wrapper that reuses MapTracker functions only; no new logic.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List

import mmcv
import numpy as np
from mmcv import Config

# Make sibling tool modules importable whether run from repo root or maptracker/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(SCRIPT_DIR)          # .../maptracker/tools
PKG_ROOT = os.path.dirname(TOOLS_DIR)            # .../maptracker
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.append(PKG_ROOT)

from tracking.cmap_utils.match_utils import get_prev2curr_matrix
from visualization.vis_global import (
    get_prev2curr_vectors,
    merge_corssing,
    merge_divider,
    merge_boundary,
)

from shapely.geometry import LineString as ShpLineString


def parse_args():
    p = argparse.ArgumentParser(description="Export per-scene aggregated vectors")
    p.add_argument("config", help="Config file path")
    p.add_argument("--pred-pkl", required=True, help="Path to pos_predictions_*.pkl")
    p.add_argument("--out-dir", required=True, help="Output dir for per-scene pkls")
    p.add_argument("--simplify", type=float, default=0.5, help="Simplify tolerance")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def load_predictions(pred_pkl: str) -> List[dict]:
    with open(pred_pkl, "rb") as f:
        preds = pickle.load(f)
    return preds


def group_by_scene(preds: List[dict]) -> Dict[str, List[int]]:
    scene2idx: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(preds):
        scene2idx[rec["scene_name"]].append(i)
    for s, idxs in scene2idx.items():
        idxs.sort(key=lambda j: preds[j].get("local_idx", j))
    return scene2idx


def export_scene(
    scene: str,
    idxs: List[int],
    preds: List[dict],
    roi_size: np.ndarray,
    origin: np.ndarray,
    out_dir: str,
    simplify_tol: float,
    overwrite: bool,
):
    out_path = os.path.join(out_dir, f"{scene}.pkl")
    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] {scene}: exists -> {out_path}")
        return

    last_rec = preds[idxs[-1]]
    bank: Dict[str, List[np.ndarray]] = defaultdict(list)
    car_trajectory: List[Dict[str, Any]] = []

    for j in idxs:
        rec = preds[j]
        raw_vecs = np.asarray(rec["vectors"]).reshape((-1, 20, 2))
        is_norm = float(np.abs(raw_vecs).max()) <= 1.0
        prev2curr = get_prev2curr_matrix(rec["meta"], last_rec["meta"])  # 4x4
        vecs_norm = get_prev2curr_vectors(raw_vecs, prev2curr, origin, roi_size, denormalize=is_norm, clip=False)
        vecs_abs = vecs_norm * roi_size + origin

        prev2curr_np = prev2curr.detach().cpu().numpy()
        car_hom = prev2curr_np @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        center_xy = car_hom[:2].astype(np.float32)
        yaw_deg = float(np.degrees(np.arctan2(prev2curr_np[1, 0], prev2curr_np[0, 0])))
        car_trajectory.append(
            {
                "frame_idx": int(rec.get("local_idx", j)),
                "center": center_xy.tolist(),
                "yaw_deg": yaw_deg,
            }
        )

        for k, (lbl, gid) in enumerate(zip(rec["labels"], rec["global_ids"])):
            key = f"{int(lbl)}_{int(gid)}"
            bank[key].append(np.asarray(vecs_abs[k]))

    # Merge vectors per global_id, then collect by label
    merged_by_label: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}

    for key, lines in bank.items():
        lbl, gid = map(int, key.split("_"))

        # Merge vectors with the same global_id
        if lbl == 0:  # crossing: merge via convex hull
            polygons = [ShpLineString(v) for v in lines if len(v) >= 2]
            if polygons:
                hull = merge_corssing(polygons)
                if hull and not hull.is_empty:
                    merged_by_label[0].append(np.asarray(hull.exterior.coords))
        elif lbl == 1:  # divider: merge via interpolation
            merged = merge_divider([np.asarray(v) for v in lines])
            merged_by_label[1].extend([np.asarray(v) for v in merged])
        elif lbl == 2:  # boundary: merge via interpolation
            merged = merge_boundary([np.asarray(v) for v in lines])
            merged_by_label[2].extend([np.asarray(v) for v in merged])

    if simplify_tol > 0:
        def _simp(arr: np.ndarray) -> np.ndarray:
            if len(arr) < 2:
                return arr
            simp = ShpLineString(arr).simplify(simplify_tol)
            coords = np.asarray(simp.coords)
            return coords if len(coords) >= 2 else arr
        for lbl in merged_by_label:
            merged_by_label[lbl] = [_simp(a) for a in merged_by_label[lbl]]

    merged_by_label["car_trajectory"] = car_trajectory

    mmcv.dump(merged_by_label, out_path)
    print(f"[ok] {scene}: {out_path}")


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    roi_size = np.array(cfg.get("roi_size", (60, 30)), dtype=np.float32)
    origin = np.array(cfg.get("pc_range", [-roi_size[0] / 2, -roi_size[1] / 2, -3, roi_size[0] / 2, roi_size[1] / 2, 5])[:2], dtype=np.float32)

    os.makedirs(args.out_dir, exist_ok=True)
    preds = load_predictions(args.pred_pkl)
    scene2idx = group_by_scene(preds)

    for scene, idxs in sorted(scene2idx.items()):
        export_scene(scene, idxs, preds, roi_size, origin, args.out_dir, args.simplify, args.overwrite)


if __name__ == "__main__":
    main()
