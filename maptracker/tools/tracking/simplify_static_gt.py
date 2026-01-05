#!/usr/bin/env python
"""
Simplify per-scene cropped static GT vectors and render a PNG aligned to 05.

Algorithm (per polyline/polygon):
- Remove middle point if the turn angle between adjacent segments is small.
- If p_{i-1} and p_{i+1} are close and turn is small, drop p_i as redundant.

Outputs per scene under --out-root/<scene>/:
- 09_static_simp.png  (same plotting bounds as 05)
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List

import mmcv
import numpy as np

# Reuse the 05 renderer and bounds helper
import sys

# Make visualization modules importable whether run from repo root or maptracker/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(SCRIPT_DIR)          # .../maptracker/tools
PKG_ROOT = os.path.dirname(TOOLS_DIR)            # .../maptracker
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.append(PKG_ROOT)

try:
    from tools.visualization.render_static_maptracker_style import (
        render_static_mt_style,
        _Args as _MTArgs,
    )
except ModuleNotFoundError:
    from visualization.render_static_maptracker_style import (
        render_static_mt_style,
        _Args as _MTArgs,
    )
try:
    from tools.visualization.vis_global import compute_bounds_from_pkls
except ModuleNotFoundError:
    from visualization.vis_global import compute_bounds_from_pkls


def _dedup_consecutive(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = [0]
    for i in range(1, len(points)):
        if not np.allclose(points[i], points[i - 1]):
            keep.append(i)
    return points[keep]


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def simplify_path(points: np.ndarray, angle_thr_deg: float, dist_eps: float, closed: bool,
                  dist2_eps: float, angle2_deg: float, angle_tiny_deg: float,
                  enable_two_hop: bool = True,
                  short_eps: float = 0.5,
                  angle_relax_deg: float = 20.0,
                  pair_merge_eps: float = 0.5,
                  pair_merge_angle_deg: float = 45.0) -> np.ndarray:
    """Simplify a polyline/polygon by removing nearly straight or redundant points."""
    pts = _dedup_consecutive(np.asarray(points, dtype=np.float64))
    min_pts = 3 if closed else 2
    if len(pts) <= min_pts:
        return pts

    changed = True
    iters = 0
    while changed and iters < 5:
        changed = False
        iters += 1
        # Pass A: merge consecutive near-duplicate points (|p_k - p_{k+1}| small)
        # Guard against sharp direction changes across the pair using (k-1->k) vs (k+1->k+2).
        merged_any = False
        new_pts: List[np.ndarray] = []
        i = 0
        N = len(pts)
        while i < N:
            j = i + 1
            can_merge = False
            if j < N:
                d = float(np.linalg.norm(pts[j] - pts[i]))
                if d <= pair_merge_eps:
                    # For open polylines, preserve endpoints
                    if not closed and (i == 0 or j == N - 1):
                        can_merge = False
                    else:
                        # Check angle between prev and next segments
                        if closed:
                            im1 = (i - 1) % N
                            ip2 = (j + 1) % N
                        else:
                            im1 = i - 1
                            ip2 = j + 1
                        if (closed or (im1 >= 0 and ip2 < N)):
                            v_prev = pts[i] - pts[im1]
                            v_next = pts[ip2] - pts[j]
                            ang_pair = _angle_deg(v_prev, v_next)
                            # Merge only if not a sharp turn across the pair
                            if ang_pair < pair_merge_angle_deg:
                                can_merge = True
                        else:
                            # If neighbors missing (ends), do not merge
                            can_merge = False
            if can_merge:
                m = (pts[i] + pts[j]) * 0.5
                new_pts.append(m)
                merged_any = True
                i += 2
            else:
                new_pts.append(pts[i])
                i += 1
        if merged_any:
            pts = np.asarray(new_pts, dtype=np.float64)
            changed = True
            continue

        # Pass B: remove nearly straight middle points (tiny turn), keep endpoints of open polylines
        if closed:
            idxs = list(range(len(pts)))
        else:
            idxs = list(range(1, len(pts) - 1))
        remove = set()
        for i in idxs:
            if closed:
                im1 = (i - 1) % len(pts)
                ip1 = (i + 1) % len(pts)
            else:
                im1 = i - 1
                ip1 = i + 1
                if im1 < 0 or ip1 >= len(pts):
                    continue
            p0, p1, p2 = pts[im1], pts[i], pts[ip1]
            v1 = p1 - p0
            v2 = p2 - p1
            ang = _angle_deg(v1, v2)
            # Condition 1 (straight line): if angle is tiny, drop regardless of distance
            if ang <= angle_tiny_deg:
                if not (not closed and (i == 0 or i == len(pts) - 1)):
                    remove.add(i)
                    continue
            # Condition 2 replaced by pair-merge; no additional 1-hop deletion here
            # Rule C (span-2): optional two-hop pruning on gentle curves
            if enable_two_hop:
                if closed:
                    im2 = (i - 2) % len(pts)
                    ip2 = (i + 2) % len(pts)
                else:
                    im2 = i - 2
                    ip2 = i + 2
                if (closed or (im2 >= 0 and ip2 < len(pts))):
                    p_2, p2p = pts[im2], pts[ip2]
                    v1b = p1 - p_2
                    v2b = p2p - p1
                    ang2 = _angle_deg(v1b, v2b)
                    if ang2 <= angle2_deg and np.linalg.norm(p2p - p_2) <= dist2_eps:
                        remove.add(i)
                        continue
        if remove:
            keep_mask = np.ones(len(pts), dtype=bool)
            if not closed:
                keep_mask[0] = True
                keep_mask[-1] = True
            for j in sorted(remove):
                if not closed and (j == 0 or j == len(pts) - 1):
                    continue
                keep_mask[j] = False
            new_pts = pts[keep_mask]
            if len(new_pts) >= min_pts:
                pts = new_pts
                changed = True
            else:
                break
    # If closed, ensure the path is not duplicated at ends
    if closed and len(pts) >= 3:
        # Do not force closure with repeated last=first; plotting functions expect open polylines
        pass
    return pts.astype(np.float32)


def to_bank(scene_vecs: Dict[int, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    bank: Dict[str, List[np.ndarray]] = {}
    for lbl, arrs in scene_vecs.items():
        for gid, a in enumerate(arrs):
            bank[f"{lbl}_{gid}"] = [np.asarray(a, dtype=np.float32)]
    return bank


def main():
    ap = argparse.ArgumentParser(description="Simplify static GT vectors and render 09_static_gt_simp.png")
    ap.add_argument("--static-root", required=True, help="Static GT pkl root (cropped)")
    ap.add_argument("--aggregated-pred", required=True, help="Aggregated pred dir for bounds")
    ap.add_argument("--aggregated-gt", required=True, help="Aggregated GT dir for bounds")
    ap.add_argument("--out-root", required=True, help="Output root containing <scene>/ folders (legacy; used if --viz-root unset)")
    ap.add_argument("--out-pkl-root", required=False, default=None, help="Output root to save simplified vectors as pkls")
    ap.add_argument("--viz-root", required=False, default=None, help="Viz root to write PNG under <viz_root>/<scene>/")
    ap.add_argument("--no-png", action="store_true", help="Skip rendering PNGs; only write simplified vector pickles")
    ap.add_argument("--png-name", required=False, default="09_static_gt_simp.png", help="Filename for rendered PNG")
    ap.add_argument("--scenes", nargs="+", required=True, help="Scene IDs to process")
    ap.add_argument("--angle-deg", type=float, default=6.0, help="Turn (deg) threshold for distance-gated dropping")
    ap.add_argument("--angle-tiny-deg", type=float, default=4.0, help="Tiny turn (deg) to drop unconditionally (straight lines)")
    ap.add_argument("--dist-eps", type=float, default=1.0, help="Distance (m) threshold for short-chord removal (p[i-1]↔p[i+1])")
    ap.add_argument("--dist2-eps", type=float, default=0.6, help="Two-hop chord distance (m) to further simplify curves")
    ap.add_argument("--angle2-deg", type=float, default=8.0, help="Max two-hop turn (deg) to drop middle point")
    ap.add_argument("--no-2hop", action="store_true", help="Disable two-hop pruning rule entirely")
    ap.add_argument("--short-eps", type=float, default=0.5, help="(Unused when pair-merge active) very short chord threshold")
    ap.add_argument("--angle-relax-deg", type=float, default=20.0, help="(Unused when pair-merge active) relaxed angle threshold")
    ap.add_argument("--pair-merge-eps", type=float, default=0.5, help="Distance (m) to merge consecutive points into midpoint")
    ap.add_argument("--pair-merge-angle-deg", type=float, default=45.0, help="Do not merge if (k-1->k) vs (k+1->k+2) angle >= this")
    ap.add_argument("--dpi", type=int, default=60, help="DPI for rendering")
    args = ap.parse_args()

    for scene in args.scenes:
        pkl_path = os.path.join(args.static_root, f"{scene}.pkl")
        if not os.path.exists(pkl_path):
            print(f"[skip] static GT missing: {pkl_path}")
            continue
        data = mmcv.load(pkl_path)
        car_traj = data.get("car_trajectory", [])
        input_bounds = data.get("bounds", None)
        scene_vecs: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        for lbl in (0, 1, 2):
            for arr in data.get(lbl, []):
                a = np.asarray(arr, dtype=np.float32)
                closed = (lbl == 0)
                simp = simplify_path(
                    a,
                    args.angle_deg,
                    args.dist_eps,
                    closed,
                    args.dist2_eps,
                    args.angle2_deg,
                    args.angle_tiny_deg,
                    enable_two_hop=(not args.no_2hop),
                    short_eps=args.short_eps,
                    angle_relax_deg=args.angle_relax_deg,
                    pair_merge_eps=args.pair_merge_eps,
                    pair_merge_angle_deg=args.pair_merge_angle_deg,
                )
                if closed and len(simp) >= 3:
                    scene_vecs[lbl].append(simp)
                elif not closed and len(simp) >= 2:
                    scene_vecs[lbl].append(simp)
        # Compute bounds from aggregated pkls (match 05 axes)
        bounds = compute_bounds_from_pkls([
            os.path.join(args.aggregated_pred, f"{scene}.pkl"),
            os.path.join(args.aggregated_gt, f"{scene}.pkl"),
        ])
        if bounds is None:
            print(f"[warn] bounds not found from pkls; skipping {scene}")
            continue
        x_min, x_max, y_min, y_max = bounds
        # Optional: dump simplified vectors as pkl
        if args.out_pkl_root:
            pkl_out_dir = Path(args.out_pkl_root)
            pkl_out_dir.mkdir(parents=True, exist_ok=True)
            out_data = {0: scene_vecs.get(0, []), 1: scene_vecs.get(1, []), 2: scene_vecs.get(2, []),
                        "car_trajectory": car_traj}
            if input_bounds is not None:
                out_data["bounds"] = input_bounds
            mmcv.dump(out_data, str(pkl_out_dir / f"{scene}.pkl"))
        if not args.no_png:
            # Decide PNG path (viz root preferred)
            if args.viz_root:
                out_dir = Path(args.viz_root) / scene
            else:
                out_dir = Path(args.out_root) / scene
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / args.png_name
            # Render using the same style function as 05
            mt_args = _MTArgs(simplify=0.0, line_opacity=0.75, dpi=args.dpi)
            # Prepare car trajectory for plotting
            if car_traj:
                car_traj_plot = [
                    [np.asarray(entry.get("center", [0.0, 0.0]), dtype=np.float32), float(entry.get("yaw_deg", 0.0))]
                    for entry in car_traj
                ]
            else:
                car_traj_plot = [[np.array([0.0, 0.0]), 0.0]]
            bank = to_bank(scene_vecs)
            render_static_mt_style(bank, out_path, x_min, x_max, y_min, y_max, mt_args, car_traj_plot)
            print(f"[ok] {scene}: {out_path}")
        else:
            print(f"[ok] {scene}: simplified vectors written (no PNG)")


if __name__ == "__main__":
    main()
