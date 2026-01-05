#!/usr/bin/env python
"""
Compute and compare polygon/polyline point counts between simplified GT static vectors
and aggregated prediction vectors for a set of scenes. Prints per-scene maxima and
global maxima across all processed scenes.

Scenes are discovered from a root containing per-scene folders (e.g., semantic out root).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import mmcv
import numpy as np


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _dedup_consecutive(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    keep = [0]
    for i in range(1, len(points)):
        if not np.allclose(points[i], points[i - 1]):
            keep.append(i)
    return points[keep]


def simplify_poly(points: np.ndarray, *, closed: bool, angle_tiny_deg: float,
                  pair_merge_eps: float, pair_merge_angle_deg: float,
                  max_passes: int = 5) -> np.ndarray:
    """Simplify by pair-merging close consecutive points (guarding corners) and
    dropping tiny-turn middle points. Endpoints of open polylines are preserved."""
    pts = _dedup_consecutive(np.asarray(points, dtype=np.float64))
    min_pts = 3 if closed else 2
    if len(pts) <= min_pts:
        return pts.astype(np.float32)
    changed = True
    passes = 0
    while changed and passes < max_passes and len(pts) > min_pts:
        passes += 1
        changed = False
        # A) Pair merge
        new_pts: List[np.ndarray] = []
        i = 0
        N = len(pts)
        merged = False
        while i < N:
            j = i + 1
            can_merge = False
            if j < N and float(np.linalg.norm(pts[j] - pts[i])) <= pair_merge_eps:
                if not closed and (i == 0 or j == N - 1):
                    can_merge = False
                else:
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
                        if ang_pair < pair_merge_angle_deg:
                            can_merge = True
            if can_merge:
                m = (pts[i] + pts[j]) * 0.5
                new_pts.append(m)
                merged = True
                i += 2
            else:
                new_pts.append(pts[i])
                i += 1
        if merged:
            pts = np.asarray(new_pts, dtype=np.float64)
            changed = True
            if len(pts) <= min_pts:
                break
            continue
        # B) Tiny-turn drop
        keep = np.ones(len(pts), dtype=bool)
        if not closed:
            keep[0] = True
            keep[-1] = True
        for i in range(len(pts)):
            if not closed and (i == 0 or i == len(pts) - 1):
                continue
            im1 = (i - 1) % len(pts)
            ip1 = (i + 1) % len(pts)
            p0, p1, p2 = pts[im1], pts[i], pts[ip1]
            ang = _angle_deg(p1 - p0, p2 - p1)
            if ang <= angle_tiny_deg:
                keep[i] = False
        new_pts = pts[keep]
        if len(new_pts) >= min_pts and len(new_pts) < len(pts):
            pts = new_pts
            changed = True
    return pts.astype(np.float32)


def count_max_points(arrs: List[np.ndarray]) -> int:
    mx = 0
    for a in arrs:
        n = int(np.asarray(a).shape[0])
        if n > mx:
            mx = n
    return mx


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare max polygon points in simplified GT vs agg pred")
    ap.add_argument("--scenes-root", required=True, help="Root with per-scene folders (discover scenes)")
    ap.add_argument("--static-root", required=True, help="Static GT pkl root")
    ap.add_argument("--agg-pred", required=True, help="Aggregated pred pkl root")
    ap.add_argument("--angle-tiny-deg", type=float, default=4.0)
    ap.add_argument("--pair-merge-eps", type=float, default=2.0)
    ap.add_argument("--pair-merge-angle-deg", type=float, default=60.0)
    args = ap.parse_args()

    scenes = [d for d in os.listdir(args.scenes_root)
              if os.path.isdir(os.path.join(args.scenes_root, d)) and '-' in d]
    scenes.sort()
    if not scenes:
        print("[warn] no scenes found")
        return

    global_max_gt = 0
    global_max_pred = 0
    per_scene = []
    for scene in scenes:
        static_pkl = os.path.join(args.static_root, f"{scene}.pkl")
        pred_pkl = os.path.join(args.agg_pred, f"{scene}.pkl")
        if not os.path.exists(static_pkl) or not os.path.exists(pred_pkl):
            per_scene.append((scene, None, None))
            continue
        d_gt = mmcv.load(static_pkl)
        d_pr = mmcv.load(pred_pkl)
        # Simplify GT vectors and count
        max_gt = 0
        for lbl in (0, 1, 2):
            for arr in d_gt.get(lbl, []):
                pts = np.asarray(arr, dtype=np.float32)
                sim = simplify_poly(
                    pts,
                    closed=(lbl == 0),
                    angle_tiny_deg=args.angle_tiny_deg,
                    pair_merge_eps=args.pair_merge_eps,
                    pair_merge_angle_deg=args.pair_merge_angle_deg,
                )
                max_gt = max(max_gt, int(sim.shape[0]))
        # Aggregated pred count
        max_pr = 0
        for lbl in (0, 1, 2):
            for arr in d_pr.get(lbl, []):
                max_pr = max(max_pr, int(np.asarray(arr).shape[0]))
        global_max_gt = max(global_max_gt, max_gt)
        global_max_pred = max(global_max_pred, max_pr)
        per_scene.append((scene, max_gt, max_pr))

    # Output summary
    for scene, mg, mp in per_scene:
        if mg is None:
            print(f"{scene}: [skip] missing pkl(s)")
        else:
            print(f"{scene}: max_pts_gt_simplified={mg}, max_pts_agg_pred={mp}")
    print(f"GLOBAL_MAX_GT_SIMPLIFIED={global_max_gt}")
    print(f"GLOBAL_MAX_AGG_PRED={global_max_pred}")


if __name__ == "__main__":
    main()

