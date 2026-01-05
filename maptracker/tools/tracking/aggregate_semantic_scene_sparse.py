#!/usr/bin/env python
"""
Sparse-majority aggregation of per-frame BEV semantic masks into a per-scene raster
that matches the aggregated prediction canvas (same coordinate extents, last-ego frame).

- Each frame's semantic mask is treated as sparse: only nonzero pixels are transformed.
- Pixels are mapped from source BEV pixels -> meters -> last-frame meters -> destination pixels.
- A per-pixel majority vote assigns the final class; low-vote pixels become blank.
- Optional static-GT footprint mask suppresses background bleed.
- Optional small-component removal reduces speckle when denoise is enabled.

Integrated outputs per scene (single pass):
- 06_agg_semantic.png: base vote thickness (optional via --save-06)
- 08_agg_semantic.png: uniform big-splat smoothing of 06 with continuity-aware overlap
- <out_dir>/<scene>.pkl with {'semantic_map': uint8[H, W]} from 06 (0=blank, 1..C)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mmcv
import numpy as np
from mmcv import Config
from shapely.geometry import Polygon
from shapely.ops import unary_union


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sparse aggregation of per-frame semantics (majority vote)")
    ap.add_argument("config", help="MapTracker config file")
    ap.add_argument("--submission-json", required=True, help="submission_vector.json (contains semantic_mask per token)")
    ap.add_argument("--pos-pkl", required=True, help="pos_predictions.pkl (frame metas with ego2global pose)")
    ap.add_argument("--out-dir", required=True, help="Output dir for per-scene maps")
    ap.add_argument("--scenes", nargs="*", default=None, help="Optional scene IDs to process (default: all)")
    ap.add_argument("--scene-list", default=None, help="File with one scene ID per line; used when --scenes not provided")
    ap.add_argument("--png", action="store_true", help="Save preview PNGs")
    # Bounds sources: match viz 01/04/05 axes
    ap.add_argument(
        "--bounds-pkl",
        nargs='+',
        default=[
            "maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/{scene}.pkl",
            "maptracker/work_dirs/agg_gt_vector/av2_oldsplit/{scene}.pkl",
        ],
        help="Bounds templates with {scene} (e.g. agg_pred and agg_gt pkls)",
    )
    ap.add_argument("--viz-root", default="maptracker/viz/av2_old", help="Viz root for 05_static_gt.png")
    ap.add_argument("--match-05", action="store_true", help="Match pixel size to viz/<scene>/05_static_gt.png if available")
    # Align ROI to 05/bounds: compute roi_size from sample mask size and 05/bounds pixel-per-meter so that px->meter matches 05
    ap.add_argument("--align-roi", action="store_true",
                    help="Compute roi_size from sample mask and 05/bounds so px->meter matches 05 canvas (overrides cfg roi_size)")
    # No ROI override/scaling or pixel-shift by default; use cfg.roi_size and meta transforms
    # Optional denoise/masking
    ap.add_argument("--mask-with-static", action="store_true", help="Mask output to static GT footprint")
    ap.add_argument("--static-root", default="maptracker/work_dirs/static_gt_vector/av2_oldsplit", help="Static GT pkl root")
    ap.add_argument("--min-votes", type=int, default=1, help="Minimum votes to assign a nonblank label")
    ap.add_argument("--min-area", type=int, default=8, help="Minimum connected-component area (px) to keep per class")
    ap.add_argument("--no-denoise", action="store_true", help="Disable morphological/area denoise (for debugging)")
    # Vote thickness for 06
    ap.add_argument("--thickness06-px", type=int, default=2, help="Vote splat radius (px) for 06 (>=1). 1=no splat")
    ap.add_argument("--save-06", action="store_true", help="Render 06_agg_semantic.png (off by default)")
    # Uniform big splat for 08 (simple dilation based on class map, previously 09)
    ap.add_argument("--splat08", type=int, default=4, help="Uniform dilation radius (px) to create 08_agg_semantic.png")
    ap.add_argument("--splat08-m", type=float, default=0.0, help="Uniform dilation line width in meters for 08 (radius = width/2). 0 disables")
    ap.add_argument("--splat08-block", type=int, default=0, help="Coarse unit block size (px) for 08; 0=auto from scale; 1=disable coarse")
    # Overlap boost (cyan) radius to thicken overlapped ped x boundary region in rendering
    ap.add_argument("--overlap-boost-px", type=int, default=0, help="Additional dilation radius (px) for overlap mask in 06/07/08 rendering")
    ap.add_argument("--overlap-boost-m", type=float, default=0.0, help="Additional dilation width (meters) for overlap mask; radius=width/2 (overrides px if >0)")
    # Continuity-aware overlap (ped edge ring near boundary)
    ap.add_argument("--overlap-edge-m", type=float, default=0.5, help="Ped edge ring thickness in meters for overlap continuity")
    ap.add_argument("--overlap-edge-px", type=int, default=0, help="Override edge ring thickness in pixels (if >0)")
    ap.add_argument("--overlap-prox-m", type=float, default=0.8, help="Boundary proximity threshold in meters for overlap continuity")
    ap.add_argument("--overlap-prox-px", type=int, default=0, help="Override boundary proximity threshold in pixels (if >0)")
    ap.add_argument("--overlap-close-px", type=int, default=1, help="Closing kernel radius (px) to connect tiny gaps in overlap continuity mask")
    ap.add_argument("--overlap-ring-mode", choices=["outer","inner","both"], default="outer", help="Use outer ped ring, inner ring, or both for overlap continuity")
    ap.add_argument("--overlap-density-k", type=int, default=2, help="Boundary density kernel radius (px) for overlap validation (window size=2k+1)")
    ap.add_argument("--overlap-density-min", type=int, default=3, help="Minimum boundary pixels within density window to keep overlap pixel")
    # 08 specific: reuse 06 overlap and extra smoothing
    ap.add_argument("--overlap-08-from-06", action="store_true", help="Use 06 continuity overlap for 08 rendering")
    ap.add_argument("--overlap08-close-px", type=int, default=2, help="Extra closing radius (px) for 08 overlap mask when using 06")
    ap.add_argument("--overlap08-dilate-px", type=int, default=1, help="Extra dilation radius (px) for 08 overlap mask when using 06")
    ap.add_argument("--overlap08-connect-m", type=float, default=2.0, help="08 overlap post-connect circle radius (meters)")
    ap.add_argument("--overlap08-connect-px", type=int, default=0, help="Override 08 overlap post-connect circle radius (pixels)")
    ap.add_argument("--overlap08-thin-px", type=int, default=0, help="Optional thinning (erosion) after connect to reduce overgrowth (px)")
    return ap.parse_args()


def _np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def get_prev2curr(prev_meta: dict, curr_meta: dict) -> np.ndarray:
    """Compute prev->curr 4x4 transform from per-frame metas (ego2global)."""
    prev_R, prev_t = _np(prev_meta["ego2global_rotation"]), _np(prev_meta["ego2global_translation"])
    curr_R, curr_t = _np(curr_meta["ego2global_rotation"]), _np(curr_meta["ego2global_translation"])
    curr_g2e = np.eye(4); curr_g2e[:3, :3] = curr_R.T; curr_g2e[:3, 3] = -(curr_R.T @ curr_t)
    prev_e2g = np.eye(4); prev_e2g[:3, :3] = prev_R; prev_e2g[:3, 3] = prev_t
    return curr_g2e @ prev_e2g


def build_token_maps(pos_list: Sequence[dict]) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    token2meta: Dict[str, dict] = {}
    scene2tokens: Dict[str, List[str]] = {}
    for rec in pos_list:
        tok = rec.get("meta", {}).get("token", rec.get("token"))
        if tok is None:
            tok = f"{rec['scene_name']}__{rec.get('local_idx', 0)}"
        token2meta[tok] = rec
        scene2tokens.setdefault(rec["scene_name"], []).append(tok)
    for s in scene2tokens:
        scene2tokens[s].sort(key=lambda t: token2meta[t].get("local_idx", 0))
    return token2meta, scene2tokens


def compute_union_bounds_last(
    frames: Sequence[dict], roi_size: Tuple[float, float], last_R2: np.ndarray, last_t2: np.ndarray
) -> Tuple[float, float, float, float]:
    half_w, half_h = roi_size[0] / 2.0, roi_size[1] / 2.0
    rect = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_h, half_h]], dtype=np.float64)
    # Correct rectangle (typo fix):
    rect = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]], dtype=np.float64)
    polys = []
    for rec in frames:
        rot = _np(rec["meta"]["ego2global_rotation"])[:2, :2]
        trans = _np(rec["meta"]["ego2global_translation"])[:2]
        world = (rot @ rect.T).T + trans
        last_xy = (last_R2.T @ (world - last_t2).T).T
        polys.append(Polygon(last_xy))
    uni = unary_union(polys)
    bx, by, Bx, By = uni.bounds
    return float(bx), float(Bx), float(by), float(By)


def compute_bounds_from_pkls(tpls: Sequence[str], scene: str) -> Optional[Tuple[float, float, float, float]]:
    """Prefer explicit 'bounds' in pkls; otherwise fall back to geometry extents.

    Note: If the template already includes a split (train/valid), do not expand
    to both splits. Only expand when the template is split-agnostic.
    """
    pts: List[np.ndarray] = []
    bounds_list: List[Tuple[float, float, float, float]] = []

    def _expand(path_tpl: str) -> List[str]:
        out = [path_tpl.format(scene=scene)]
        if "/av2_oldsplit/" in path_tpl and "{scene}" in path_tpl:
            base, tail = path_tpl.split("/av2_oldsplit/", 1)
            tl = tail.lower()
            # If user provided a split-specific path, do NOT expand
            if ("/train/" in tl) or ("/valid/" in tl) or tl.startswith("train/") or tl.startswith("valid/"):
                return out
            out.append(os.path.join(base, "av2_oldsplit", "train", f"{scene}.pkl"))
            out.append(os.path.join(base, "av2_oldsplit", "valid", f"{scene}.pkl"))
        return out

    for tpl in tpls:
        for p in _expand(tpl):
            if not os.path.exists(p):
                continue
            d = mmcv.load(p)
            b = d.get("bounds", None)
            if isinstance(b, (list, tuple)) and len(b) == 4:
                minx, miny, maxx, maxy = map(float, b)
                # Convert to (minx,maxx,miny,maxy)
                bounds_list.append((minx, maxx, miny, maxy))
                continue
            # Fallback: accumulate geometry extents
            for lbl in (0, 1, 2):
                for arr in d.get(lbl, []):
                    a = np.asarray(arr)
                    if a.size:
                        pts.append(a)
            for e in d.get("car_trajectory", []):
                c = _np(e.get("center", []))
                if c.size == 2:
                    pts.append(c.reshape(1, 2))

    if bounds_list:
        xs_min = min(b[0] for b in bounds_list)
        xs_max = max(b[1] for b in bounds_list)
        ys_min = min(b[2] for b in bounds_list)
        ys_max = max(b[3] for b in bounds_list)
        return float(xs_min), float(xs_max), float(ys_min), float(ys_max)

    if not pts:
        return None
    P = np.concatenate(pts, axis=0)
    return float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 1].min()), float(P[:, 1].max())


def get_05_shape(viz_root: str, scene: str) -> Optional[Tuple[int, int]]:
    # Try common layouts: <viz_root>/<scene>/05_static_gt.png, and split subdirs
    candidates = [
        os.path.join(viz_root, scene, "05_static_gt.png"),
        os.path.join(viz_root, "val", scene, "05_static_gt.png"),
        os.path.join(viz_root, "train", scene, "05_static_gt.png"),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            import imageio.v2 as imageio
            im = imageio.imread(p)
            return (im.shape[0], im.shape[1])
        except Exception:
            continue
    return None


def build_static_mask(static_pkl: str, W: int, H: int, minx: float, maxx: float, miny: float, maxy: float) -> Optional[np.ndarray]:
    if not os.path.exists(static_pkl):
        return None
    data = mmcv.load(static_pkl)
    mask = np.zeros((H, W), dtype=np.uint8)
    sx_px, sy_px = W / float(maxx - minx), H / float(maxy - miny)
    th = max(1, int(round(0.5 * (sx_px + sy_px))))
    # polygons (ped crossings)
    for arr in data.get(0, []):
        a = np.asarray(arr)
        if a.shape[0] >= 3:
            xs = (a[:, 0] - minx) * sx_px
            ys = (a[:, 1] - miny) * sy_px
            pts = np.stack([xs, ys], 1).round().astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    # polylines (divider/boundary)
    for lbl in (1, 2):
        for arr in data.get(lbl, []):
            a = np.asarray(arr)
            if a.shape[0] >= 2:
                xs = (a[:, 0] - minx) * sx_px
                ys = (a[:, 1] - miny) * sy_px
                pts = np.stack([xs, ys], 1).round().astype(np.int32)
                cv2.polylines(mask, [pts], False, 1, thickness=th)
    return mask


def render_png(label_map: np.ndarray, out_path: Path, overlap_mask: Optional[np.ndarray] = None) -> None:
    H, W = label_map.shape
    # white background; class colors BGR aligned to vis_global:
    # 1 (ped_crossing) = blue, 2 (divider) = red, 3 (boundary) = green
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    palette = {
        1: (255, 0, 0),   # blue in BGR
        2: (0, 0, 255),   # red in BGR
        3: (0, 255, 0),   # green in BGR
    }
    for k, col in palette.items():
        on = (label_map == k)
        if np.any(on):
            img[on] = col
    # Overlap: ped-crossing (1) and boundary (3)
    if overlap_mask is not None:
        ov = overlap_mask.astype(bool)
        if ov.shape == (H, W) and np.any(ov):
            # New color for overlap: cyan (blue+green)
            img[ov] = (255, 255, 0)
    # Do not flip: row index 0 corresponds to Y=maxy by construction,
    # matching the 01/04/05 rendering convention.
    cv2.imwrite(str(out_path), img)


def _disk_offsets(radius: int) -> List[Tuple[int, int]]:
    r = max(1, int(radius))
    offs: List[Tuple[int, int]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                offs.append((dy, dx))
    return offs


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    roi = tuple(cfg.get("roi_size", (60, 30)))

    # load sources
    with open(args.submission_json, "r") as f:
        submission = json.load(f)
    results = submission.get("results", {})
    pos_list = mmcv.load(args.pos_pkl)
    token2meta, scene2tokens = build_token_maps(pos_list)

    # token -> (C,H,W) mask
    token2mask: Dict[str, np.ndarray] = {}
    for tok, rec in results.items():
        m = rec.get("semantic_mask")
        if m is None:
            continue
        m = np.array(m, dtype=np.uint8)
        if m.ndim == 2:
            C = int(m.max())
            H, W = m.shape
            oh = np.zeros((C, H, W), dtype=np.uint8)
            for c in range(1, C + 1):
                oh[c - 1] = (m == c).astype(np.uint8)
            m = oh
        token2mask[tok] = m

    if args.scenes:
        scenes = list(args.scenes)
    elif args.scene_list and os.path.exists(args.scene_list):
        scenes = [ln.strip() for ln in open(args.scene_list, 'r').read().splitlines() if ln.strip()]
    else:
        scenes = sorted(scene2tokens.keys())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        toks = scene2tokens.get(scene, [])
        if not toks:
            continue
        last_meta = token2meta[toks[-1]]["meta"]
        last_R2 = _np(last_meta["ego2global_rotation"])[:2, :2]
        last_t2 = _np(last_meta["ego2global_translation"])[:2]

        # sample mask for size/density
        sample = None
        for t in toks:
            if t in token2mask:
                sample = token2mask[t]
                break
        if sample is None:
            continue
        C, H0, W0 = sample.shape

        # bounds: from pkls only; do not fall back silently
        b = compute_bounds_from_pkls(args.bounds_pkl, scene) if args.bounds_pkl else None
        if b is None:
            print(f"[error] {scene}: bounds pkls not found. Provide split-aware --bounds-pkl (e.g., .../av2_oldsplit/<split>/{scene}.pkl)")
            continue
        minx, maxx, miny, maxy = b
        if not (maxx > minx and maxy > miny):
            print(f"[warn] {scene}: invalid bounds {b}; skipping")
            continue

        # canvas size: must match 05 when requested; otherwise compute from scale
        # Precompute pixel-per-meter on the sample mask using cfg roi to avoid unbound sx0/sy0
        sx0, sy0 = W0 / float(roi[0]), H0 / float(roi[1])
        shape05 = get_05_shape(args.viz_root, scene) if args.match_05 else None
        if args.match_05 and not shape05:
            print(f"[error] {scene}: 05_static_gt.png not found under viz_root={args.viz_root}. Set correct --viz-root or pre-generate 05.")
            continue
        if shape05:
            H, W = shape05
        else:
            W = int(np.ceil(sx0 * (maxx - minx)))
            H = int(np.ceil(sy0 * (maxy - miny)))
            W = max(W, 1)
            H = max(H, 1)

        # Optionally align roi to 05/bounds so px->meter matches 05 (overrides cfg roi_size)
        if args.align_roi:
            # pixel-per-meter on 05 canvas
            Sx05 = W / float(maxx - minx)
            Sy05 = H / float(maxy - miny)
            # compute roi to make W0/roi_x == Sx05 and H0/roi_y == Sy05
            roi_x = float(W0) / max(Sx05, 1e-6)
            roi_y = float(H0) / max(Sy05, 1e-6)
            roi = (roi_x, roi_y)

        # Recompute pixel-per-meter if roi was updated (e.g., --align-roi)
        sx0, sy0 = W0 / float(roi[0]), H0 / float(roi[1])
        

        counts = np.zeros((C + 1, H, W), dtype=np.uint16)  # 0=blank
        cx, cy = W0 / 2.0, H0 / 2.0
        Sx = W / float(maxx - minx)
        Sy = H / float(maxy - miny)

        def aggregate_with_radius(r_vote_px: int) -> Tuple[np.ndarray, np.ndarray]:
            r_vote = int(max(1, r_vote_px))
            disk = _disk_offsets(r_vote)
            counts_local = np.zeros((C + 1, H, W), dtype=np.uint16)
            for t in toks:
                m = token2mask.get(t)
                if m is None:
                    continue
                prev_meta = token2meta[t]["meta"]
                P = get_prev2curr(prev_meta, last_meta)  # prev->last 4x4
                R = P[:2, :2]
                tvec = P[:2, 3]
                for c in range(C):
                    ys, xs = np.nonzero(m[c] > 0)
                    if ys.size == 0:
                        continue
                    # px -> meters in prev ego (y-up): center at (cx,cy), y flips sign
                    dx = (xs.astype(np.float64) - cx) / sx0
                    dy = (cy - ys.astype(np.float64)) / sy0
                    X = R[0, 0] * dx + R[0, 1] * dy + tvec[0]
                    Y = R[1, 0] * dx + R[1, 1] * dy + tvec[1]
                    # meters (last ego) -> dest pixels (row down): top corresponds to maxy
                    xd = np.round((X - minx) * Sx).astype(np.int32)
                    yd = np.round((maxy - Y) * Sy).astype(np.int32)
                    keep = (xd >= 0) & (xd < W) & (yd >= 0) & (yd < H)
                    if not np.any(keep):
                        continue
                    xd = xd[keep]
                    yd = yd[keep]
                    if r_vote <= 1:
                        np.add.at(counts_local[c + 1], (yd, xd), 1)
                    else:
                        # vote splat: add votes to a small disk around each pixel
                        for ddy, ddx in disk:
                            xk = xd + ddx
                            yk = yd + ddy
                            k = (xk >= 0) & (xk < W) & (yk >= 0) & (yk < H)
                            if np.any(k):
                                np.add.at(counts_local[c + 1], (yk[k], xk[k]), 1)
            votes_local = counts_local[1:].sum(axis=0)
            cls_local = counts_local[1:].argmax(axis=0) + 1  # 1..C
            cls_local[votes_local < max(1, int(args.min_votes))] = 0
            # Per-class support (any vote > 0)
            support = (counts_local[1:] > 0).astype(np.uint8)  # shape (C,H,W)
            return cls_local, support

        # Aggregate once to produce 06 (kept for optional rendering and overlap)
        cls06, sup06 = aggregate_with_radius(int(max(1, args.thickness06_px)))

        # Optional static mask to suppress background
        if args.mask_with_static:
            static_pkl = os.path.join(args.static_root, f"{scene}.pkl")
            sm = build_static_mask(static_pkl, W, H, minx, maxx, miny, maxy)
            if sm is not None:
                cls06[sm == 0] = 0

        # Optional denoise (skip when --no-denoise)
        if not args.no_denoise:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            def prune_small(cls_map: np.ndarray) -> np.ndarray:
                out = cls_map.copy()
                for c in range(1, C + 1):
                    m = (out == c).astype(np.uint8)
                    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                    keep = np.zeros(n, dtype=bool)
                    for i in range(1, n):
                        if stats[i, cv2.CC_STAT_AREA] >= int(args.min_area):
                            keep[i] = True
                    m_keep = (keep[labels]).astype(np.uint8)
                    out[(out == c) & (m_keep == 0)] = 0
                return out
            cls06 = prune_small(cls06)

        # Gap filling step removed per request

        # Bridging step removed per request (avoid filling sparse gaps)

        # Skip ped-crossing hollow outline post-process (reverted due to sparsity)

        # Debug summary
        nz = int((cls06 > 0).sum())
        total = cls06.size
        print(f"[dbg] {scene}: nonblank06={nz}/{total} ({nz*100.0/total:.2f}%)")

        out_scene = out_dir / scene
        out_scene.mkdir(parents=True, exist_ok=True)
        mmcv.dump({"semantic_map": cls06.astype(np.uint8)}, str(out_dir / f"{scene}.pkl"))
        if args.png:
            # Overlap mask: continuity-aware (ped edge ring near boundary)
            def _cont_overlap(sup: np.ndarray) -> Optional[np.ndarray]:
                if sup.shape[0] < 3:
                    return None
                ped = sup[0].astype(np.uint8)
                bnd = sup[2].astype(np.uint8)
                # Ring thickness
                if args.overlap_edge_px and args.overlap_edge_px > 0:
                    r_edge = int(max(1, args.overlap_edge_px))
                else:
                    spx = 0.5 * (Sx + Sy)
                    r_edge = int(max(1, round(0.5 * float(args.overlap_edge_m) * spx)))
                # Proximity threshold
                if args.overlap_prox_px and args.overlap_prox_px > 0:
                    prox_px = int(max(1, args.overlap_prox_px))
                else:
                    spx = 0.5 * (Sx + Sy)
                    prox_px = int(max(1, round(float(args.overlap_prox_m) * spx)))
                # Ped edge rings
                k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_edge + 1, 2 * r_edge + 1))
                ped_er = cv2.erode(ped, k_edge)
                ped_in = (ped > 0) & (ped_er == 0)   # inner ring
                ped_di = cv2.dilate(ped, k_edge)
                ped_out = (ped_di > 0) & (ped == 0)  # outer ring
                if args.overlap_ring_mode == "outer":
                    ped_edge = ped_out
                elif args.overlap_ring_mode == "inner":
                    ped_edge = ped_in
                else:
                    ped_edge = ped_out | ped_in
                # Distance to boundary (distance to nearest bnd pixel)
                inv_bnd = (bnd == 0).astype(np.uint8)
                dist = cv2.distanceTransform(inv_bnd, cv2.DIST_L2, 3)
                near = dist <= float(prox_px)
                ov = (ped_edge & near)
                # Clip overlap to ped area to avoid spill outside crossing
                ov = ov & (ped > 0)
                # Boundary density validation
                if args.overlap_density_k and args.overlap_density_k > 0 and args.overlap_density_min > 0:
                    ksz = 2 * int(args.overlap_density_k) + 1
                    ker = np.ones((ksz, ksz), dtype=np.uint8)
                    # convolve boundary mask (0/1)
                    bnd_u8 = (bnd > 0).astype(np.uint8)
                    dens = cv2.filter2D(bnd_u8, ddepth=-1, kernel=ker, borderType=cv2.BORDER_CONSTANT)
                    keep = dens >= int(args.overlap_density_min)
                    ov = ov & keep
                # Optional closing
                if args.overlap_close_px and args.overlap_close_px > 0:
                    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(args.overlap_close_px) + 1, 2 * int(args.overlap_close_px) + 1))
                    ov = cv2.morphologyEx(ov.astype(np.uint8), cv2.MORPH_CLOSE, k_close).astype(bool)
                return ov

            ov06 = _cont_overlap(sup06)
            # Optional boost (dilation) for overlap mask thickness
            def _boost(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if mask is None:
                    return None
                r = 0
                if args.overlap_boost_m and args.overlap_boost_m > 0:
                    spx = 0.5 * (Sx + Sy)
                    r = int(max(1, round(0.5 * float(args.overlap_boost_m) * spx)))
                else:
                    r = int(max(0, args.overlap_boost_px))
                if r <= 0:
                    return mask
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                return cv2.dilate(mask.astype(np.uint8), k).astype(bool)
            ov06 = _boost(ov06)
            if args.save_06:
                render_png(cls06, out_scene / "06_agg_semantic.png", overlap_mask=ov06)
            # Copy 05 for side-by-side comparison if present
            p05 = Path(args.viz_root) / scene / "05_static_gt.png"
            if p05.exists():
                try:
                    import shutil
                    shutil.copy(str(p05), str(out_scene / "05_static_gt.png"))
                except Exception:
                    pass
        # 08: Uniform dilation + erosion (closing) of class map (big splat with smoothing)
        if args.png:
            # Determine pixel radius from meters (if provided) or pixels
            if args.splat08_m and args.splat08_m > 0:
                spx = 0.5 * (Sx + Sy)
                r9_px = int(max(1, round(0.5 * float(args.splat08_m) * spx)))
            else:
                r9_px = max(1, int(args.splat08))
            # Pixel-level dilation (previous logic)
            k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r9_px + 1, 2 * r9_px + 1))
            dil = {c: np.zeros((H, W), dtype=np.uint8) for c in (1,2,3)}
            for c in (1,2,3):
                m0 = (cls06 == c).astype(np.uint8)
                if m0.max() == 0:
                    continue
                dil[c] = cv2.dilate(m0, k9)
            # Coarse unit erosion only
            B = int(args.splat08_block)
            if B == 0:
                scale_h = max(1.0, H / float(max(1, H0)))
                scale_w = max(1.0, W / float(max(1, W0)))
                B = int(round(max(scale_h, scale_w)))
                B = max(2, min(B, 24))
            elif B < 1:
                B = 1
            counts9 = np.zeros((C + 1, H, W), dtype=np.uint16)
            if B > 1:
                Hc, Wc = (H + B - 1)//B, (W + B - 1)//B
                r9c = int(max(1, round(r9_px / float(B))))
                k9c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r9c + 1, 2 * r9c + 1))
                for c in (1,2,3):
                    md = dil[c]
                    if md.max() == 0:
                        continue
                    mc = cv2.resize(md.astype(np.float32), (Wc, Hc), interpolation=cv2.INTER_AREA)
                    mc = (mc >= 0.5).astype(np.uint8)
                    e = cv2.erode(mc, k9c)
                    up = cv2.resize(e, (W, H), interpolation=cv2.INTER_NEAREST)
                    counts9[c] = up.astype(np.uint16)
            else:
                # Direct pixel-level erosion
                for c in (1,2,3):
                    md = dil[c]
                    if md.max() == 0:
                        continue
                    e = cv2.erode(md, k9)
                    counts9[c] = e.astype(np.uint16)
            votes9 = counts9[1:].sum(axis=0)
            cls9 = counts9[1:].argmax(axis=0) + 1
            cls9[votes9 == 0] = 0
            # Overlap for 08: use class presence after smoothing
            if args.overlap_08_from_06 and ov06 is not None:
                ov08 = ov06.copy().astype(bool)
                # Extra smoothing for 08 overlap to reduce curvature-induced breaks
                if args.overlap08_close_px and args.overlap08_close_px > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(args.overlap08_close_px) + 1, 2 * int(args.overlap08_close_px) + 1))
                    ov08 = cv2.morphologyEx(ov08.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
                if args.overlap08_dilate_px and args.overlap08_dilate_px > 0:
                    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(args.overlap08_dilate_px) + 1, 2 * int(args.overlap08_dilate_px) + 1))
                    ov08 = cv2.dilate(ov08.astype(np.uint8), k2).astype(bool)
                # Post-connect using circle radius to bridge broken outlines
                r_conn = 0
                if args.overlap08_connect_px and args.overlap08_connect_px > 0:
                    r_conn = int(args.overlap08_connect_px)
                else:
                    spx = 0.5 * (Sx + Sy)
                    r_conn = int(max(1, round(float(args.overlap08_connect_m) * spx)))
                if r_conn > 0:
                    kcon = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_conn + 1, 2 * r_conn + 1))
                    # Closing to connect nearby overlap edges/regions
                    ov08 = cv2.morphologyEx(ov08.astype(np.uint8), cv2.MORPH_CLOSE, kcon).astype(bool)
                    # Optional thinning to avoid overgrowth
                    if args.overlap08_thin_px and args.overlap08_thin_px > 0:
                        kth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(args.overlap08_thin_px) + 1, 2 * int(args.overlap08_thin_px) + 1))
                        ov08 = cv2.erode(ov08.astype(np.uint8), kth).astype(bool)
            else:
                if counts9.shape[0] > 3:
                    sup08 = np.zeros((3, H, W), dtype=np.uint8)
                    sup08[0] = (counts9[1] > 0).astype(np.uint8)  # ped
                    sup08[2] = (counts9[3] > 0).astype(np.uint8)  # boundary
                    ov08 = _cont_overlap(sup08)
                else:
                    ov08 = None
            # Boost overlap for 08 as well
            if ov08 is not None:
                r = 0
                if args.overlap_boost_m and args.overlap_boost_m > 0:
                    spx = 0.5 * (Sx + Sy)
                    r = int(max(1, round(0.5 * float(args.overlap_boost_m) * spx)))
                else:
                    r = int(max(0, args.overlap_boost_px))
                if r > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                    ov08 = cv2.dilate(ov08.astype(np.uint8), k).astype(bool)
            render_png(cls9.astype(np.uint8), out_scene / "08_agg_semantic.png", overlap_mask=ov08)
        print(f"[ok] semantic aggregated (08 rendered; 06 available): {str(out_dir / (scene + '.pkl'))}")
        


if __name__ == "__main__":
    main()
