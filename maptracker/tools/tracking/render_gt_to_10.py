#!/usr/bin/env python
"""
Render GT static vectors into 10_render_gt.png under
maptracker/work_dirs/rendered_gt/av2_oldsplit/val/<scene>/, aligned to 09 canvases.

Rules:
- Align canvas and coordinate to 09_agg_semantic.png (or 05 if 09 missing).
- No car trajectory or icon.
- Use similar line width as 09 (meter-based thickness by default).
- Overlap ped_crossing (blue) and boundary (green) edges in cyan.
- Fill ped_crossing interior: set remaining blank pixels inside polygons to blue;
  preserve overlapped boundary edges (do not overwrite cyan edges).

Inputs:
- --static-root: cropped static GT pkls (val)
- --aggregated-pred/--aggregated-gt: bounds for mapping meters→pixels
- --semantic-root: for 09 alignment (reads shape HxW if available)
- --out-root: base folder for 10 outputs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mmcv
import numpy as np

# Import helpers from visualization module
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(SCRIPT_DIR)
PKG_ROOT = os.path.dirname(TOOLS_DIR)
for p in (TOOLS_DIR, PKG_ROOT):
    if p not in sys.path:
        sys.path.append(p)

try:
    from tools.visualization.vis_global import compute_bounds_from_pkls as _compute_bounds_from_pkls
except ModuleNotFoundError:
    from visualization.vis_global import compute_bounds_from_pkls as _compute_bounds_from_pkls


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render 10_render_gt.png aligned to 09")
    ap.add_argument("--static-root", required=True, help="Static GT val root with <scene>.pkl")
    ap.add_argument("--aggregated-pred", required=True, help="Aggregated pred dir")
    ap.add_argument("--aggregated-gt", required=True, help="Aggregated GT dir")
    ap.add_argument("--semantic-root", required=True, help="Semantic root with 09_agg_semantic.png")
    ap.add_argument("--out-root", required=True, help="Output base (e.g., rendered_gt/av2_oldsplit/val)")
    ap.add_argument("--scenes", nargs="*", default=None, help="Optional scene IDs (default: pick 5 random)")
    ap.add_argument("--num-random", type=int, default=5, help="Number of random scenes when --scenes not set")
    # Default edge width: 3.0 m (meter-normalized)
    ap.add_argument("--thickness-m", type=float, default=3.0, help="Edge line width in meters (radius=width/2)")
    ap.add_argument("--thickness-px", type=int, default=0, help="Override px thickness (if >0)")
    # Note: previous augmentation options (--aug, --jag-*, --unsmooth-*) were removed.
    return ap.parse_args()


def compute_bounds_from_pkls(paths: Sequence[str]) -> Optional[Tuple[float, float, float, float]]:
    return _compute_bounds_from_pkls(paths)


def _static_bounds_strict(static_pkl: Path) -> Tuple[float, float, float, float]:
    """STRICT: only accept bounds from static GT pkl; no fallback.

    Returns (minx, maxx, miny, maxy). Raises on missing.
    """
    data = mmcv.load(str(static_pkl))
    b = data.get('bounds', None)
    if b is None or len(b) != 4:
        raise RuntimeError(f"bounds-missing: {static_pkl} has no canonical 'bounds' entry")
    minx, miny, maxx, maxy = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return (minx, maxx, miny, maxy)


def load_static(scene_pkl: Path) -> Dict:
    return mmcv.load(str(scene_pkl))


def get_canvas_shape(semantic_root: Path, viz_root: Path, scene: str) -> Optional[Tuple[int, int]]:
    # Prefer 09_agg_semantic.png in semantic_root
    p09 = semantic_root / scene / "08_agg_semantic.png"
    if p09.exists():
        try:
            import imageio.v2 as imageio
            im = imageio.imread(p09)
            return int(im.shape[0]), int(im.shape[1])
        except Exception:
            pass
    # Fallback to 05 in viz root
    p05 = viz_root / scene / "05_static_gt.png"
    if p05.exists():
        try:
            import imageio.v2 as imageio
            im = imageio.imread(p05)
            return int(im.shape[0]), int(im.shape[1])
        except Exception:
            pass
    return None


def to_px_coords(arr: np.ndarray, minx: float, maxx: float, miny: float, maxy: float, W: int, H: int) -> np.ndarray:
    Sx = W / float(maxx - minx)
    Sy = H / float(maxy - miny)
    xs = np.clip(np.round((arr[:, 0] - minx) * Sx), 0, W - 1)
    ys = np.clip(np.round((maxy - arr[:, 1]) * Sy), 0, H - 1)
    pts = np.stack([xs, ys], 1).astype(np.int32)
    return pts


def main() -> None:
    args = parse_args()
    static_root = Path(args.static_root)
    sem_root = Path(args.semantic_root)
    out_base = Path(args.out_root)
    viz_root = Path("maptracker/viz/av2_old")  # used only for fallback 05 size

    all_scenes = [p.stem for p in static_root.glob("*.pkl")]
    if not all_scenes:
        print(f"[err] no scenes in {static_root}")
        return
    if args.scenes:
        scenes = args.scenes
    else:
        import random
        scenes = random.sample(all_scenes, k=min(args.num_random, len(all_scenes)))

    for scene in scenes:
        static_pkl = static_root / f"{scene}.pkl"
        if not static_pkl.exists():
            print(f"[skip] {scene}: missing static pkl")
            continue
        shape = get_canvas_shape(sem_root, viz_root, scene)
        if shape is None:
            print(f"[skip] {scene}: missing 09/05 for shape")
            continue
        H, W = shape
        # STRICT: only read bounds from static pkl; error if missing
        try:
            bounds = _static_bounds_strict(static_pkl)
        except Exception as e:
            print(f"[skip] {scene}: {e}")
            continue
        # Our internal format uses (minx, maxx, miny, maxy)
        minx, maxx, miny, maxy = bounds
        Sx = W / float(maxx - minx)
        Sy = H / float(maxy - miny)
        if args.thickness_px > 0:
            r_px = max(1, int(args.thickness_px))
        else:
            spx = 0.5 * (Sx + Sy)
            r_px = int(max(1, round(0.5 * float(args.thickness_m) * spx)))

        data = load_static(static_pkl)
        img = np.ones((H, W, 3), dtype=np.uint8) * 255
        ped_edge = np.zeros((H, W), dtype=np.uint8)
        bnd_edge = np.zeros((H, W), dtype=np.uint8)
        div_edge = np.zeros((H, W), dtype=np.uint8)

        # Draw edges with cv2
        for lbl in (1, 2):  # 1 divider (red), 2 boundary (green)
            for arr in data.get(lbl, []):
                a = np.asarray(arr)
                if a.shape[0] < 2:
                    continue
                pts = to_px_coords(a, minx, maxx, miny, maxy, W, H)
                if lbl == 1:
                    cv2.polylines(div_edge, [pts], False, 1, thickness=r_px)
                else:
                    cv2.polylines(bnd_edge, [pts], False, 1, thickness=r_px)

        # Ped polygons: edges + optional fills
        ped_fills: List[np.ndarray] = []
        for arr in data.get(0, []):
            a = np.asarray(arr)
            if a.shape[0] < 3:
                continue
            pts = to_px_coords(a, minx, maxx, miny, maxy, W, H)
            ped_fills.append(pts)
            cv2.polylines(ped_edge, [pts], True, 1, thickness=r_px)

        # Augmentation removed: edges are rendered cleanly without post-processing

        # Build class masks after augmentation
        ped_mask = (ped_edge > 0).astype(np.uint8)
        if ped_fills:
            ped_fill_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(ped_fill_mask, ped_fills, 1)
            ped_mask = np.maximum(ped_mask, ped_fill_mask)
        bnd_mask = (bnd_edge > 0).astype(np.uint8)
        div_mask = (div_edge > 0).astype(np.uint8)

        # Augmentation removed: masks are used as-is

        # Overlap mask and coloring
        ov = (ped_mask > 0) & (bnd_mask > 0)
        img[ov] = (255, 255, 0)
        ped_only = (ped_mask > 0) & (~ov)
        img[ped_only] = (255, 0, 0)
        bnd_only = (bnd_mask > 0) & (~ov)
        img[bnd_only] = (0, 255, 0)
        img[div_mask > 0] = (0, 0, 255)

        # Write output
        out_dir = out_base / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "10_render_gt.png"
        cv2.imwrite(str(out_path), img)
        print(f"[ok] {scene}: {out_path}")


if __name__ == "__main__":
    main()
