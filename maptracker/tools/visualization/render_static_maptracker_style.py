#!/usr/bin/env python
"""
Render static (city->anchor ego, ROI-clipped) scene GT with MapTracker's plot functions
to match the style of gt_merged.png for fair comparison.

Input per scene: maptracker/datasets/av2_scene_static_gt/<split>/<scene>.pkl
  Format: {0: [N0x2 arrays], 1: [N1x2 arrays], 2: [N2x2 arrays]}

Outputs per scene under --out-dir/<scene>/
  - static.png (raw static GT vectors, no temporal merge)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import mmcv
import sys
import os

# Make visualization modules importable whether run from repo root or maptracker/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(SCRIPT_DIR)          # .../maptracker/tools
PKG_ROOT = os.path.dirname(TOOLS_DIR)            # .../maptracker
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)
if PKG_ROOT not in sys.path:
    sys.path.append(PKG_ROOT)

# Reuse MapTracker's plot functions
try:
    from tools.visualization.vis_global import plot_fig_merged, plot_fig_unmerged
except ModuleNotFoundError:
    from visualization.vis_global import plot_fig_merged, plot_fig_unmerged
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:
    from tools.visualization.vis_global import compute_bounds_from_pkls as _compute_bounds_from_pkls
except ModuleNotFoundError:
    from visualization.vis_global import compute_bounds_from_pkls as _compute_bounds_from_pkls


def compute_bounds_from_pkls(paths: Optional[Sequence[str]]) -> Optional[Tuple[float, float, float, float]]:
    if not paths:
        return None
    return _compute_bounds_from_pkls(paths)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--static-root', required=True, help='root of per-scene static GT pkls')
    p.add_argument('--scene-list', required=True, help='txt with scene ids')
    p.add_argument('--out-dir', required=True, help='output dir for images')
    p.add_argument('--dpi', type=int, default=40)
    p.add_argument('--roi-json', default=None, help='optional JSON {scene:[xmin,xmax,ymin,ymax]} to fix axes + crop')
    p.add_argument('--simplify', type=float, default=0.5)
    p.add_argument('--line-opacity', type=float, default=0.75)
    p.add_argument(
        '--bounds-pkl',
        nargs='+',
        default=None,
        help='Optional aggregated scene pickle(s) to reuse plotting bounds.',
    )
    return p.parse_args()


class _Args:
    def __init__(self, simplify: float, line_opacity: float, dpi: int):
        self.simplify = simplify
        self.line_opacity = line_opacity
        self.transparent = False
        self.dpi = dpi


def to_maptracker_bank(scene_vecs: Dict[int, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """Map static vectors to MapTracker's id_prev2curr_pred_vectors structure.
    Key format: f"{label}_{global_id}"; value: list of polylines across frames.
    For static GT, we store a single polyline per id (one-frame equivalent).
    """
    bank: Dict[str, List[np.ndarray]] = {}
    for lbl, arrs in scene_vecs.items():
        for gid, poly in enumerate(arrs):
            key = f"{lbl}_{gid}"
            bank[key] = [np.asarray(poly)]
    return bank


def compute_bounds(bank: Dict[str, List[np.ndarray]], pad: float = 5.0):
    xs, ys = [], []
    for lines in bank.values():
        for line in lines:
            if len(line) == 0:
                continue
            xs.extend(line[:, 0].tolist())
            ys.extend(line[:, 1].tolist())
    if not xs:
        return -30, 30, -15, 15  # fallback
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def render_static_mt_style(
    bank: Dict[str, List[np.ndarray]],
    out_path: Path,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    mt_args: _Args,
    car_traj: List[List[np.ndarray]],
):
    """Render using MapTracker's plot_fig_unmerged (style only). No merging performed."""
    if not car_traj:
        car_traj = [[np.array([0.0, 0.0]), 0.0]]
    plot_fig_unmerged(car_traj, x_min, x_max, y_min, y_max, str(out_path), bank, mt_args)


def main():
    args = parse_args()
    scenes = [s.strip() for s in Path(args.scene_list).read_text().splitlines() if s.strip()]
    static_root = Path(args.static_root)
    out_root = Path(args.out_dir)
    mt_args = _Args(args.simplify, args.line_opacity, args.dpi)
    roi_json = mmcv.load(args.roi_json) if args.roi_json else None
    bounds_from_pkls = compute_bounds_from_pkls(args.bounds_pkl)

    for scene in scenes:
        pkl_path = static_root / f'{scene}.pkl'
        if not pkl_path.exists():
            print(f'[skip] static GT missing for scene {scene}: {pkl_path}')
            continue
        scene_vecs_raw = mmcv.load(str(pkl_path))
        # Normalize keys to int and values to np.ndarray lists
        scene_vecs = {}
        for k, arrs in scene_vecs_raw.items():
            if not isinstance(k, int):
                try:
                    ik = int(k)
                except (TypeError, ValueError):
                    # Ignore non-vector metadata such as car trajectories.
                    continue
            else:
                ik = k
            scene_vecs[ik] = [np.asarray(a) for a in arrs]

        car_traj = scene_vecs_raw.get("car_trajectory", [])
        if car_traj:
            car_traj_plot = [
                [np.asarray(entry["center"], dtype=np.float32), float(entry["yaw_deg"])]
                for entry in car_traj
            ]
        else:
            car_traj_plot = [[np.array([0.0, 0.0]), 0.0]]
        # Optional crop by ROI JSON
        if roi_json and scene in roi_json:
            xmin, xmax, ymin, ymax = roi_json[scene]
            crop_rect = (xmin, xmax, ymin, ymax)
            # Clip lines to rectangle
            from shapely.geometry import LineString, box
            rect = box(xmin, ymin, xmax, ymax)
            cropped_scene = {k: [] for k in scene_vecs}
            for lbl, arrs in scene_vecs.items():
                for a in arrs:
                    inter = LineString(a).intersection(rect)
                    if inter.is_empty:
                        continue
                    if inter.geom_type == 'MultiLineString':
                        for sub in inter.geoms:
                            if len(sub.coords) >= 2:
                                cropped_scene[lbl].append(np.asarray(sub.coords))
                    elif inter.geom_type == 'LineString' and len(inter.coords) >= 2:
                        cropped_scene[lbl].append(np.asarray(inter.coords))
            scene_vecs = cropped_scene
            bank = to_maptracker_bank(scene_vecs)
            x_min, x_max, y_min, y_max = crop_rect
        else:
            bank = to_maptracker_bank(scene_vecs)
            if bounds_from_pkls:
                x_min, x_max, y_min, y_max = bounds_from_pkls
            else:
                x_min, x_max, y_min, y_max = compute_bounds(bank)
        out_dir = out_root / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        static_path = str(out_dir / 'static.png')

        # Render static view using MapTracker plotting (style parity)
        try:
            render_static_mt_style(bank, Path(static_path), x_min, x_max, y_min, y_max, mt_args, car_traj_plot)
        except Exception as e:
            print(f"[warn] plot_fig_unmerged failed for {scene}: {e}; falling back to simple renderer")
            # Fallback to a minimal inline renderer using the same color mapping
            COLOR = {0: 'b', 1: 'r', 2: 'g'}
            fig = plt.figure(figsize=(int(abs(x_min) + abs(x_max)) + 10 , int(abs(y_min) + abs(y_max)) + 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(False)
            for key, lines in bank.items():
                if not lines:
                    continue
                lbl = int(key.split('_')[0])
                color = COLOR.get(lbl, 'k')
                for line in lines:
                    line = np.asarray(line)
                    if len(line) < 2:
                        continue
                    ax.plot(line[:, 0], line[:, 1], 'o-', color=color, linewidth=20, markersize=50, alpha=0.9)
            fig.savefig(static_path, dpi=mt_args.dpi, bbox_inches='tight')
            plt.close(fig)
        print(f'[ok] {scene}: {static_path}')


if __name__ == '__main__':
    main()
