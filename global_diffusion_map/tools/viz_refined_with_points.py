#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import MapTracker's plot function directly
import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
VIS_MOD = osp.join(REPO_ROOT, 'maptracker', 'tools', 'visualization')
if VIS_MOD not in sys.path:
    sys.path.insert(0, VIS_MOD)
from vis_global import plot_fig_unmerged, compute_bounds_from_pkls  # type: ignore


def load_pkl(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_id_prev2curr(pred: Dict[int, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    bank: Dict[str, List[np.ndarray]] = {}
    for lbl in (0, 1, 2):
        arrs = pred.get(lbl, []) or []
        for i, poly in enumerate(arrs):
            key = f"{lbl}_{i}"
            bank[key] = [np.asarray(poly)]
    return bank


def main():
    ap = argparse.ArgumentParser(description='Visualize refined vectors using MapTracker plot (points visible).')
    ap.add_argument('--refined-root', required=True, help='Root of refined pickles (<scene>.pkl)')
    ap.add_argument('--out-root', required=True, help='Output dir for PNGs per scene')
    ap.add_argument('--scenes', nargs='+', required=True, help='Scene IDs to visualize')
    ap.add_argument('--bounds-pkl', nargs='*', default=None, help='Optional pkl(s) to derive shared bounds (e.g., aggregated preds)')
    ap.add_argument('--dpi', type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # Minimal args namespace expected by plot_fig_unmerged
    class _Args:
        def __init__(self, dpi: int):
            self.transparent = False
            self.dpi = dpi
    viz_args = _Args(args.dpi)

    # Compute global bounds if provided
    shared_bounds: Optional[Tuple[float, float, float, float]] = None
    if args.bounds_pkl:
        shared_bounds = compute_bounds_from_pkls(args.bounds_pkl)

    # Dummy trajectory (car icon anchor); not important for static render
    car_traj = [[np.array([0.0, 0.0]), 0.0]]

    for scene in args.scenes:
        in_pkl = osp.join(args.refined_root, f'{scene}.pkl')
        if not osp.exists(in_pkl):
            print(f'[skip] missing refined pkl: {in_pkl}')
            continue
        data = load_pkl(in_pkl)
        bank = to_id_prev2curr({k: v for k, v in data.items() if isinstance(k, int)})
        # Derive bounds
        if shared_bounds is not None:
            x_min, x_max, y_min, y_max = shared_bounds
        elif 'bounds' in data:
            minx, miny, maxx, maxy = data['bounds']
            x_min, x_max, y_min, y_max = float(minx), float(maxx), float(miny), float(maxy)
        else:
            # fallback from points
            pts = []
            for arrs in bank.values():
                for a in arrs:
                    a = np.asarray(a)
                    if a.size:
                        pts.append(a)
            if pts:
                cat = np.concatenate(pts, axis=0)
                x_min, x_max = float(cat[:, 0].min()), float(cat[:, 0].max())
                y_min, y_max = float(cat[:, 1].min()), float(cat[:, 1].max())
            else:
                x_min, x_max, y_min, y_max = -30, 30, -15, 15

        out_dir = osp.join(args.out_root, scene)
        os.makedirs(out_dir, exist_ok=True)
        out_png = osp.join(out_dir, '12_refined_points.png')
        try:
            plot_fig_unmerged(car_traj, x_min, x_max, y_min, y_max, out_png, bank, viz_args)
        except Exception as e:
            print(f'[warn] plot failed for {scene}: {e}')
        else:
            print(f'[ok] wrote {out_png}')


if __name__ == '__main__':
    main()

