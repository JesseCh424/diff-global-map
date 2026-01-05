#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
from typing import Dict, List

import numpy as np

# Reuse MapTracker renderer (same style as 05/09)
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MT_RENDER = REPO_ROOT / 'maptracker' / 'tools' / 'visualization' / 'render_static_maptracker_style.py'
if str(MT_RENDER.parent) not in sys.path:
    sys.path.insert(0, str(MT_RENDER.parent))
from render_static_maptracker_style import render_static_mt_style, _Args as MtArgs, to_maptracker_bank


def _load_pkl(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(description='Render refined vectors + input gt_simp + condition into aligned PNGs')
    ap.add_argument('--refined-root', required=True, help='Refined pickles root (from inference), <scene>.pkl')
    ap.add_argument('--gt-simp-root', required=True, help='GT simplified pickles root, <scene>.pkl')
    ap.add_argument('--condition-root', required=True, help='Root containing condition PNGs per scene')
    ap.add_argument('--cond-filename', default='11_gt_aug.png', help='Condition filename under each scene (e.g., 10_render_gt.png or 11_gt_aug.png)')
    ap.add_argument('--out-root', required=True, help='Output viz root; per-scene subfolders created')
    ap.add_argument('--input-name', required=False, default='09_static_gt_simp.png', help='Filename for input panel (default: 09_static_gt_simp.png)')
    ap.add_argument('--scenes', nargs='+', default=None, help='Optional scene ids list; if omitted, scan refined-root')
    ap.add_argument('--dpi', type=int, default=60)
    args = ap.parse_args()

    scenes = args.scenes
    if not scenes:
        scenes = sorted([osp.splitext(x)[0] for x in os.listdir(args.refined_root) if x.endswith('.pkl')])

    for scene in scenes:
        ref_pkl = osp.join(args.refined_root, f'{scene}.pkl')
        gt_pkl = osp.join(args.gt_simp_root, f'{scene}.pkl')
        cond_png = osp.join(args.condition_root, scene, args.cond_filename)
        if not osp.exists(ref_pkl):
            print(f'[skip] refined missing: {ref_pkl}')
            continue
        if not osp.exists(gt_pkl):
            print(f'[skip] gt_simp missing: {gt_pkl}')
            continue
        # Load
        ref = _load_pkl(ref_pkl)
        gt = _load_pkl(gt_pkl)
        bounds = ref.get('bounds') or gt.get('bounds')
        if not bounds or len(bounds) != 4:
            # fallback bounds from data points
            def _pts(data: Dict[int, List[np.ndarray]]):
                arrs = []
                for k in (0, 1, 2):
                    for a in data.get(k, []):
                        arrs.append(np.asarray(a))
                if not arrs:
                    return [0, 0, 1, 1]
                cat = np.concatenate(arrs, axis=0)
                mn = cat.min(0)
                mx = cat.max(0)
                return [float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])]
            bounds = _pts(ref)

        minx, miny, maxx, maxy = [float(v) for v in bounds]
        out_dir = osp.join(args.out_root, scene)
        os.makedirs(out_dir, exist_ok=True)

        # Render refined
        ref_bank = to_maptracker_bank({k: [np.asarray(a) for a in v] for k, v in ref.items() if isinstance(k, int)})
        mt_args = MtArgs(simplify=0.0, line_opacity=0.95, dpi=args.dpi)
        out_ref = Path(out_dir) / '12_refined.png'
        render_static_mt_style(ref_bank, out_ref, minx, maxx, miny, maxy, mt_args, car_traj=[])
        print(f'[ok] {scene}: {out_ref}')

        # Render input gt_simp
        gt_bank = to_maptracker_bank({k: [np.asarray(a) for a in v] for k, v in gt.items() if isinstance(k, int)})
        out_gt = Path(out_dir) / args.input_name
        render_static_mt_style(gt_bank, out_gt, minx, maxx, miny, maxy, mt_args, car_traj=[])
        print(f'[ok] {scene}: {out_gt}')

        # Copy condition PNG as-is; fallback to alternative if requested file missing
        if not osp.exists(cond_png):
            alt = '10_render_gt.png' if args.cond_filename != '10_render_gt.png' else '11_gt_aug.png'
            alt_path = osp.join(args.condition_root, scene, alt)
            if osp.exists(alt_path):
                cond_png = alt_path
        if osp.exists(cond_png):
            dst = osp.join(out_dir, '11_gt_aug.png')
            try:
                # Prefer hardlink to save space; fall back to copy
                if osp.exists(dst):
                    os.remove(dst)
                os.link(cond_png, dst)
            except Exception:
                import shutil
                shutil.copy2(cond_png, dst)
            print(f'[ok] {scene}: {dst}')
        else:
            print(f'[warn] condition missing under {args.condition_root}/{scene}: tried {args.cond_filename} and fallback')


if __name__ == '__main__':
    main()
