#!/usr/bin/env python
from __future__ import annotations

"""
Visual check: condition raster (10), proposals, and GT are aligned for training.

For each scene:
- Load static GT pickle to get canonical bounds [minx,miny,maxx,maxy].
- Load 10_render_gt.png (condition raster).
- Pack GT to fixed slots (GT white), pack proposals (colored) from aggregated
  vectors when available; optionally synthesize jittered proposals.
- Render two overlays per mode: letterbox 1024×1024 and raw-canvas size.
- Also write a slots overlay that emits a .meta.txt alongside the PNG (scale, W0×H0, bounds) for audit.

Usage (proposal mode for a list of scenes):
  python -u global_diffusion_map/refine/tools/viz_check_training_alignment.py \
    --static-root   maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --agg-pred-root maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid \
    --out-root      global_diffusion_map/refine/work_dirs/viz_training_alignment \
    --mode prop \
    --scenes <ID ...>
"""

import argparse
import os
import os.path as osp
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import pack_gt_to_slots, pack_vectors_to_slots
from global_diffusion_map.refine.single_scene_overfit import overlay_on_raster, overlay_slots_annot


def load_pickle(path: str):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def jitter_from_gt(gt_pack: np.ndarray, gt_mask: np.ndarray, shift_sigma: float, point_sigma: float) -> np.ndarray:
    x = gt_pack.copy()
    valid = ~gt_mask
    shifts = np.random.normal(scale=shift_sigma, size=(x.shape[0], 1, 2)).astype(np.float32)
    local = np.random.normal(scale=point_sigma, size=x.shape).astype(np.float32)
    noise = shifts + local
    x[valid] = np.clip(x[valid] + noise[valid], -1.0, 1.0)
    return x


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Visualize training alignment: condition 10 vs proposals vs GT')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', default='', help='Aggregated proposals root (per-scene pkl)')
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--mode', choices=['prop', 'synth', 'both'], default='prop')
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/viz_training_alignment')
    # synth noise defaults mirrored from training small/synth
    ap.add_argument('--shift-sigma', type=float, default=0.10)
    ap.add_argument('--point-sigma', type=float, default=0.02)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    import json
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20)); N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}

    os.makedirs(args.out_root, exist_ok=True)

    for sid in args.scenes:
        gt_pkl = osp.join(args.static_root, f'{sid}.pkl')
        if not osp.exists(gt_pkl):
            print(f'[skip] {sid}: missing static pkl')
            continue
        gt = load_pickle(gt_pkl)
        bounds = gt.get('bounds', None)
        if bounds is None or len(bounds) != 4:
            print(f'[skip] {sid}: static pkl has no bounds')
            continue
        gt_pack, gt_mask, _ = pack_gt_to_slots(gt, bounds, budgets, num_points=P, num_queries=N)

        # raster → CHW float [0,1]
        img_path = osp.join(args.rendered_root, sid, '10_render_gt.png')
        if not osp.exists(img_path):
            print(f'[skip] {sid}: missing 10_render_gt.png')
            continue
        ras = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0
        ras = ras.transpose(2, 0, 1)

        modes = ['prop'] if args.mode == 'prop' else (['synth'] if args.mode == 'synth' else ['prop', 'synth'])
        for m in modes:
            if m == 'prop':
                if not args.agg_pred_root:
                    print(f'[skip] {sid}: no agg-pred-root set; skip prop overlay')
                    continue
                ap = osp.join(args.agg_pred_root, f'{sid}.pkl')
                if not osp.exists(ap):
                    print(f'[skip] {sid}: missing agg proposal pkl {ap}')
                    continue
                agg = load_pickle(ap)
                x_prop, m_prop, _ = pack_vectors_to_slots(agg, bounds, budgets, num_points=P, num_queries=N)
                tag = 'prop'
                slots = x_prop
                slot_mask = m_prop
            else:  # synth
                x_jit = jitter_from_gt(gt_pack, gt_mask, shift_sigma=float(args.shift_sigma), point_sigma=float(args.point_sigma))
                tag = 'synth'
                slots = x_jit
                slot_mask = gt_mask

            out_dir = osp.join(args.out_root, sid)
            os.makedirs(out_dir, exist_ok=True)
            # Letterbox overlay (edges highlighted)
            overlay_on_raster(
                osp.join(out_dir, f'check_{tag}_letterbox.png'), ras, bounds,
                gt_pack, slots, cond_max_side=1024, cond_fixed_size=(1024, 1024), highlight_edges=True)
            # Raw canvas overlay
            overlay_on_raster(
                osp.join(out_dir, f'check_{tag}_raw.png'), ras, bounds,
                gt_pack, slots, cond_max_side=None, cond_fixed_size=None, highlight_edges=True)
            # Slots overlay that emits .meta.txt for audit (letterbox)
            overlay_slots_annot(
                osp.join(out_dir, f'check_{tag}_slots_letterbox.png'), ras, bounds,
                slots=slots, mask=slot_mask, labels=None, title=f'{sid} {tag}',
                cond_max_side=1024, cond_fixed_size=(1024, 1024), gt_slots=gt_pack, gt_mask=gt_mask)
            print(f'[ok] {sid}: wrote {tag} overlays under {out_dir}')


if __name__ == '__main__':
    main()
