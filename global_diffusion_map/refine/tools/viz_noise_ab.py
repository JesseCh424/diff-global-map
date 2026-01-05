#!/usr/bin/env python
from __future__ import annotations

"""
Visualize Plan A (GT-derived simulated proposal) and Plan B (real proposal + micro noise).

Outputs per scene under out-root/<scene>/:
  - 00_gt.png                       (GT only)
  - 01_propA.png                    (Plan-A proposal over GT)
  - 02_propA_xk.png                 (Plan-A xK over GT)
  - 10_real_prop.png                (Plan-B real proposal over GT)
  - 11_real_prop_noise.png          (Plan-B real proposal + micro noise over GT)
  - 12_real_prop_xk.png             (Plan-B xK over GT)

Notes
  - Proposal/XK are in normalized coords [-1,1]; GT used for overlay only; bounds from static pkl.
  - Jitter applies to valid points only; drop removes a fraction of GT instances; ghost injects random lines to empty slots.
  - xK  = (1-alpha) * proposal + alpha * N(0,1), clamped to [-1,1].
"""

import argparse
import os
import os.path as osp
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import sys
# Ensure repo top-level is on sys.path so that `global_diffusion_map.*` imports work
REPO_TOP = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_TOP not in sys.path:
    sys.path.insert(0, REPO_TOP)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots, pack_vectors_to_slots
from global_diffusion_map.refine.single_scene_overfit import overlay_on_raster, load_pickle
from global_diffusion_map.refine.augment import augment_planA_from_gt_numpy


def set_seed(s: int = 0) -> None:
    import torch
    import random as _r
    _r.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def list_scenes(static_root: str) -> List[str]:
    out = []
    for fn in os.listdir(static_root):
        if fn.endswith('.pkl'):
            out.append(fn[:-4])
    return sorted(out)


def _rand_ghost(P: int) -> np.ndarray:
    pts = np.random.uniform(-1.0, 1.0, size=(P, 2)).astype(np.float32)
    for k in range(1, P):
        pts[k] = 0.7 * pts[k] + 0.3 * pts[k - 1]
    return pts


def build_planA(static_pkl: str, caps: RefineCaps, budgets: Dict[int, int],
                jitter_sigma: float, drop_lo: float, drop_hi: float,
                ghosts_lo: int, ghosts_hi: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    gt = load_pickle(static_pkl)
    bounds = gt.get('bounds')
    gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, caps.num_points, caps.num_queries)
    # 使用与训练一致的增强函数
    prop, present = augment_planA_from_gt_numpy(
        gt_pack, gt_mask,
        jitter_sigma=jitter_sigma,
        drop_lo=drop_lo, drop_hi=drop_hi,
        ghosts_lo=ghosts_lo, ghosts_hi=ghosts_hi,
    )
    return prop, gt_pack, gt_mask, {'bounds': bounds, 'prop_present': present}


def build_planB(agg_pkl: str, caps: RefineCaps, budgets: Dict[int, int],
                jitter_sigma: float, bounds_fallback: Sequence[float] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    agg = load_pickle(agg_pkl)
    bounds = agg.get('bounds')
    if bounds is None and bounds_fallback is not None:
        bounds = list(map(float, bounds_fallback))
    # pack real proposal vectors into slots
    x, m, _labs = pack_vectors_to_slots(agg, bounds, budgets, caps.num_points, caps.num_queries)
    # micro noise: apply only to valid (non-masked) points
    if jitter_sigma > 0:
        noise = np.random.normal(scale=jitter_sigma, size=x.shape).astype(np.float32)
        present = ~m  # [N,P]
        x[present] = np.clip(x[present] + noise[present], -1.0, 1.0)
    return x, {'bounds': bounds}


def main() -> None:
    ap = argparse.ArgumentParser(description='Visualize Plan A (GT-sim) and Plan B (real proposal + micro noise) for one scene')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--aggregated-pred-root', required=False, default='')
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--scene', default='')
    ap.add_argument('--random', action='store_true')
    # Plan A params (align training defaults: jitter=0, drop∈[0.2,0.4], ghost∈[2,4])
    ap.add_argument('--jitter-a', type=float, default=0.0)
    ap.add_argument('--drop-lo', type=float, default=0.2)
    ap.add_argument('--drop-hi', type=float, default=0.4)
    ap.add_argument('--ghost-lo', type=int, default=2)
    ap.add_argument('--ghost-hi', type=int, default=4)
    # Plan B params
    ap.add_argument('--jitter-b', type=float, default=0.02, help='micro noise sigma for real proposal')
    # xK params
    ap.add_argument('--alpha', type=float, default=0.03)
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/viz_noise_ab')
    args = ap.parse_args()

    set_seed(0)
    with open(args.stats_json, 'r') as f:
        import json
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_queries=N, num_points=P)

    # pick scene
    if args.scene:
        scene = args.scene
    else:
        scenes = list_scenes(args.static_root)
        if not scenes:
            raise FileNotFoundError('no static gt pkl found')
        scene = random.choice(scenes) if args.random else scenes[0]

    out_dir = osp.join(args.out_root, scene)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] scene={scene}  out={out_dir}")

    # load raster for overlay
    from global_diffusion_map.refine.single_scene_overfit import load_raster_png
    raster = load_raster_png(osp.join(args.rendered_root, scene, '10_render_gt.png'))

    # Plan A
    propA, gt_pack, gt_mask, metaA = build_planA(
        osp.join(args.static_root, f'{scene}.pkl'), caps, budgets,
        jitter_sigma=args.jitter_a, drop_lo=args.drop_lo, drop_hi=args.drop_hi,
        ghosts_lo=args.ghost_lo, ghosts_hi=args.ghost_hi
    )
    boundsA = metaA['bounds']
    # xK for A
    xK_A = np.clip((1.0 - float(args.alpha)) * propA + float(args.alpha) * np.random.normal(size=propA.shape).astype(np.float32), -1.0, 1.0)
    # Viz（仅绘制 proposal 中“存在”的槽位；避免把填充/替换噪声一并画出）
    overlay_on_raster(osp.join(out_dir, '00_gt.png'), raster, boundsA, gt_pack, [])
    prop_present = metaA.get('prop_present', ~gt_mask.all(axis=1))
    predA = [propA[i] for i in range(propA.shape[0]) if bool(prop_present[i])]
    predA_xk = [xK_A[i] for i in range(xK_A.shape[0]) if bool(prop_present[i])]
    overlay_on_raster(osp.join(out_dir, '01_propA.png'), raster, boundsA, gt_pack, predA)
    overlay_on_raster(osp.join(out_dir, '02_propA_xk.png'), raster, boundsA, gt_pack, predA_xk)

    # Plan B (optional if aggregated exists)
    if args.aggregated_pred_root:
        agg_pkl = osp.join(args.aggregated_pred_root, f'{scene}.pkl')
        if osp.isfile(agg_pkl):
            propB_orig, metaB = build_planB(agg_pkl, caps, budgets, jitter_sigma=0.0, bounds_fallback=boundsA)
            propB = np.clip(propB_orig + np.random.normal(scale=float(args.jitter_b), size=propB_orig.shape).astype(np.float32), -1.0, 1.0)
            boundsB = metaB['bounds']
            xK_B = np.clip((1.0 - float(args.alpha)) * propB + float(args.alpha) * np.random.normal(size=propB.shape).astype(np.float32), -1.0, 1.0)
            overlay_on_raster(osp.join(out_dir, '10_real_prop.png'), raster, boundsB, gt_pack, [propB_orig[i] for i in range(propB_orig.shape[0])])
            overlay_on_raster(osp.join(out_dir, '11_real_prop_noise.png'), raster, boundsB, gt_pack, [propB[i] for i in range(propB.shape[0])])
            overlay_on_raster(osp.join(out_dir, '12_real_prop_xk.png'), raster, boundsB, gt_pack, [xK_B[i] for i in range(xK_B.shape[0])])
        else:
            print(f"[warn] aggregated proposal not found: {agg_pkl}")

    print(f"[ok] saved to {out_dir}")


if __name__ == '__main__':
    main()
