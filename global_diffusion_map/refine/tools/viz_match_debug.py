#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import List, Sequence, Tuple

import numpy as np
import torch

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.single_scene_overfit import (
    load_pickle,
    load_raster_png,
    denorm_xy,
    letterbox,
    to_px,
)
from global_diffusion_map.refine.loss_refine import hungarian_match_perm


def _draw_poly(canvas: np.ndarray, pts_norm: np.ndarray, bounds: Sequence[float],
               color: Tuple[int, int, int], scale: float, W0: int, H0: int,
               thickness: int = 2, mark_dir: bool = True) -> None:
    import cv2
    if pts_norm.size == 0:
        return
    xy = denorm_xy(pts_norm, bounds)
    pts = to_px(xy, bounds, W0, H0)
    pts = (pts.astype(np.float32) * scale).round().astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts], False, (0, 0, 0), thickness=max(1, thickness + 2))
        cv2.polylines(canvas, [pts], False, color, thickness=max(1, thickness))
    # mark direction (start green, end red)
    if mark_dir and pts.shape[0] >= 1:
        cv2.circle(canvas, (int(pts[0, 0]), int(pts[0, 1])), 4, (0, 255, 0), -1)
        cv2.circle(canvas, (int(pts[-1, 0]), int(pts[-1, 1])), 4, (0, 0, 255), -1)


def main() -> None:
    ap = argparse.ArgumentParser(description='Visualize identity vs Hungarian matching with direction markers')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/refine/work_dirs/av2_stats.json')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--slot', type=int, default=-1, help='GT slot index to shift (default: first present)')
    ap.add_argument('--dx', type=float, default=-0.12, help='normalized X shift to apply to selected GT (left is negative)')
    ap.add_argument('--dy', type=float, default=0.08, help='normalized Y shift to apply to selected GT (up is positive)')
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/viz_match_debug')
    args = ap.parse_args()

    # Load caps/budgets
    stats = load_pickle(args.stats_json) if args.stats_json.endswith('.pkl') else None
    if stats is None:
        import json
        with open(args.stats_json, 'r') as f:
            stats = json.load(f)
    P = int(stats.get('M', 20)); N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_points=P, num_queries=N)

    # Load GT
    gt_pkl = osp.join(args.static_root, f'{args.scene}.pkl')
    gt = load_pickle(gt_pkl)
    bounds = gt.get('bounds')
    if bounds is None:
        raise RuntimeError('static GT lacks canonical bounds')
    gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, num_points=P, num_queries=N)

    # Choose a slot
    present_ids = np.where(gt_present > 0)[0]
    if present_ids.size == 0:
        raise RuntimeError('no present GT slots in this scene')
    slot = int(args.slot) if args.slot >= 0 else int(present_ids[0])
    if slot < 0 or slot >= N:
        raise RuntimeError(f'slot out of range: {slot}')

    # Build input by shifting the selected GT
    x_in = gt_pack.copy()
    x_in[slot, :, 0] = np.clip(x_in[slot, :, 0] + float(args.dx), -1.0, 1.0)
    x_in[slot, :, 1] = np.clip(x_in[slot, :, 1] + float(args.dy), -1.0, 1.0)

    # Identity orientation decision (min masked L1 between fwd vs rev)
    m = gt_mask[slot]
    v = (~m).astype(np.float32)
    l1_f = (np.abs(x_in[slot] - gt_pack[slot]).sum(-1) * v).sum() / max(v.sum(), 1.0)
    l1_r = (np.abs(x_in[slot] - gt_pack[slot][::-1]).sum(-1) * v).sum() / max(v.sum(), 1.0)
    use_rev_id = bool(l1_r < l1_f)
    id_tgt = (gt_pack[slot][::-1] if use_rev_id else gt_pack[slot])

    # Hungarian matching (perm-invariant) between single pred and all GT
    pred_t = torch.from_numpy(x_in[slot:slot+1]).float()  # [1,P,2]
    gt_t = torch.from_numpy(gt_pack).float()
    mask_t = torch.from_numpy(gt_mask).bool()
    pairs, perm_choice = hungarian_match_perm(pred_t, None, gt_t, mask_t, None, cls_weight=0.0, reg_weight=50.0, use_l1_beta=0.0)
    if pairs:
        _, gj = pairs[0]
        k = int(perm_choice[int(gj)]) if (perm_choice is not None and len(perm_choice) > int(gj)) else 0
        hung_tgt = gt_pack[gj].copy() if k == 0 else gt_pack[gj][::-1].copy()
    else:
        gj, k = -1, 0
        hung_tgt = gt_pack[slot].copy()

    # Raster and canvas
    ras = load_raster_png(osp.join(args.rendered_root, args.scene, '10_render_gt.png'))  # [3,H,W]
    canvas, scale, H0, W0 = letterbox(ras, 1024, (1024, 1024))

    import cv2
    # Viz 1: Identity decision vs GT and Input
    can1 = canvas.copy()
    _draw_poly(can1, gt_pack[slot], bounds, (255, 255, 255), scale, W0, H0, thickness=2, mark_dir=True)
    _draw_poly(can1, x_in[slot], bounds, (255, 0, 0), scale, W0, H0, thickness=2, mark_dir=True)   # blue (BGR)
    _draw_poly(can1, id_tgt, bounds, (0, 255, 255), scale, W0, H0, thickness=2, mark_dir=True)     # yellow
    cv2.putText(can1, f'Identity: {"rev" if use_rev_id else "fwd"} (slot {slot}) dx={args.dx:+.3f} dy={args.dy:+.3f}  L1f={l1_f:.4f} L1r={l1_r:.4f}',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(can1, f'Identity: {"rev" if use_rev_id else "fwd"} (slot {slot}) dx={args.dx:+.3f} dy={args.dy:+.3f}  L1f={l1_f:.4f} L1r={l1_r:.4f}',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

    # Viz 2: Hungarian decision vs GT and Input
    can2 = canvas.copy()
    _draw_poly(can2, x_in[slot], bounds, (255, 0, 0), scale, W0, H0, thickness=2, mark_dir=True)
    # Draw the matched GT instance (white baseline)
    if gj >= 0:
        _draw_poly(can2, gt_pack[gj], bounds, (255, 255, 255), scale, W0, H0, thickness=2, mark_dir=True)
    _draw_poly(can2, hung_tgt, bounds, (255, 255, 0), scale, W0, H0, thickness=2, mark_dir=True)  # cyan-ish
    cv2.putText(can2, f'Hungarian: gt {gj} perm={"rev" if k==1 else "fwd"}  dx={args.dx:+.3f} dy={args.dy:+.3f}',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(can2, f'Hungarian: gt {gj} perm={"rev" if k==1 else "fwd"}  dx={args.dx:+.3f} dy={args.dy:+.3f}',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1, cv2.LINE_AA)

    out_dir = osp.join(args.out_root, args.scene)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(osp.join(out_dir, f'identity_vs_gt_slot{slot:02d}.png'), can1)
    cv2.imwrite(osp.join(out_dir, f'hungarian_vs_gt_slot{slot:02d}.png'), can2)
    print('[ok] saved:', osp.join(out_dir, f'identity_vs_gt_slot{slot:02d}.png'))
    print('[ok] saved:', osp.join(out_dir, f'hungarian_vs_gt_slot{slot:02d}.png'))


if __name__ == '__main__':
    main()
