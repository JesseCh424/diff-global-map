#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.single_scene_overfit import (
    load_pickle,
    load_raster_png,
    overlay_slots_annot,
    denorm_xy,
    letterbox,
    to_px,
)
from global_diffusion_map.refine.loss_refine import hungarian_match_perm
# reuse the same jitter/drop/ghost logic as training-one-scene
from global_diffusion_map.refine.clean.train_one_scene_clean import jitter_drop_ghost  # type: ignore


def _labels_from_budgets(budgets: Dict[int, int], N: int) -> np.ndarray:
    # MapTR orig ids: divider=1, ped=0, boundary=2 -> labels 0,1,2 respectively
    order: List[int] = []
    for orig in (1, 0, 2):
        cap = int(budgets.get(orig, 0))
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        order += [lab] * max(0, cap)
    out = np.full((N,), -1, dtype=np.int64)
    m = min(N, len(order))
    if m > 0:
        out[:m] = np.asarray(order[:m], dtype=np.int64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Class-wise visualization of all instances (3 PNGs)')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/refine/work_dirs/av2_stats.json')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/viz_classwise')
    ap.add_argument('--thickness', type=int, default=2)
    # Shift mode params
    ap.add_argument('--dx', type=float, default=-0.12, help='normalized X shift to apply to GT as input for viz')
    ap.add_argument('--dy', type=float, default=0.08, help='normalized Y shift to apply to GT as input for viz')
    # Train-noise mode params (match training one-scene jitter)
    ap.add_argument('--use-train-noise', action='store_true', help='use training jitter/drop/ghost to generate input from GT')
    ap.add_argument('--shift-sigma', type=float, default=0.10, help='global per-line shift sigma for jitter')
    ap.add_argument('--point-sigma', type=float, default=0.02, help='local point sigma for jitter')
    ap.add_argument('--drop-frac', type=float, default=0.15, help='drop fraction for jitter (create)')
    ap.add_argument('--ghosts', type=int, default=2, help='num ghost slots to add')
    args = ap.parse_args()

    # Load stats
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20)); N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    _ = RefineCaps(num_points=P, num_queries=N)

    # Load static GT
    gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
    bounds = gt.get('bounds')
    if bounds is None:
        raise RuntimeError('static GT lacks canonical bounds')
    gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, num_points=P, num_queries=N)
    labels = _labels_from_budgets(budgets, N)  # per-slot class labels (0/1/2)

    # Raster
    ras = load_raster_png(osp.join(args.rendered_root, args.scene, '10_render_gt.png'))  # [3,H,W]

    # Build input proposal from GT for visualization
    if bool(getattr(args, 'use_train_noise', False)):
        # Use same jitter as training-one-scene
        x_in, keep_identity_np, is_drop_np, is_ghost_np = jitter_drop_ghost(
            gt_pack, gt_mask,
            shift_sigma=float(getattr(args, 'shift_sigma', 0.10)),
            point_sigma=float(getattr(args, 'point_sigma', 0.02)),
            drop_frac=float(getattr(args, 'drop_frac', 0.15)),
            ghosts=int(getattr(args, 'ghosts', 2)),
        )
        keep_identity = keep_identity_np.astype(bool)
    else:
        x_in = gt_pack.copy()
        x_in[:, :, 0] = np.clip(x_in[:, :, 0] + float(args.dx), -1.0, 1.0)
        x_in[:, :, 1] = np.clip(x_in[:, :, 1] + float(args.dy), -1.0, 1.0)
        # Without train-noise, treat all present as identity for the identity canvas
        keep_identity = (gt_present > 0)

    # For each class, build a mask that hides all non-class slots
    class_names = {0: 'divider', 1: 'ped', 2: 'boundary'}
    for cls in (0, 1, 2):
        mask = np.ones((N, P), dtype=bool)
        for i in range(N):
            if int(gt_present[i]) > 0 and int(labels[i]) == cls:
                mask[i, :] = False  # show this slot
        out_dir = osp.join(args.out_root, args.scene)
        os.makedirs(out_dir, exist_ok=True)
        # Build instance list for this class
        cls_ids = [i for i in range(N) if (int(gt_present[i]) > 0 and int(labels[i]) == cls)]
        # Helper to draw polylines with start/end markers
        def _draw_poly(canvas: np.ndarray, pts_norm: np.ndarray, color: Tuple[int, int, int],
                       bounds: Sequence[float], scale: float, W0: int, H0: int,
                       thickness: int = 2) -> None:
            import cv2
            if pts_norm.size == 0:
                return
            xy = denorm_xy(pts_norm, bounds)
            pts = to_px(xy, bounds, W0, H0)
            pts = (pts.astype(np.float32) * scale).round().astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, (0, 0, 0), thickness=max(1, thickness + 2))
                cv2.polylines(canvas, [pts], False, color, thickness=max(1, thickness))
            if pts.shape[0] >= 1:
                cv2.circle(canvas, (int(pts[0, 0]), int(pts[0, 1])), 3, (0, 255, 0), -1)
                cv2.circle(canvas, (int(pts[-1, 0]), int(pts[-1, 1])), 3, (0, 0, 255), -1)

        # Prepare canvas
        import cv2
        canvas_id, scale, H0, W0 = letterbox(ras, 1024, (1024, 1024))
        canvas_hg = canvas_id.copy()
        # Identity debug: draw GT (class) and input + identity-chosen orientation for each instance
        for i in cls_ids:
            _draw_poly(canvas_id, gt_pack[i], (255, 255, 255), bounds, scale, W0, H0, args.thickness)  # GT white
        for i in cls_ids:
            # input (blue)
            _draw_poly(canvas_id, x_in[i], (255, 0, 0), bounds, scale, W0, H0, args.thickness)
            # identity orientation decision only for identity slots
            if bool(keep_identity[i]):
                m = gt_mask[i]
                v = (~m).astype(np.float32)
                l1f = (np.abs(x_in[i] - gt_pack[i]).sum(-1) * v).sum() / max(v.sum(), 1.0)
                l1r = (np.abs(x_in[i] - gt_pack[i][::-1]).sum(-1) * v).sum() / max(v.sum(), 1.0)
                use_rev = bool(l1r < l1f)
                g_final = gt_pack[i][::-1] if use_rev else gt_pack[i]
                # chosen identity orientation (yellow)
                _draw_poly(canvas_id, g_final, (0, 255, 255), bounds, scale, W0, H0, args.thickness)
        out_png_id = osp.join(out_dir, f'class_{cls}_{class_names.get(cls, cls)}_identity.png')
        id_title = (f'class={cls} ({class_names.get(cls, cls)})  Identity ' +
                    (f'dx={args.dx:+.3f} dy={args.dy:+.3f}' if not args.use_train_noise else
                     f'jitter shift={args.shift_sigma:.2f} point={args.point_sigma:.2f} drop={args.drop_frac:.2f} ghosts={int(args.ghosts)}'))
        cv2.putText(canvas_id, id_title,
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas_id, id_title,
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
        cv2.imwrite(out_png_id, canvas_id)
        print('[ok] saved:', out_png_id)

        # Hungarian debug: draw GT class baseline, then input (blue) and matched orientation (cyan)
        for i in cls_ids:
            _draw_poly(canvas_hg, gt_pack[i], (255, 255, 255), bounds, scale, W0, H0, args.thickness)
        # run Hungarian per instance against all GT
        gt_t = torch.from_numpy(gt_pack).float()
        mask_t = torch.from_numpy(gt_mask).bool()
        for i in cls_ids:
            pred_t = torch.from_numpy(x_in[i:i+1]).float()
            pairs, perm_choice = hungarian_match_perm(pred_t, None, gt_t, mask_t, None,
                                                      cls_weight=0.0, reg_weight=50.0, use_l1_beta=0.0)
            gj = pairs[0][1] if pairs else i
            k = int(perm_choice[int(gj)]) if (perm_choice is not None and len(perm_choice) > int(gj)) else 0
            # input (blue)
            _draw_poly(canvas_hg, x_in[i], (255, 0, 0), bounds, scale, W0, H0, args.thickness)
            # matched orientation (cyan)
            g_sel = gt_pack[gj] if k == 0 else gt_pack[gj][::-1]
            _draw_poly(canvas_hg, g_sel, (255, 255, 0), bounds, scale, W0, H0, args.thickness)
        out_png_hg = osp.join(out_dir, f'class_{cls}_{class_names.get(cls, cls)}_hungarian.png')
        hg_title = (f'class={cls} ({class_names.get(cls, cls)})  Hungarian ' +
                    (f'dx={args.dx:+.3f} dy={args.dy:+.3f}' if not args.use_train_noise else
                     f'jitter shift={args.shift_sigma:.2f} point={args.point_sigma:.2f} drop={args.drop_frac:.2f} ghosts={int(args.ghosts)}'))
        cv2.putText(canvas_hg, hg_title,
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas_hg, hg_title,
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1, cv2.LINE_AA)
        cv2.imwrite(out_png_hg, canvas_hg)
        print('[ok] saved:', out_png_hg)


if __name__ == '__main__':
    main()
