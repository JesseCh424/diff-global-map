#!/usr/bin/env python
"""
Visual check: verify conditioning raster (10/11) aligns with vectors after the
same downscale+letterbox used in training/inference.

Produces two PNGs under out_root/<scene>/:
- align_overlay_orig.png: overlay on original 10/11 canvas (pre-letterbox)
- align_overlay_letterbox.png: overlay on letterboxed canvas (post-letterbox)
Both use bounds→pixel mapping; the letterboxed version uses the exact combined
scale (cond_max_side pre-scale × letterbox scale), pasted at (0,0).

Usage example:
  python -u global_diffusion_map/tools/check_cond_alignment.py \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --cond-filename 10_render_gt.png \
    --scenes 02678d04-cc9f-3148-9f95-1ba66347dff9 \
    --out-root global_diffusion_map/viz/cond_align \
    --cond-max-side 1024 --cond-fixed-size 1024 1024
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import List, Tuple

import cv2
import mmcv
import cv2
import numpy as np


def letterbox(img: np.ndarray, cond_max_side: int | None, cond_fixed_size: Tuple[int, int] | None) -> Tuple[np.ndarray, float]:
    """Resize + letterbox to cond_fixed_size. Return canvas and scalar scale.
    Paste at (0,0) to mirror training dataset and inference.
    """
    h, w = img.shape[:2]
    if cond_max_side is not None:
        ms = max(h, w)
        if ms > 0 and ms > cond_max_side:
            s = float(cond_max_side) / float(ms)
            nh = max(1, int(round(h * s)))
            nw = max(1, int(round(w * s)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            h, w = img.shape[:2]
            s_pre = s
        else:
            s_pre = 1.0
    else:
        s_pre = 1.0
    s = 1.0
    content_size = (w, h)
    if cond_fixed_size is not None:
        tgt_h, tgt_w = int(cond_fixed_size[0]), int(cond_fixed_size[1])
        if h == 0 or w == 0:
            canvas = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)
            return canvas, 1.0
        s = min(float(tgt_w) / float(w), float(tgt_h) / float(h))
        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        if (nw, nh) != (w, h):
            img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        else:
            img_r = img
        canvas = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)
        canvas[:nh, :nw] = img_r
        return canvas, s_pre * s
    return img, s_pre * s


def to_px(arr: np.ndarray, bounds: List[float], W: int, H: int) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    Sx = W / max(maxx - minx, 1e-6)
    Sy = H / max(maxy - miny, 1e-6)
    xs = np.clip(np.round((arr[:, 0] - minx) * Sx), 0, W - 1)
    ys = np.clip(np.round((maxy - arr[:, 1]) * Sy), 0, H - 1)
    return np.stack([xs, ys], 1).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--cond-filename', default='10_render_gt.png')
    ap.add_argument('--scene', default=None)
    ap.add_argument('--scenes', nargs='*', default=None)
    ap.add_argument('--scene-list', default=None)
    ap.add_argument('--out-root', default='global_diffusion_map/viz/cond_align')
    ap.add_argument('--bounds-pred', default=None, help='Dir of aggregated predictions (<scene>.pkl) to compute bounds if static lacks bounds')
    ap.add_argument('--bounds-gt', default=None, help='Dir of aggregated GT (<scene>.pkl) to compute bounds if static lacks bounds')
    ap.add_argument('--cond-max-side', type=int, default=1024)
    ap.add_argument('--cond-fixed-size', type=int, nargs=2, default=[1024, 1024])
    ap.add_argument('--thickness-px', type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    # Build scene list
    scenes: List[str] = []
    if args.scene:
        scenes = [args.scene]
    elif args.scenes:
        scenes = list(args.scenes)
    elif args.scene_list and osp.exists(args.scene_list):
        with open(args.scene_list, 'r') as f:
            scenes = [ln.strip() for ln in f if ln.strip()]
    else:
        scenes = sorted([osp.splitext(x)[0] for x in os.listdir(args.static_root) if x.endswith('.pkl')])

    for s in scenes:
        pkl = osp.join(args.static_root, f'{s}.pkl')
        if not osp.exists(pkl):
            print(f'[skip] {s}: no static pkl')
            continue
        data = mmcv.load(pkl)
        bounds = data.get('bounds', None)
        if bounds is None:
            # Try compute from aggregated pred/gt pkls
            cand = []
            if args.bounds_pred:
                bp = osp.join(args.bounds_pred, f'{s}.pkl')
                if osp.exists(bp):
                    try:
                        d = mmcv.load(bp)
                        for k in (0,1,2):
                            for arr in d.get(k, []):
                                cand.append(np.asarray(arr))
                    except Exception:
                        pass
            if args.bounds_gt:
                bg = osp.join(args.bounds_gt, f'{s}.pkl')
                if osp.exists(bg):
                    try:
                        d = mmcv.load(bg)
                        for k in (0,1,2):
                            for arr in d.get(k, []):
                                cand.append(np.asarray(arr))
                    except Exception:
                        pass
            if len(cand) == 0:
                print(f'[skip] {s}: missing bounds and no bounds sources')
                continue
            cat = np.concatenate(cand, axis=0)
            minx, miny = cat.min(0)
            maxx, maxy = cat.max(0)
            bounds = [float(minx), float(miny), float(maxx), float(maxy)]
        img_path = osp.join(args.rendered_root, s, args.cond_filename)
        if not osp.exists(img_path):
            print(f'[skip] {s}: missing {img_path}')
            continue
        img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
        H0, W0 = img0.shape[:2]
        canvas, scale = letterbox(img0, args.cond_max_side, tuple(args.cond_fixed_size))
        # Overlay on letterboxed (post-normalization) and on original (pre-normalization)
        overlay = canvas.copy()
        overlay_orig = img0.copy()
        colors = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}  # BGR: ped, div, bnd
        for cls_id, color in colors.items():
            for arr in data.get(cls_id, []):
                a = np.asarray(arr)
                if a.shape[0] < 2:
                    continue
                H0, W0 = img0.shape[:2]
                pts = to_px(a, bounds, W0, H0)
                pts_s = (pts.astype(np.float32) * scale).round().astype(np.int32)
                cv2.polylines(overlay, [pts_s], cls_id == 0, color, thickness=max(1, args.thickness_px))
                # also draw on original-sized image for pre-letterbox check
                cv2.polylines(overlay_orig, [pts], cls_id == 0, color, thickness=max(1, args.thickness_px))
        out_dir = osp.join(args.out_root, s)
        os.makedirs(out_dir, exist_ok=True)
        # Clean legacy file name from older script versions
        legacy = osp.join(out_dir, 'align_overlay.png')
        try:
            if osp.exists(legacy):
                os.remove(legacy)
        except Exception:
            pass
        out_path1 = osp.join(out_dir, 'align_overlay_letterbox.png')
        out_path2 = osp.join(out_dir, 'align_overlay_orig.png')
        cv2.imwrite(out_path1, overlay)
        cv2.imwrite(out_path2, overlay_orig)
        print(f'[ok] {s}: wrote {out_path1} and {out_path2}')


if __name__ == '__main__':
    main()
