#!/usr/bin/env python
"""
Visualize ProposalEncoder conditioning strength per query.

Outputs per scene:
- proposals_overlay.png : proposals over letterboxed raster (10/11) with top-K
  queries (by |gamma-1| mean or beta-norm) highlighted.
- stats.txt : per-class mean conditioning magnitude and valid counts.

Usage:
  python -u global_diffusion_map/tools/viz_proposal_condition.py \
    --config global_diffusion_map/plugin/configs/global_diffusion/av2_polydiffuse_official_base.py \
    --ckpt <denoise_snapshot.pth> \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --proposal-root global_diffusion_map/work_dirs/packed_proposals/av2_oldsplit/val \
    --scenes <SCENE1> <SCENE2> ... \
    --out-root global_diffusion_map/viz/prop_cond \
    --cond-filename 10_render_gt.png
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import json
from typing import List, Tuple

import cv2
import numpy as np
from mmcv import Config
from mmdet3d.models import build_model


def letterbox(img: np.ndarray, cond_max_side: int | None, cond_fixed_size: Tuple[int, int] | None) -> tuple[np.ndarray, float]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--proposal-root', required=True)
    ap.add_argument('--scenes', nargs='*', required=True)
    ap.add_argument('--out-root', default='global_diffusion_map/viz/prop_cond')
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--cond-filename', default='10_render_gt.png')
    ap.add_argument('--cond-max-side', type=int, default=1024)
    ap.add_argument('--cond-fixed-size', type=int, nargs=2, default=[1024, 1024])
    ap.add_argument('--topk', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # Load stats: M, num_queries
    with open(args.stats_json, 'r') as f:
        s = json.load(f)
    M = int(s.get('M', 20))
    num_queries = int(s.get('num_queries', 50))

    # Build model (MapTR only; not running EDM sampler)
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin') and cfg.plugin:
        assert hasattr(cfg, 'plugin_dir')
        import importlib
        module_path = cfg.plugin_dir.replace('/', '.')[:-1] if cfg.plugin_dir.endswith('/') else cfg.plugin_dir.replace('/', '.')
        importlib.import_module(module_path)
    model = build_model(cfg.model)
    # Load denoiser weights into MapTR backbone/head (state_dict key 'net') if compatible
    try:
        sd = __import__('torch').load(args.ckpt, map_location='cpu')
        state = sd.get('net', sd)
        model.load_state_dict(state, strict=False)
    except Exception:
        pass
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    for scene in args.scenes:
        # Load condition (10/11)
        img_path = osp.join(args.rendered_root, scene, args.cond_filename)
        img0 = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img0 is None:
            print(f'[skip] {scene}: missing {img_path}')
            continue
        canvas, scale = letterbox(img0, args.cond_max_side, tuple(args.cond_fixed_size))

        # Load packed proposals
        npz = osp.join(args.proposal_root, f'{scene}.npz')
        if not osp.exists(npz):
            print(f'[skip] {scene}: missing {npz}')
            continue
        arr = np.load(npz)
        pts = arr['pts'].astype(np.float32)      # [NQ,M,2] normalized
        mask = arr['mask'].astype(bool)          # [NQ,M]
        labels = arr['labels'].astype(np.int64)  # [NQ]
        # Build model kwargs
        labs = labels[None, ...]
        msk = mask[None, ...]
        pts_b = pts[None, ...]
        H, W = canvas.shape[:2]
        meta = [[{'img_shape': [(H, W)], 'lidar2img': np.eye(4, dtype=np.float32)[None, ...], 'can_bus': np.zeros(18, dtype=np.float32)}]]
        with __import__('torch').no_grad():
            x_t = __import__('torch').zeros((1, num_queries, M, 2), dtype=__import__('torch').float32, device=device)
            t = __import__('torch').zeros((1,), dtype=__import__('torch').float32, device=device)
            model(
                x_t, t,
                img=__import__('torch').from_numpy(canvas.transpose(2,0,1)[None,None,...]/255.0).to(device),
                poly_class=__import__('torch').from_numpy(labs).to(device),
                poly_mask=__import__('torch').from_numpy(msk).to(device),
                img_metas=meta,
                proposal_pts=__import__('torch').from_numpy(pts_b).to(device),
                proposal_mask=__import__('torch').from_numpy(msk).to(device),
                proposal_labels=__import__('torch').from_numpy(labs).to(device),
                cache_image_feat=False,
                use_cached_feat=False,
            )
            head = model.pts_bbox_head
            g = getattr(head, 'debug_last_gamma', None)
            b = getattr(head, 'debug_last_beta', None)
            v = getattr(head, 'debug_last_valid', None)
            if g is None or b is None or v is None:
                print(f'[warn] {scene}: no debug gamma/beta captured (no proposal fusion?)')
                continue
            # Convert to numpy [NQ]
            g = g.squeeze(0).cpu().numpy()  # mean over C already
            b = b.squeeze(0).cpu().numpy()
            v = v.squeeze(0).cpu().numpy().astype(bool)
            mag = np.abs(g - 1.0)  # how much FiLM gamma deviates from identity
            mag[~v] = 0.0
            idx = np.argsort(-mag)[:max(1, min(args.topk, mag.size))]

        # Draw proposals, highlight top-K
        overlay = canvas.copy()
        colors = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}  # ped/div/bnd BGR
        for j in range(num_queries):
            if v[j] == 0:
                continue
            cls = int(labs[0, j])
            col = colors.get(cls, (200, 200, 200))
            crv = pts[j]
            # denorm from [-1,1] to pixel on letterboxed canvas size (H,W) under unit square (since bounds already normalized)
            # here, pts are already normalized [-1,1] within bounds, but letterbox draws on fixed canvas; we scale from unit square to pixels
            # Convert normalized [-1,1] to [0,1]
            uv = (crv + 1.0) * 0.5
            # Map to canvas pixels directly
            px = np.stack([uv[:,0] * W, (1.0 - uv[:,1]) * H], 1).round().clip([0,0],[W-1,H-1]).astype(np.int32)
            thick = 3 if j in idx else 1
            cv2.polylines(overlay, [px], False, col, thickness=thick)
        # Legend and stats
        out_dir = osp.join(args.out_root, scene)
        os.makedirs(out_dir, exist_ok=True)
        cv2.putText(overlay, f'topK={len(idx)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        out_png = osp.join(out_dir, 'proposals_overlay.png')
        cv2.imwrite(out_png, overlay)
        with open(osp.join(out_dir, 'stats.txt'), 'w') as f:
            for c in [0,1,2]:
                m = mag[(labs[0]==c) & v]
                f.write(f'class {c}: valid={int(((labs[0]==c) & v).sum())}, mean|gamma-1|={float(m.mean()) if m.size>0 else 0.0:.4f}\n')
        print(f'[ok] {scene}: {out_png}')


if __name__ == '__main__':
    main()

