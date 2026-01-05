#!/usr/bin/env python
from __future__ import annotations

"""
Compare inference between Clean (RasterEncoder) and PolyDiffuse (MapTR encoder)
on a single scene, across one or more start modes.

This script is a thin orchestrator that:
- Calls the existing clean and poly inference CLIs with matched settings
- Forces consistent visualization (same overlay thickness; no letterbox)
- Optionally stitches side-by-side overlays for quick visual comparison

Example:
  python -u global_diffusion_map/refine/clean/compare_infer_one_scene.py \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --agg-pred-root maptracker/work_dirs/aggregated_scene_vectors/av2_oldsplit/valid \
    --stats-json global_diffusion_map/work_dirs/av2_stats.json \
    --scene 02a00399-3857-444e-8db3-a8f58489c394 \
    --ckpt-clean <clean_ckpt.pth> --ckpt-poly <poly_ckpt.pth> \
    --modes noise gt_noise proposal --start-sigma 1.5 \
    --steps 30 --sigma-min 0.002 --sigma-max 0.6 --second-order \
    --out-root global_diffusion_map/refine/work_dirs/compare_clean_poly
"""

import argparse
import os
import os.path as osp
import subprocess
from typing import List

from PIL import Image


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _stitch_side_by_side(a_path: str, b_path: str, out_path: str) -> None:
    try:
        a = Image.open(a_path).convert('RGB')
        b = Image.open(b_path).convert('RGB')
        H = max(a.height, b.height)
        W = a.width + b.width
        canvas = Image.new('RGB', (W, H), (255, 255, 255))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        os.makedirs(osp.dirname(out_path), exist_ok=True)
        canvas.save(out_path)
    except Exception as e:
        print(f"[warn] stitch failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description='Compare Clean vs PolyDiffuse inference on one scene')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', default=None)
    ap.add_argument('--stats-json', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--ckpt-clean', required=True)
    ap.add_argument('--ckpt-poly', required=True)
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/compare_clean_poly')

    # Sampler + filtering (matched for both)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--thr', type=float, default=0.2)
    ap.add_argument('--nms-meters', type=float, default=0.0)
    ap.add_argument('--topk', type=int, default=0)
    ap.add_argument('--min-len-norm', type=float, default=0.05)

    # Mode control
    ap.add_argument('--modes', nargs='+', default=['noise', 'gt_noise', 'proposal'],
                    help='Which start modes to run (subset of: noise gt_noise proposal)')
    ap.add_argument('--start-sigma', type=float, default=0.0, help='Used when mode includes gt_noise')

    # Poly encoder config (defaults should work)
    ap.add_argument('--polydiff-cfg', default='official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py')
    ap.add_argument('--pretrained-maptr-ckpt', default='global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth')

    args = ap.parse_args()

    scene = args.scene
    base_out = osp.join(args.out_root, scene)
    modes = [m for m in args.modes if m in ('noise', 'gt_noise', 'proposal')]

    if 'proposal' in modes and not args.agg_pred_root:
        print('[info] proposal mode requested but --agg-pred-root not provided; skipping proposal')
        modes = [m for m in modes if m != 'proposal']

    for mode in modes:
        print(f"\n=== Mode: {mode} ===")
        # Clean
        out_clean = osp.join(base_out, 'clean', mode)
        os.makedirs(out_clean, exist_ok=True)
        cmd_clean = [
            'python', '-u', 'global_diffusion_map/refine/clean/infer_one_scene_clean.py',
            '--static-root', args.static_root,
            '--rendered-root', args.rendered_root,
            '--stats-json', args.stats_json,
            '--scene', scene,
            '--ckpt', args.ckpt_clean,
            '--steps', str(int(args.steps)),
            '--sigma-min', str(float(args.sigma_min)),
            '--sigma-max', str(float(args.sigma_max)),
            '--rho', str(float(args.rho)),
            '--thr', str(float(args.thr)),
            '--nms-meters', str(float(args.nms_meters)),
            '--topk', str(int(args.topk)),
            '--min-len-norm', str(float(args.min_len_norm)),
            '--out-root', out_clean,
            '--overlay-raw',  # force non-letterbox to match Poly encoder viz
        ]
        if mode == 'proposal' and args.agg_pred_root:
            cmd_clean += ['--start', 'proposal', '--agg-pred-root', args.agg_pred_root]
        elif mode == 'gt_noise':
            cmd_clean += ['--start', 'gt_noise']
            if float(args.start_sigma) > 0.0:
                cmd_clean += ['--start-sigma', str(float(args.start_sigma))]
        else:
            cmd_clean += ['--start', 'noise']

        _run(cmd_clean)

        # Poly
        out_poly = osp.join(base_out, 'poly', mode)
        os.makedirs(out_poly, exist_ok=True)
        cmd_poly = [
            'python', '-u', 'global_diffusion_map/refine/clean/infer_polydiffuse_aligned.py',
            '--static-root', args.static_root,
            '--rendered-root', args.rendered_root,
            '--stats-json', args.stats_json,
            '--scene', scene,
            '--ckpt', args.ckpt_poly,
            '--steps', str(int(args.steps)),
            '--sigma-min', str(float(args.sigma_min)),
            '--sigma-max', str(float(args.sigma_max)),
            '--rho', str(float(args.rho)),
            '--thr', str(float(args.thr)),
            '--nms-meters', str(float(args.nms_meters)),
            '--topk', str(int(args.topk)),
            '--min-len-norm', str(float(args.min_len_norm)),
            '--out-root', out_poly,
            '--polydiff-cfg', args.polydiff_cfg,
            '--pretrained-maptr-ckpt', args.pretrained_maptr_ckpt,
        ]
        if mode == 'proposal' and args.agg_pred_root:
            cmd_poly += ['--start', 'proposal', '--agg-pred-root', args.agg_pred_root]
        elif mode == 'gt_noise':
            cmd_poly += ['--start', 'gt_noise']
            if float(args.start_sigma) > 0.0:
                cmd_poly += ['--start-sigma', str(float(args.start_sigma))]
        else:
            cmd_poly += ['--start', 'noise']

        _run(cmd_poly)

        # Stitch side-by-side overlays for quick visual diff
        clean_png = osp.join(out_clean, scene, 'refined_overlay.png')
        poly_png = osp.join(out_poly, scene, 'refined_overlay.png')
        out_cmp = osp.join(base_out, f'compare_{mode}.png')
        _stitch_side_by_side(clean_png, poly_png, out_cmp)
        print(f"[ok] compare overlay saved: {out_cmp}")

    print(f"\n[ok] comparison finished. Root: {base_out}")


if __name__ == '__main__':
    main()

