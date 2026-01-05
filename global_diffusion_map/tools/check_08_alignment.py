#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


def _list_scenes(root: str) -> List[str]:
    if not osp.isdir(root):
        return []
    out = []
    for x in os.listdir(root):
        p = osp.join(root, x)
        if osp.isdir(p):
            out.append(x)
    out.sort()
    return out


def _load_mask(path: str) -> np.ndarray:
    im = imageio.imread(path)
    if im.ndim == 3:
        # non-white as foreground
        if im.dtype == np.uint8:
            fg = (im != 255).any(axis=-1)
        else:
            fg = (im != im.max()).any(axis=-1)
    else:
        fg = im > 0
    return fg.astype(np.uint8)


def _best_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 20) -> Tuple[int, int, float]:
    H, W = a.shape
    best = (0, 0, -1.0)
    # Precompute sums for normalization
    a_sum = a.sum()
    if a_sum <= 0:
        return 0, 0, 0.0
    for dy in range(-max_shift, max_shift + 1):
        y0 = max(0, dy)
        y1 = min(H, H + dy)
        y0b = max(0, -dy)
        y1b = min(H, H - dy)
        for dx in range(-max_shift, max_shift + 1):
            x0 = max(0, dx)
            x1 = min(W, W + dx)
            x0b = max(0, -dx)
            x1b = min(W, W - dx)
            # Overlapping window
            if (y1 - y0) <= 0 or (x1 - x0) <= 0:
                continue
            inter = (a[y0:y1, x0:x1] & b[y0b:y1b, x0b:x1b]).sum()
            denom = max(1, a_sum)
            score = inter / float(denom)
            if score > best[2]:
                best = (dy, dx, score)
    return best


def check_split(sem_root: str, split_name: str, max_shift: int = 20) -> Dict[str, Dict]:
    scenes = _list_scenes(sem_root)
    out: Dict[str, Dict] = {}
    for s in scenes:
        d = osp.join(sem_root, s)
        p05 = osp.join(d, '05_static_gt.png')
        p08 = osp.join(d, '08_agg_semantic.png')
        if not (osp.exists(p05) and osp.exists(p08)):
            continue
        m05 = _load_mask(p05)
        m08 = _load_mask(p08)
        H05, W05 = m05.shape
        H08, W08 = m08.shape
        same_shape = (H05 == H08 and W05 == W08)
        if not same_shape:
            out[s] = {'shape05': (H05, W05), 'shape08': (H08, W08), 'aligned': False, 'dy': None, 'dx': None, 'score': 0.0}
            continue
        dy, dx, score = _best_shift(m05, m08, max_shift=max_shift)
        aligned = (dy == 0 and dx == 0)
        out[s] = {'shape05': (H05, W05), 'shape08': (H08, W08), 'aligned': aligned, 'dy': dy, 'dx': dx, 'score': float(score)}
    # summary
    total = len(out)
    bad = sum(1 for v in out.values() if (not v['aligned']))
    print(f'[summary] {split_name}: total={total}, misaligned={bad}, aligned={total-bad}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-root', default='maptracker/work_dirs/semantic/valid')
    ap.add_argument('--train-root', default='maptracker/work_dirs/semantic/train')
    ap.add_argument('--max-shift', type=int, default=20)
    ap.add_argument('--out-json', default='global_diffusion_map/work_dirs/align_check/summary.json')
    args = ap.parse_args()

    os.makedirs(osp.dirname(args.out_json), exist_ok=True)
    val_res = check_split(args.val_root, 'val', max_shift=args.max_shift)
    train_res = check_split(args.train_root, 'train', max_shift=args.max_shift)
    import json
    with open(args.out_json, 'w') as f:
        json.dump({'val': val_res, 'train': train_res}, f, indent=2)
    print(f"[ok] wrote {args.out_json}")


if __name__ == '__main__':
    main()

