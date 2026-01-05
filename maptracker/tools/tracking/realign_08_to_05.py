#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import List, Tuple

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


def _list_scenes(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if osp.isdir(osp.join(root, d))])


def _load_mask(path: str) -> np.ndarray:
    im = imageio.imread(path)
    if im.ndim == 3:
        fg = (im != 255).any(axis=-1)
    else:
        fg = (im > 0)
    return fg.astype(np.uint8)


def _best_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 10) -> Tuple[int, int, float]:
    H, W = a.shape
    best = (0, 0, -1.0)
    a_sum = a.sum(); b_sum = b.sum()
    if a_sum == 0 or b_sum == 0:
        return 0, 0, 0.0
    for dy in range(-max_shift, max_shift + 1):
        y0 = max(0, dy); y1 = min(H, H + dy)
        y0b = max(0, -dy); y1b = min(H, H - dy)
        for dx in range(-max_shift, max_shift + 1):
            x0 = max(0, dx); x1 = min(W, W + dx)
            x0b = max(0, -dx); x1b = min(W, W - dx)
            if y1 - y0 <= 0 or x1 - x0 <= 0:
                continue
            inter = (a[y0:y1, x0:x1] & b[y0b:y1b, x0b:x1b]).sum()
            score = inter / float(a_sum)
            if score > best[2]:
                best = (dy, dx, score)
    return best


def _shift_img(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    H, W = img.shape[:2]
    out = np.ones_like(img) * 255  # white background
    y0 = max(0, dy); y1 = min(H, H + dy)
    y0b = max(0, -dy); y1b = min(H, H - dy)
    x0 = max(0, dx); x1 = min(W, W + dx)
    x0b = max(0, -dx); x1b = min(W, W - dx)
    if y1 - y0 > 0 and x1 - x0 > 0:
        out[y0:y1, x0:x1] = img[y0b:y1b, x0b:x1b]
    return out


def main():
    ap = argparse.ArgumentParser(description='Fast realign 08_agg_semantic.png to 05_static_gt.png by pixel shift only')
    ap.add_argument('--sem-root', required=True, help='Root with per-scene folders containing 05_*.png and 08_*.png')
    ap.add_argument('--scenes', nargs='*', default=None, help='Specific scenes; default = all subdirs')
    ap.add_argument('--max-shift', type=int, default=10, help='Search radius in pixels')
    ap.add_argument('--in-place', action='store_true', help='Overwrite 08_agg_semantic.png instead of writing *_aligned.png')
    args = ap.parse_args()

    scenes = args.scenes or _list_scenes(args.sem_root)
    for s in scenes:
        d = osp.join(args.sem_root, s)
        p05 = osp.join(d, '05_static_gt.png')
        p08 = osp.join(d, '08_agg_semantic.png')
        if not (osp.exists(p05) and osp.exists(p08)):
            continue
        im05 = imageio.imread(p05)
        im08 = imageio.imread(p08)
        if im05.shape[:2] != im08.shape[:2]:
            print(f'[skip] {s}: shape mismatch {im05.shape[:2]} vs {im08.shape[:2]} (needs regeneration)')
            continue
        m05 = _load_mask(p05)
        m08 = _load_mask(p08)
        dy, dx, score = _best_shift(m05, m08, max_shift=args.max_shift)
        if dy == 0 and dx == 0:
            print(f'[ok] {s}: already aligned (score={score:.3f})')
            continue
        print(f'[fix] {s}: shift dy={dy}, dx={dx}, score={score:.3f}')
        im08_fix = _shift_img(im08, dy, dx)
        out_path = p08 if args.in_place else osp.join(d, '08_agg_semantic_aligned.png')
        imageio.imwrite(out_path, im08_fix)

if __name__ == '__main__':
    main()

