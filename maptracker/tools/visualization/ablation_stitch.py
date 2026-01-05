#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np


def read_img(p: Path):
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img


def pad_to_same_size(imgs: List[np.ndarray]) -> List[np.ndarray]:
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    out = []
    for im in imgs:
        dh = h - im.shape[0]
        dw = w - im.shape[1]
        if dh == 0 and dw == 0:
            out.append(im)
            continue
        pad = cv2.copyMakeBorder(im, 0, dh, 0, dw, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
        out.append(pad)
    return out


def main():
    ap = argparse.ArgumentParser(description="Stitch ablation images horizontally")
    ap.add_argument("scene_dir", help="Scene directory under rendered_gt/val/<scene>")
    ap.add_argument("--names", nargs="*", default=["08_agg_semantic.png", "10_render_gt.png", "10_render_gt_unsmooth.png", "10_render_gt_jag.png"], help="Image names in order")
    ap.add_argument("--out", default="ablation_10_compare.png", help="Output filename")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    imgs = []
    used = []
    for nm in args.names:
        p = scene_dir / nm
        im = read_img(p)
        if im is None:
            continue
        imgs.append(im)
        used.append(nm)
    if not imgs:
        print(f"[err] no input images in {scene_dir}")
        return
    imgs = pad_to_same_size(imgs)
    canvas = np.concatenate(imgs, axis=1)
    out_path = scene_dir / args.out
    cv2.imwrite(str(out_path), canvas)
    print(f"[ok] stitched {len(imgs)} images -> {out_path}")


if __name__ == "__main__":
    main()

