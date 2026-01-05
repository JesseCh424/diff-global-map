#!/usr/bin/env python
"""
FAST‑ONLY augmentation for rendered static GT maps (10_render_gt.png → 11_gt_aug.png).

Method (single path):
- Per‑class ROI crop (pad by units), signed‑distance transform (SDF) computed at
  a downscaled factor then upsampled for outline jag: new = (SDF + amp_px*noise) > 0.
- Boundary bubbles: random discs along outlines (erode or bulge), aggregated and
  applied in batch; capped per‑MPx for runtime.
- Apply to ped, boundary, and divider. Recompute ped×boundary overlap (cyan) and recolor.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

# BGR colors
COL_PED = (255, 0, 0)
COL_BND = (0, 255, 0)
COL_DIV = (0, 0, 255)
COL_OVL = (255, 255, 0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="FAST unit-scale jag augmentation for 10_render_gt → 11_gt_aug")
    ap.add_argument("--val-root", required=True, help="rendered_gt val root")
    ap.add_argument("--scenes", nargs="*", default=None, help="Scene IDs to process (else first N)")
    ap.add_argument("--first-n", type=int, default=5, help="When --scenes not set, process first N folders")
    # Unit scale + SDF noise
    ap.add_argument("--unit-px", type=int, default=20, help="Atomic unit radius (px)")
    ap.add_argument("--noise-amp-units", type=float, default=1.0, help="Boundary shift amplitude (units) for ped/bnd")
    ap.add_argument("--noise-amp-units-div", type=float, default=0.7, help="Boundary shift amplitude (units) for divider")
    ap.add_argument("--noise-sigma-units", type=float, default=1.2, help="Gaussian smoothing sigma (units) for noise field")
    ap.add_argument("--noise-block-units", type=float, default=2.0, help="Coarse block size (units) for base noise tiling")
    ap.add_argument("--roi-pad-units", type=float, default=2.0, help="ROI padding around class bbox (units)")
    ap.add_argument("--sdf-downscale", type=float, default=0.33, help="SDF compute downscale (0.25–1.0)")
    ap.add_argument("--skip-div-aug", action="store_true", help="Fast mode: skip divider augmentation for speed")
    # Bubbles
    ap.add_argument("--bubble-prob", type=float, default=0.002, help="Prob per boundary pixel to seed a bubble")
    ap.add_argument("--bubbles-per-mpx", type=float, default=80.0, help="Cap bubble seeds per MPx per class")
    ap.add_argument("--bubble-rmin-units", type=float, default=0.3, help="Min bubble radius (units)")
    ap.add_argument("--bubble-rmax-units", type=float, default=1.0, help="Max bubble radius (units)")
    ap.add_argument("--bubble-erosion-frac", type=float, default=0.9, help="Fraction of bubbles that erode (else bulge)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    return ap.parse_args()


def mask_from_color(img: np.ndarray, bgr: Tuple[int, int, int]) -> np.ndarray:
    b, g, r = bgr
    return (img[:, :, 0] == b) & (img[:, :, 1] == g) & (img[:, :, 2] == r)


def signed_distance(mask: np.ndarray, downscale: float) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return np.zeros_like(m, dtype=np.float32)
    if float(downscale) < 1.0:
        ds = max(0.1, float(downscale))
        H, W = m.shape
        Wd, Hd = max(1, int(round(W * ds))), max(1, int(round(H * ds)))
        md = cv2.resize(m, (Wd, Hd), interpolation=cv2.INTER_AREA)
        md = (md >= 0.5).astype(np.uint8)
        invd = (1 - md).astype(np.uint8)
        din = cv2.distanceTransform(md, cv2.DIST_L2, 5)
        dout = cv2.distanceTransform(invd, cv2.DIST_L2, 5)
        sdfd = din - dout
        return cv2.resize(sdfd, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    else:
        inv = (1 - m).astype(np.uint8)
        din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        dout = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        return din - dout


def make_noise_field(shape: Tuple[int, int], unit_px: int, sigma_units: float, block_units: float, rng: np.random.Generator) -> np.ndarray:
    H, W = shape
    block = max(1, int(round(block_units * unit_px)))
    hc, wc = max(1, H // block), max(1, W // block)
    coarse = rng.standard_normal((hc, wc)).astype(np.float32)
    noise = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC)
    sigma_px = max(0.1, float(sigma_units) * float(unit_px))
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)
    m, s = float(noise.mean()), float(noise.std() + 1e-6)
    noise = (noise - m) / s
    noise = noise / max(1.0, float(np.max(np.abs(noise))) + 1e-6)
    return noise.astype(np.float32)


def outline_jag(mask: np.ndarray, unit_px: int, amp_units: float, sigma_units: float, block_units: float, downscale: float, rng: np.random.Generator) -> np.ndarray:
    if mask.max() == 0:
        return mask
    sdf = signed_distance(mask, downscale)
    noise = make_noise_field(mask.shape, unit_px, sigma_units, block_units, rng)
    amp_px = float(amp_units) * float(unit_px)
    warped = sdf + amp_px * noise
    return (warped > 0).astype(np.uint8)


def bubbles(mask: np.ndarray, unit_px: int, prob: float, rmin_units: float, rmax_units: float, erosion_frac: float, max_per_mpx: float, rng: np.random.Generator) -> np.ndarray:
    if mask.max() == 0:
        return mask
    m = (mask > 0).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k3)
    kthin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, unit_px // 2) * 2 + 1, max(1, unit_px // 2) * 2 + 1))
    thinned = cv2.erode(grad, kthin)
    if thinned.max() == 0:
        thinned = grad
    picks = (rng.random(m.shape) < float(prob)) & (thinned > 0)
    ys, xs = np.where(picks)
    if len(xs) == 0:
        return m
    H, W = m.shape
    max_seeds = int(round(max_per_mpx * (H * W) / 1_000_000.0))
    n = len(xs)
    if n > max_seeds and max_seeds > 0:
        idx = rng.choice(n, size=max_seeds, replace=False)
        xs, ys = xs[idx], ys[idx]
        n = len(xs)
    if n == 0:
        return m
    holes_mask = np.zeros_like(m, dtype=np.uint8)
    bulge_mask = np.zeros_like(m, dtype=np.uint8)
    for i in range(n):
        x, y = int(xs[i]), int(ys[i])
        r_units = float(rmin_units) + (float(rmax_units) - float(rmin_units)) * float(rng.random())
        r = max(1, int(round(r_units * float(unit_px))))
        if rng.random() < float(erosion_frac):
            cv2.circle(holes_mask, (x, y), r, 1, thickness=-1)
        else:
            cv2.circle(bulge_mask, (x, y), r, 1, thickness=-1)
    out = m.copy()
    if holes_mask.max() > 0:
        out[holes_mask > 0] = 0
    if bulge_mask.max() > 0:
        r_eff = max(1, int(round(((rmin_units + rmax_units) * 0.5) * float(unit_px))))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_eff + 1, 2 * r_eff + 1))
        local = cv2.dilate(out, k)
        out[bulge_mask > 0] = local[bulge_mask > 0]
    return out


def augment_mask_fast(mask: np.ndarray, unit_px: int, amp_units: float, sigma_units: float, block_units: float, roi_pad_units: float, downscale: float, bubble_prob: float, rmin_units: float, rmax_units: float, erosion_frac: float, bubbles_per_mpx: float, rng: np.random.Generator) -> np.ndarray:
    if mask.max() == 0:
        return mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    pad_px = int(round(max(0.0, roi_pad_units) * float(unit_px)))
    y0, y1 = max(0, int(ys.min()) - pad_px), min(mask.shape[0], int(ys.max()) + 1 + pad_px)
    x0, x1 = max(0, int(xs.min()) - pad_px), min(mask.shape[1], int(xs.max()) + 1 + pad_px)
    sub = mask[y0:y1, x0:x1]
    sub = outline_jag(sub, unit_px, amp_units, sigma_units, block_units, downscale, rng)
    sub = bubbles(sub, unit_px, bubble_prob, rmin_units, rmax_units, erosion_frac, bubbles_per_mpx, rng)
    out = np.zeros_like(mask, dtype=np.uint8)
    out[y0:y1, x0:x1] = sub
    return out


def process_scene(scene_dir: Path, unit_px: int, noise_amp_units: float, noise_amp_units_div: float, sigma_units: float, block_units: float, roi_pad_units: float, sdf_downscale: float, bubble_prob: float, rmin_units: float, rmax_units: float, erosion_frac: float, bubbles_per_mpx: float, rng: np.random.Generator) -> Optional[Path]:
    img_path = scene_dir / "10_render_gt.png"
    if not img_path.exists():
        return None
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    ped_mask = mask_from_color(img, COL_PED).astype(np.uint8)
    bnd_mask = mask_from_color(img, COL_BND).astype(np.uint8)
    div_mask = mask_from_color(img, COL_DIV).astype(np.uint8)
    ov_mask = mask_from_color(img, COL_OVL).astype(np.uint8)

    ped = np.maximum(ped_mask, ov_mask)
    bnd = np.maximum(bnd_mask, ov_mask)
    div = div_mask.copy()

    # Fast path only: ROI crop + SDF downscale + capped bubbles for all classes (divider included)
    ped = augment_mask_fast(ped, unit_px, noise_amp_units, sigma_units, block_units, roi_pad_units, max(0.25, float(sdf_downscale)), bubble_prob, rmin_units, rmax_units, erosion_frac, bubbles_per_mpx, rng)
    bnd = augment_mask_fast(bnd, unit_px, noise_amp_units, sigma_units, block_units, roi_pad_units, max(0.25, float(sdf_downscale)), bubble_prob, rmin_units, rmax_units, erosion_frac, bubbles_per_mpx, rng)
    div = augment_mask_fast(div, unit_px, noise_amp_units_div, sigma_units, block_units, roi_pad_units, max(0.25, float(sdf_downscale)), bubble_prob, rmin_units, rmax_units, erosion_frac, bubbles_per_mpx, rng)

    ov_new = (ped > 0) & (bnd > 0)
    ped_only = (ped > 0) & (~ov_new)
    bnd_only = (bnd > 0) & (~ov_new)
    out = np.ones_like(img, dtype=np.uint8) * 255
    out[ov_new] = COL_OVL
    out[ped_only] = COL_PED
    out[bnd_only] = COL_BND
    out[div > 0] = COL_DIV

    out_path = scene_dir / "11_gt_aug.png"
    cv2.imwrite(str(out_path), out)
    return out_path


def main() -> None:
    args = parse_args()
    root = Path(args.val_root)
    if not root.exists():
        print(f"[err] val_root not found: {root}")
        return
    if args.scenes:
        scenes: Iterable[str] = args.scenes
    else:
        scenes = [p.name for p in sorted(root.iterdir()) if p.is_dir()][: max(0, int(args.first_n))]
    rng = np.random.default_rng(args.seed)

    done = 0
    for s in scenes:
        out = process_scene(
            root / s,
            unit_px=int(args.unit_px),
            noise_amp_units=float(args.noise_amp_units),
            noise_amp_units_div=float(args.noise_amp_unites_div) if hasattr(args, 'noise_amp_unites_div') else float(args.noise_amp_units_div),
            sigma_units=float(args.noise_sigma_units),
            block_units=float(args.noise_block_units),
            roi_pad_units=float(args.roi_pad_units),
            sdf_downscale=float(args.sdf_downscale),
            bubble_prob=float(args.bubble_prob),
            rmin_units=float(args.bubble_rmin_units),
            rmax_units=float(args.bubble_rmax_units),
            erosion_frac=float(args.bubble_erosion_frac),
            bubbles_per_mpx=float(args.bubbles_per_mpx),
            rng=rng,
        )
        if out is None:
            print(f"[skip] {s}: missing or unreadable 10_render_gt.png")
            continue
        print(f"[ok] {s}: {out}")
        done += 1
    print(f"[done] 11_gt_aug generated for {done} scene(s)")


if __name__ == "__main__":
    main()
