#!/usr/bin/env python
from __future__ import annotations

"""
Clean one-scene inference with step visualizations (PNG + GIF).

Features
- Input overlay: condition map + GT (white) + input polylines (semantic color)
- Final overlay: refined output polylines with predicted classes
- Step visualizations: per-step PNGs of x_t states and an animated GIF

EDM aligns with PolyDiffuse style: Karras schedule + Heun/Euler, time input log(sigma).
"""

import argparse
import os
import os.path as osp
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import (
    pack_gt_to_slots,
    pack_vectors_to_slots,
)
from global_diffusion_map.refine.single_scene_overfit import (
    RasterEncoder,
    overlay_on_raster,
    overlay_slots_annot,
    load_pickle,
    letterbox,
)
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train
from global_diffusion_map.refine.infer_refine import save_refined_pkl as _save_refined_pkl  # optional normalized saver


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _labels_from_budgets(budgets: Dict[int, int], N: int) -> np.ndarray:
    order: List[int] = []
    for orig in (1, 0, 2):  # MapTR orig ids: divider=1, ped=0, boundary=2
        cap = int(budgets.get(orig, 0))
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        order += [lab] * max(0, cap)
    out = np.full((N,), -1, dtype=np.int64)
    m = min(N, len(order))
    if m > 0:
        out[:m] = np.asarray(order[:m], dtype=np.int64)
    return out


def _present_from_mask(mask: np.ndarray) -> np.ndarray:
    # mask: [N,P] True=pad; present if any valid point exists
    return (~mask).any(axis=1)


def _nms_by_center(coords: np.ndarray, probs: np.ndarray, bounds: Sequence[float], nms_m: float) -> List[int]:
    if nms_m <= 0 or coords.shape[0] <= 1:
        return list(range(coords.shape[0]))
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    Wm = max(maxx - minx, 1e-6); Hm = max(maxy - miny, 1e-6)
    idx = list(range(coords.shape[0]))
    idx.sort(key=lambda i: float(probs[i]), reverse=True)
    keep: List[int] = []
    centers: Dict[int, np.ndarray] = {}
    def _c(i: int) -> np.ndarray:
        if i in centers:
            return centers[i]
        xy = coords[i]
        xm = (xy[:, 0] + 1.0) * 0.5 * Wm + minx
        ym = (xy[:, 1] + 1.0) * 0.5 * Hm + miny
        centers[i] = np.array([xm.mean(), ym.mean()], dtype=np.float32)
        return centers[i]
    for i in idx:
        ci = _c(i)
        ok = True
        for j in keep:
            cj = _c(j)
            if float(np.linalg.norm(ci - cj)) < nms_m:
                ok = False; break
        if ok:
            keep.append(i)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description='Clean one-scene inference with step visualizations')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', required=False, help='optional proposals root (per-scene pkl)')
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--ckpt', required=True)
    # EDM sampler
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=1.5)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    # SDEdit start
    ap.add_argument('--start', choices=['proposal', 'gt_noise', 'noise'], default='proposal')
    ap.add_argument('--start-sigma', type=float, default=0.0, help='SDEdit start sigma (>0 to enable)')
    ap.add_argument('--no-schedule-truncate', action='store_true',
                    help='Do NOT slice the Karras schedule when start-sigma>0; keep full steps and only initialize xK with noise')
    # outputs
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/infer_one_scene_clean')
    ap.add_argument('--csv-move', action='store_true', help='Write a CSV logging per-step movement (slot centers) with GT centers')
    ap.add_argument('--thr', type=float, default=0.2)
    ap.add_argument('--nms-meters', type=float, default=0.0)
    ap.add_argument('--keep-only-proposal', action='store_true')
    ap.add_argument('--topk', type=int, default=0)
    ap.add_argument('--min-len-norm', type=float, default=0.05, help='filter out short polylines by normalized arc length')
    ap.add_argument('--gif-fps', type=int, default=6)
    # Step viz mode: draw noisy states x_t or denoised predictions D(x_t)
    ap.add_argument('--viz-mode', choices=['state', 'denoised'], default='denoised',
                    help='Per-step visualization: noisy state x_t (state) or denoised prediction D(x_t) (denoised)')
    # Overlay canvas choice: default letterbox to 1024; with --overlay-raw draw on raw 10 PNG size
    ap.add_argument('--overlay-raw', action='store_true', help='Draw overlays directly on raw 10_render_gt.png without letterbox')
    # Blind test: disable raster condition (rv=0)
    ap.add_argument('--blind', action='store_true', help='Disable raster condition (use zero rv) to test reliance on 10 map')
    ap.add_argument('--debug', action='store_true')
    # Priors toggle
    ap.add_argument('--disable-prior', action='store_true', help='Disable shape/class prior at inference (ablation).')
    # Ghost injection (hallucination test)
    ap.add_argument('--inject-ghost', action='store_true', help='Inject a fake ghost polyline into proposals for hallucination test')
    ap.add_argument('--ghost-slot-index', type=int, default=-1, help='Slot index to place ghost (default: last slot)')
    ap.add_argument('--ghost-class', type=int, default=2, help='Ghost class label in MapTR space: 0=divider,1=ped,2=boundary')
    ap.add_argument('--ghost-center-norm-x', type=float, default=-0.85)
    ap.add_argument('--ghost-center-norm-y', type=float, default=-0.85)
    ap.add_argument('--ghost-len-norm', type=float, default=0.12, help='Length in normalized units (~ canvas fraction)')
    # Rigid shift test (normalized coords). When enabled, you can construct a shifted
    # proposal directly from GT via --rigid-shift-from-gt, or apply a shift on the
    # existing x_prop.
    ap.add_argument('--rigid-shift-norm-x', type=float, default=0.0)
    ap.add_argument('--rigid-shift-norm-y', type=float, default=0.0)
    ap.add_argument('--rigid-shift-from-gt', action='store_true', help='build proposal from GT then apply normalized shift (for rigid-shift test)')
    # Proposal drop test (when building proposal from GT): randomly drop a fraction or fixed number of GT instances
    ap.add_argument('--prop-drop-frac', type=float, default=0.0, help='Fraction of present GT instances to drop from proposal (0..1) when --rigid-shift-from-gt is used')
    ap.add_argument('--prop-drop-num', type=int, default=0, help='Exact number of GT instances to drop (overrides frac>0), when --rigid-shift-from-gt is used')
    args = ap.parse_args()

    set_seed(0)
    # load stats
    import json
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20)); N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    labels_default = _labels_from_budgets(budgets, N)

    # load GT + bounds
    gt_pkl = osp.join(args.static_root, f'{args.scene}.pkl')
    gt = load_pickle(gt_pkl)
    bounds = gt.get('bounds')
    if bounds is None:
        raise RuntimeError('bounds-missing in static GT pkl')
    gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, num_points=P, num_queries=N)
    # raster (letterbox to fixed 1024×1024 for conditioning parity with training)
    img = Image.open(osp.join(args.rendered_root, args.scene, '10_render_gt.png')).convert('RGB')
    ras_np = np.asarray(img, dtype=np.float32) / 255.0   # HWC RGB
    ras = ras_np.transpose(2, 0, 1)                      # CHW RGB

    # input proposals
    x_prop = np.zeros((N, P, 2), dtype=np.float32)
    m_prop = np.ones((N, P), dtype=bool)
    # Rigid-shift-from-GT overrides proposal source for testing
    if bool(getattr(args, 'rigid_shift_from_gt', False)):
        x_prop = gt_pack.copy()
        m_prop = gt_mask.copy()
        dx = float(getattr(args, 'rigid_shift_norm_x', 0.0))
        dy = float(getattr(args, 'rigid_shift_norm_y', 0.0))
        if abs(dx) > 0.0 or abs(dy) > 0.0:
            x_prop = np.clip(x_prop + np.array([dx, dy], dtype=np.float32)[None, None, :], -1.0, 1.0)
        # labels: use budget order by default
        input_labels = labels_default.copy()
        # Randomly drop GT instances from proposal if requested
        try:
            present_slots = (~gt_mask).any(axis=1)  # [N]
            idx_all = np.where(present_slots)[0]
            k = 0
            if int(getattr(args, 'prop_drop_num', 0)) > 0:
                k = min(int(args.prop_drop_num), int(idx_all.size))
            elif float(getattr(args, 'prop_drop_frac', 0.0)) > 0.0:
                k = int(round(float(args.prop_drop_frac) * float(idx_all.size)))
            if k > 0:
                np.random.shuffle(idx_all)
                drop_ids = idx_all[:k]
                for di in drop_ids:
                    m_prop[di, :] = True  # mark all points as padding → absent
                    # optional: clear label to unknown for dropped slots
                    try:
                        input_labels[di] = -1
                    except Exception:
                        pass
        except Exception:
            pass
    else:
        if args.start == 'proposal' and args.agg_pred_root:
            agg = load_pickle(osp.join(args.agg_pred_root, f'{args.scene}.pkl'))
            x_prop, m_prop, labs_prop = pack_vectors_to_slots(agg, bounds, budgets, num_points=P, num_queries=N)
            # prefer packed labels if available
            input_labels = labs_prop
            dx = float(getattr(args, 'rigid_shift_norm_x', 0.0))
            dy = float(getattr(args, 'rigid_shift_norm_y', 0.0))
            if abs(dx) > 0.0 or abs(dy) > 0.0:
                x_prop = np.clip(x_prop + np.array([dx, dy], dtype=np.float32)[None, None, :], -1.0, 1.0)
        elif args.start == 'gt_noise':
            x_prop = gt_pack.copy()
            m_prop = gt_mask.copy()
            dx = float(getattr(args, 'rigid_shift_norm_x', 0.0))
            dy = float(getattr(args, 'rigid_shift_norm_y', 0.0))
            if abs(dx) > 0.0 or abs(dy) > 0.0:
                x_prop = np.clip(x_prop + np.array([dx, dy], dtype=np.float32)[None, None, :], -1.0, 1.0)
        else:  # noise
            x_prop = np.random.uniform(-1.0, 1.0, size=(N, P, 2)).astype(np.float32)
            m_prop = np.zeros((N, P), dtype=bool)

    # Optional: inject a fake ghost line into proposals (for hallucination robustness test)
    ghost_idx = int(getattr(args, 'ghost_slot_index', -1))
    if bool(getattr(args, 'inject_ghost', False)) and N > 0:
        gi = (N - 1) if (ghost_idx < 0 or ghost_idx >= N) else ghost_idx
        cx = float(getattr(args, 'ghost_center_norm_x', -0.85))
        cy = float(getattr(args, 'ghost_center_norm_y', -0.85))
        L  = float(getattr(args, 'ghost_len_norm', 0.12))
        # Build a short straight segment around (cx,cy)
        t = np.linspace(-0.5, 0.5, num=P, dtype=np.float32)
        seg = np.stack([cx + L * t, cy + L * t], axis=1).astype(np.float32)
        x_prop[gi] = np.clip(seg, -1.0, 1.0)
        # mark as present
        m_prop[gi, :] = False
        # set its input class prior
        if 'input_labels' not in locals():
            input_labels = labels_default.copy()
        try:
            input_labels[gi] = int(getattr(args, 'ghost_class', 2))
        except Exception:
            pass

    # present mask for input (proposal)
    prop_present = _present_from_mask(m_prop)
    # labels for input overlay: default budget order if not from proposal
    if 'input_labels' not in locals():
        input_labels = labels_default.copy()
    # 对于 padding 槽位，将类别先验设为未知 -1（避免把空槽当作有效先验）
    try:
        if m_prop is not None:
            pad_slots = m_prop.all(axis=1)
            input_labels[pad_slots] = -1
    except Exception:
        pass

    # model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = RasterEncoder(out_dim=256).to(device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    data = torch.load(args.ckpt, map_location=device)
    if 'encoder' in data:
        enc.load_state_dict(data['encoder'], strict=False)
    if 'net' in data:
        base.load_state_dict(data['net'], strict=False)

    # Prepare tensors
    x0 = torch.from_numpy(x_prop).float().unsqueeze(0).to(device)        # [1,N,P,2]
    # For encoder, use letterboxed 1024 canvas to match dataset policy
    ras_enc_chw = ras.copy()
    canvas_bgr, _, _, _ = letterbox(ras_enc_chw, 1024, (1024, 1024))
    canvas_rgb = canvas_bgr[:, :, ::-1].astype(np.float32) / 255.0
    ras_enc = canvas_rgb.transpose(2, 0, 1)
    ras_t = torch.from_numpy(ras_enc).float().unsqueeze(0).to(device)        # [1,3,1024,1024]
    with torch.cuda.amp.autocast(enabled=False):
        if bool(getattr(args, 'blind', False)):
            rv = torch.zeros((1, 256), device=device, dtype=torch.float32)
        else:
            rv = enc(ras_t)

    # Build schedule; optionally slice from start_sigma (unless --no-schedule-truncate)
    sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
    if args.start_sigma > 0.0:
        if not bool(getattr(args, 'no_schedule_truncate', False)):
            # find first index where sigma <= start_sigma, then slice schedule
            mask = (sigmas <= float(args.start_sigma) + 1e-6)
            idx = int(torch.nonzero(mask, as_tuple=False)[0].item()) if bool(mask.any()) else int(sigmas.numel() - 1)
            sigmas = sigmas[idx:]
        # initialize xK with start-sigma noise irrespective of truncation
        xK = torch.clamp(x0 + torch.randn_like(x0) * float(args.start_sigma), -1.0, 1.0)
    else:
        xK = x0.clone()

    # Run EDM steps
    # Pass proposal as shape prior and proposal/budget labels as class prior
    x_prior = torch.from_numpy(x_prop).float().unsqueeze(0).to(device)
    input_labels_t = torch.from_numpy(input_labels).long().unsqueeze(0).to(device)
    pred_coords, pred_logits, preds, states = edm_unrolled_train(
        net, xK, rv, sigmas, second_order=bool(args.second_order),
        cond_prior=(None if bool(args.disable_prior) else x_prior),
        input_labels=(None if bool(args.disable_prior) else input_labels_t))

    # Final predictions
    coords_np = pred_coords[0].detach().cpu().numpy()  # [N,P,2]
    logits_np = pred_logits[0].detach().cpu().numpy().reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    # 1) prob filter
    keep_indices = np.where(probs >= float(args.thr))[0].tolist()
    # 2) optional: keep only proposal-present slots (disabled by default)
    if bool(args.keep_only_proposal):
        keep_indices = [i for i in keep_indices if (i < prop_present.shape[0] and bool(prop_present[i]))]
    # 3) geometric length filter (remove center noise)
    min_len = float(getattr(args, 'min_len_norm', 0.05))
    if min_len > 0 and len(keep_indices) > 0:
        valid_keep: List[int] = []
        for i in keep_indices:
            line = coords_np[i]
            L = np.linalg.norm(line[1:] - line[:-1], axis=1).sum()
            if float(L) > min_len:
                valid_keep.append(i)
        keep_indices = valid_keep
    # 4) NMS on subset then map back to original indices
    if float(args.nms_meters) > 0.0 and len(keep_indices) > 0:
        subset_coords = coords_np[keep_indices]
        subset_probs = probs[keep_indices]
        subset_keep = _nms_by_center(subset_coords, subset_probs, bounds, float(args.nms_meters))
        keep_indices = [keep_indices[k] for k in subset_keep]
    # 5) top-k
    if int(args.topk) > 0 and len(keep_indices) > int(args.topk):
        keep_indices.sort(key=lambda i: float(probs[i]), reverse=True)
        keep_indices = keep_indices[: int(args.topk)]
    keep = keep_indices

    # Semantic labels from final step
    sem_labels = None
    try:
        if len(preds) > 0 and len(preds[-1]) >= 3 and preds[-1][2] is not None:
            sem_np = preds[-1][2][0].detach().cpu().numpy()  # [N,3]
            sem_labels = np.argmax(sem_np, axis=-1).astype(np.int64)
    except Exception:
        sem_labels = None

    # Output dir
    out_dir = osp.join(args.out_root, args.scene)
    steps_dir = osp.join(out_dir, 'steps')
    os.makedirs(steps_dir, exist_ok=True)

    # 1) Input overlay (GT white + input colored)
    present_mask = np.ones((N, P), dtype=bool)
    for i in range(N):
        if bool(prop_present[i]):
            present_mask[i, :] = False
    if bool(getattr(args, 'overlay_raw', False)):
        overlay_slots_annot(
            osp.join(out_dir, 'input_overlay.png'), ras, bounds,
            slots=x_prop, mask=present_mask, labels=input_labels,
            title='Input (proposal) + GT', gt_slots=gt_pack, gt_mask=gt_mask,
            cond_max_side=None, cond_fixed_size=None)
    else:
        overlay_slots_annot(
            osp.join(out_dir, 'input_overlay.png'), ras, bounds,
            slots=x_prop, mask=present_mask, labels=input_labels,
            title='Input (proposal) + GT', gt_slots=gt_pack, gt_mask=gt_mask,
            cond_max_side=1024, cond_fixed_size=(1024, 1024))

    # 2) Final refined overlay (GT white + refined colored)
    mask_keep = np.ones((N, P), dtype=bool)
    for i in keep:
        mask_keep[i, :] = False
    if bool(getattr(args, 'overlay_raw', False)):
        overlay_slots_annot(
            osp.join(out_dir, 'refined_overlay.png'), ras, bounds,
            slots=coords_np, mask=mask_keep, labels=sem_labels,
            title=f'Refined (thr={args.thr})', gt_slots=gt_pack, gt_mask=gt_mask,
            cond_max_side=None, cond_fixed_size=None)
    else:
        overlay_slots_annot(
            osp.join(out_dir, 'refined_overlay.png'), ras, bounds,
            slots=coords_np, mask=mask_keep, labels=sem_labels,
            title=f'Refined (thr={args.thr})', gt_slots=gt_pack, gt_mask=gt_mask,
            cond_max_side=1024, cond_fixed_size=(1024, 1024))

    # If ghost injected, report whether the ghost slot survived filtering
    if bool(getattr(args, 'inject_ghost', False)):
        gi = (N - 1) if (ghost_idx < 0 or ghost_idx >= N) else ghost_idx
        survived = gi in keep
        prob_g = float(probs[gi]) if 0 <= gi < probs.shape[0] else float('nan')
        print(f"[ghost] slot={gi} survived={survived} prob={prob_g:.3f}")

    # 3) Step visualizations (states or denoised predictions)
    # states: list of [B,N,P,2]; preds[k] = (coords, logits, sem) aligns with step k (1..K)
    frames: List[Image.Image] = []
    for si in range(len(states)):
        # Choose what to draw per step
        draw_mode = getattr(args, 'viz_mode', 'denoised')
        if draw_mode == 'denoised' and si > 0 and (si - 1) < len(preds):
            # Denoised prediction at step si uses preds[si-1]
            xt = preds[si - 1][0][0].detach().cpu().numpy()  # [N,P,2]
            # semantic labels from current step if available
            lab_step = None
            if len(preds[si - 1]) >= 3 and preds[si - 1][2] is not None:
                sem_np = preds[si - 1][2][0].detach().cpu().numpy()
                lab_step = np.argmax(sem_np, axis=-1).astype(np.int64)
        else:
            # Default: raw state x_t (noisy) — step 0 always uses the state
            xt = states[si][0].detach().cpu().numpy()
            lab_step = None
            if si == 0:
                # Step 00 尚无模型语义输出，使用输入先验类别为初始着色
                try:
                    lab_step = input_labels if isinstance(input_labels, np.ndarray) else input_labels_t[0].detach().cpu().numpy()
                except Exception:
                    lab_step = None
            elif (si - 1) < len(preds) and len(preds[si - 1]) >= 3 and preds[si - 1][2] is not None:
                sem_np = preds[si - 1][2][0].detach().cpu().numpy()
                lab_step = np.argmax(sem_np, axis=-1).astype(np.int64)
        # 仅可视化最终“保留”的槽位，避免将背景/中心噪声的历史轨迹画进 GIF
        mask_step = np.ones((N, P), dtype=bool)
        for i in keep:
            mask_step[i, :] = False
        title = f'step {si:02d}/{len(states)-1}' if len(states) > 1 else 'step 00'
        out_path = osp.join(steps_dir, f'step_{si:03d}.png')
        if bool(getattr(args, 'overlay_raw', False)):
            overlay_slots_annot(out_path, ras, bounds, slots=xt, mask=mask_step, labels=lab_step, title=title,
                                cond_max_side=None, cond_fixed_size=None)
        else:
            overlay_slots_annot(out_path, ras, bounds, slots=xt, mask=mask_step, labels=lab_step, title=title,
                                cond_max_side=1024, cond_fixed_size=(1024, 1024))
        try:
            frames.append(Image.open(out_path).convert('RGB'))
        except Exception:
            pass

    # 3.5) CSV: per-step movement (centers + orientation) + GT/Proposal centers + orientation
    if bool(getattr(args, 'csv_move', False)):
        def _center_xy(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
            if mask is not None:
                valid = ~mask
                if bool(valid.any()):
                    return arr[valid].mean(axis=0)
            return arr.mean(axis=0)
        def _endpoints(arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
            # Return (p_first, p_last) considering mask as True=pad
            if mask is not None:
                valid_idx = np.where(~mask)[0]
                if valid_idx.size >= 2:
                    i0, i1 = int(valid_idx[0]), int(valid_idx[-1])
                    return arr[i0], arr[i1]
            # fallback: first and last
            return arr[0], arr[-1]
        def _orient_unit(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
            v = (p1 - p0).astype(np.float32)
            n = float(np.linalg.norm(v))
            if n <= 1e-8:
                return np.array([0.0, 0.0], dtype=np.float32)
            return v / n
        gt_centers = [ _center_xy(gt_pack[i], gt_mask[i]) for i in range(N) ]
        prop_centers = [ _center_xy(x_prop[i], (m_prop[i] if m_prop is not None else None)) for i in range(N) ]
        gt_orients = []
        prop_orients = []
        for i in range(N):
            g0, g1 = _endpoints(gt_pack[i], gt_mask[i])
            p0, p1 = _endpoints(x_prop[i], (m_prop[i] if m_prop is not None else None))
            gt_orients.append(_orient_unit(g0, g1))
            prop_orients.append(_orient_unit(p0, p1))
        csv_path = osp.join(out_dir, 'steps_move.csv')
        import csv as _csv
        with open(csv_path, 'w', newline='') as fcsv:
            wr = _csv.writer(fcsv)
            wr.writerow([
                'step','sigma','slot',
                'cx_norm','cy_norm',
                'orient_dx','orient_dy','orient_dot_prev','flip',
                'gt_cx_norm','gt_cy_norm','gt_o_dx','gt_o_dy',
                'prop_cx_norm','prop_cy_norm','prop_o_dx','prop_o_dy',
                'label','prob'])
            for si in range(len(states)):
                xt = states[si][0].detach().cpu().numpy()  # [N,P,2]
                sigma_val = float('nan')
                if si > 0 and (si - 1) < len(sigmas):
                    try:
                        sigma_val = float(sigmas[si - 1].detach().cpu().item())
                    except Exception:
                        sigma_val = float('nan')
                step_logits = None
                if si > 0 and (si - 1) < len(preds):
                    try:
                        step_logits = preds[si - 1][1][0].detach().cpu().numpy().reshape(-1)
                    except Exception:
                        step_logits = None
                xt_prev = None
                if si > 0:
                    try:
                        xt_prev = states[si-1][0].detach().cpu().numpy()
                    except Exception:
                        xt_prev = None
                for i in range(N):
                    cxy = _center_xy(xt[i])
                    # orientation for this step (unit vector from first→last point)
                    o0, o1 = xt[i, 0], xt[i, -1]
                    o = _orient_unit(o0, o1)
                    dot_prev = float('nan')
                    flip = 0
                    if xt_prev is not None:
                        op = _orient_unit(xt_prev[i, 0], xt_prev[i, -1])
                        dot_prev = float(o[0]*op[0] + o[1]*op[1])
                        if dot_prev < -0.5:
                            flip = 1
                    gc = gt_centers[i]
                    pc = prop_centers[i]
                    go = gt_orients[i]
                    po = prop_orients[i]
                    lab = int(input_labels[i]) if input_labels is not None and i < len(input_labels) else -1
                    prob = float('nan')
                    if step_logits is not None:
                        try:
                            prob = 1.0 / (1.0 + np.exp(-float(step_logits[i])))
                        except Exception:
                            prob = float('nan')
                    wr.writerow([
                        si, sigma_val, i,
                        float(cxy[0]), float(cxy[1]),
                        float(o[0]), float(o[1]), dot_prev, flip,
                        float(gc[0]), float(gc[1]), float(go[0]), float(go[1]),
                        float(pc[0]), float(pc[1]), float(po[0]), float(po[1]),
                        lab, prob])

    # Save GIF for main viz-mode
    if len(frames) > 1:
        gif_suffix = 'denoise' if draw_mode == 'denoised' else 'state'
        gif_path = osp.join(out_dir, f'steps_{gif_suffix}.gif')
        try:
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000.0/max(1,args.gif_fps)), loop=0)
        except Exception:
            pass

    # Optional: draw the other mode into a side folder
    if bool(getattr(args, 'viz_both', False)):
        alt_mode = 'state' if draw_mode == 'denoised' else 'denoised'
        alt_dir = 'steps_state' if alt_mode == 'state' else 'steps_denoised'
        alt_steps_dir = osp.join(out_dir, alt_dir)
        os.makedirs(alt_steps_dir, exist_ok=True)
        frames2: List[Image.Image] = []
        for si in range(len(states)):
            if alt_mode == 'denoised' and si > 0 and (si - 1) < len(preds):
                xt2 = preds[si - 1][0][0].detach().cpu().numpy()
                lab2 = None
                if len(preds[si - 1]) >= 3 and preds[si - 1][2] is not None:
                    sem_np = preds[si - 1][2][0].detach().cpu().numpy()
                    lab2 = np.argmax(sem_np, axis=-1).astype(np.int64)
            else:
                xt2 = states[si][0].detach().cpu().numpy()
                lab2 = None
                if si == 0:
                    try:
                        lab2 = input_labels if isinstance(input_labels, np.ndarray) else input_labels_t[0].detach().cpu().numpy()
                    except Exception:
                        lab2 = None
                elif (si - 1) < len(preds) and len(preds[si - 1]) >= 3 and preds[si - 1][2] is not None:
                    sem_np = preds[si - 1][2][0].detach().cpu().numpy()
                    lab2 = np.argmax(sem_np, axis=-1).astype(np.int64)
            mask2 = np.ones((N, P), dtype=bool)
            for i in keep:
                mask2[i, :] = False
            title2 = f'{alt_mode} step {si:02d}/{len(states)-1}' if len(states) > 1 else f'{alt_mode} step 00'
            out2 = osp.join(alt_steps_dir, f'step_{si:03d}.png')
            overlay_slots_annot(out2, ras, bounds, slots=xt2, mask=mask2, labels=lab2, title=title2)
            try:
                frames2.append(Image.open(out2).convert('RGB'))
            except Exception:
                pass
        if len(frames2) > 1:
            alt_suffix = 'state' if alt_mode == 'state' else 'denoise'
            gif2 = osp.join(out_dir, f'steps_{alt_suffix}.gif')
            try:
                frames2[0].save(gif2, save_all=True, append_images=frames2[1:], duration=int(1000.0/max(1,args.gif_fps)), loop=0)
            except Exception:
                pass

    # Optional: save refined and input proposals as PKL in meters for downstream evaluation
    try:
        def _denorm_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
            minx, miny, maxx, maxy = [float(v) for v in bounds]
            w = max(maxx - minx, 1e-6); h = max(maxy - miny, 1e-6)
            out = np.empty_like(xy, dtype=np.float32)
            out[:, 0] = (xy[:, 0] + 1.0) * 0.5 * w + minx
            out[:, 1] = (xy[:, 1] + 1.0) * 0.5 * h + miny
            return out
        def _maptr_to_orig(lab: int) -> int:
            return 1 if lab == 0 else (0 if lab == 1 else 2)
        # Save refined (meters)
        ref_dict = {0: [], 1: [], 2: []}
        labs_for_save = sem_labels if sem_labels is not None else input_labels
        for i in keep:
            lab = int(labs_for_save[i]) if labs_for_save is not None and i < len(labs_for_save) else -1
            if lab < 0:
                continue
            orig = _maptr_to_orig(lab)
            ref_dict[orig].append(_denorm_xy(coords_np[i], bounds))
        ref_dict['bounds'] = [float(v) for v in bounds]
        import pickle
        with open(osp.join(out_dir, f'{args.scene}_refined_m.pkl'), 'wb') as f:
            pickle.dump(ref_dict, f)
        # Save input proposal (meters)
        in_dict = {0: [], 1: [], 2: []}
        for i in range(N):
            if not bool(prop_present[i]):
                continue
            lab = int(input_labels[i]) if input_labels is not None and i < len(input_labels) else -1
            if lab < 0:
                continue
            orig = _maptr_to_orig(lab)
            in_dict[orig].append(_denorm_xy(x_prop[i], bounds))
        in_dict['bounds'] = [float(v) for v in bounds]
        with open(osp.join(out_dir, f'{args.scene}_input_m.pkl'), 'wb') as f:
            pickle.dump(in_dict, f)
    except Exception as e:
        if bool(getattr(args, 'debug', False)):
            print(f'[warn] save pkl failed: {e}')

    # Optional: report simple Chamfer metrics vs GT and vs input (meters)
    try:
        import pickle
        from shapely.geometry import LineString  # type: ignore
        def _sample_line(arr: np.ndarray, num: int = 100) -> np.ndarray:
            ls = LineString(arr)
            if ls.length <= 1e-8:
                p = np.array(ls.coords[0], dtype=np.float32)
                return np.tile(p[None, :], (num, 1))
            dists = np.linspace(0.0, ls.length, num=num, dtype=np.float32)
            pts = [list(ls.interpolate(float(d)).coords)[0] for d in dists]
            return np.asarray(pts, dtype=np.float32)
        def _stack_points(bank: Dict[int, List[np.ndarray]], samples_per_line: int) -> Dict[int, np.ndarray]:
            out: Dict[int, np.ndarray] = {}
            for cls_id in (0, 1, 2):
                pts_list: List[np.ndarray] = []
                for arr in bank.get(cls_id, []) or []:
                    if arr is None or len(arr) == 0:
                        continue
                    pts_list.append(_sample_line(np.asarray(arr, dtype=np.float32), num=samples_per_line))
                out[cls_id] = np.concatenate(pts_list, axis=0) if pts_list else np.zeros((0, 2), dtype=np.float32)
            return out
        def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
            if a.shape[0] == 0 or b.shape[0] == 0:
                return float('nan')
            # blockwise nearest neighbor mean
            def nn_mean(src: np.ndarray, dst: np.ndarray, blk: int = 4096) -> float:
                n = src.shape[0]
                mins = np.empty((n,), dtype=np.float32)
                for i in range(0, n, blk):
                    s = src[i:i+blk]
                    d2 = ((s[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
                    mins[i:i+blk] = np.sqrt(d2.min(axis=1))
                return float(mins.mean())
            return 0.5 * (nn_mean(a, b) + nn_mean(b, a))
        # Load GT
        gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
        gt_bank = {0: gt.get(0, []), 1: gt.get(1, []), 2: gt.get(2, [])}
        # Load saved refined/input
        ref_bank = ref_dict  # from above
        in_bank = in_dict
        M = 100
        G = _stack_points(gt_bank, M)
        R = _stack_points(ref_bank, M)
        I = _stack_points(in_bank, M)
        def _report(name: str, A: Dict[int, np.ndarray], B: Dict[int, np.ndarray]) -> Dict[str, float]:
            res: Dict[str, float] = {}
            for nm, cid in [('ped', 0), ('div', 1), ('bnd', 2)]:
                res[f'{name}_{nm}_Chamfer_m'] = _chamfer(A[cid], B[cid])
            a_all = np.concatenate([A[0], A[1], A[2]], axis=0) if any([A[k].size for k in (0,1,2)]) else np.zeros((0,2), np.float32)
            b_all = np.concatenate([B[0], B[1], B[2]], axis=0) if any([B[k].size for k in (0,1,2)]) else np.zeros((0,2), np.float32)
            res[f'{name}_all_Chamfer_m'] = _chamfer(a_all, b_all)
            return res
        res_ref_vs_gt = _report('Ref_vs_GT', R, G)
        res_ref_vs_in = _report('Ref_vs_Input', R, I)
        # Print concise summary
        print('# Quantitative (meters) — symmetric Chamfer (lower is better)')
        for k in ['Ref_vs_GT_all_Chamfer_m','Ref_vs_Input_all_Chamfer_m',
                  'Ref_vs_GT_ped_Chamfer_m','Ref_vs_GT_div_Chamfer_m','Ref_vs_GT_bnd_Chamfer_m',
                  'Ref_vs_Input_ped_Chamfer_m','Ref_vs_Input_div_Chamfer_m','Ref_vs_Input_bnd_Chamfer_m']:
            v = res_ref_vs_gt.get(k, None)
            if v is None:
                v = res_ref_vs_in.get(k, float('nan'))
            print(f'{k}: {v:.4f}' if (isinstance(v, float) and v==v) else f'{k}: nan')
    except Exception as e:
        if bool(getattr(args, 'debug', False)):
            print(f'[warn] metrics computation failed: {e}')

    print(f"[ok] inference done. Output dir: {out_dir}")


if __name__ == '__main__':
    main()
