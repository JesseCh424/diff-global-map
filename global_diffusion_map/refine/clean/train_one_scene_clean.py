#!/usr/bin/env python
from __future__ import annotations

"""
Clean single-scene refine training (EDM, one scene) — GT+noise with hybrid delete/create.

Design
- Input (x): pack GT vectors to fixed N slots, then apply small jitter (alpha=0.03 SDEdit start)
- Hybrid: randomly drop a fraction of GT instances (simulate create) and add a few ghost polylines (simulate delete)
- Matching:
  * When input is GT+small-noise for a slot, keep identity mapping and preserve point order (x1..xP)
  * For dropped GT, assign a free slot via GPU greedy matcher (geometry-only), with forward/reverse direction choice
  * For ghosts, targets are background (present=0)
- Loss: presence BCE/Focal, orientation-agnostic L1 for coords on matched slots, optional semantic CE on matched
- EDM: Karras schedule with Heun/Euler, alpha=0.03 (SDEdit start) around x

This script is intentionally minimal and self-contained, reusing existing building blocks
from refine/ (packers, model, EDM, viz) to avoid duplication.
"""

import argparse
import subprocess
import time
import os
import json
import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
import torch

# Performance: enable TF32 on Ampere+ for faster matmul/convs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.loss_refine import criterion, gpu_greedy_match, hungarian_match_perm
from global_diffusion_map.refine.single_scene_overfit import (
    RasterEncoder,
    overlay_slots_annot,
    load_pickle,
)
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _rand_ghost(P: int, scale: float = 0.6) -> np.ndarray:
    pts = (np.random.rand(P, 2).astype(np.float32) * 2.0 - 1.0) * scale
    # light smoothing along sequence
    for k in range(1, P):
        pts[k] = 0.7 * pts[k] + 0.3 * pts[k - 1]
    return np.clip(pts, -1.0, 1.0)


def jitter_drop_ghost(
    gt_pack: np.ndarray,   # [N,P,2] normalized
    gt_mask: np.ndarray,   # [N,P] True=pad
    shift_sigma: float = 0.10,   # global shift per poly (structure-preserving)
    point_sigma: float = 0.02,   # small local noise to avoid being too rigid
    drop_frac: float = 0.15,
    ghosts: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build input proposals from GT with structure-preserving jitter, drop, and ghosts.
    Returns (x_in [N,P,2], keep_identity [N] bool, is_drop [N] bool, is_ghost [N] bool) — keep_identity=True means we
    should keep identity mapping (no re-matching) and preserve point order.
    """
    N, P, _ = gt_pack.shape
    x = gt_pack.copy()
    valid = ~gt_mask  # [N,P]
    # A) global shift per line (rigid)
    shifts = np.random.normal(scale=shift_sigma, size=(N, 1, 2)).astype(np.float32)
    # B) small local noise per point (smooth)
    local = np.random.normal(scale=point_sigma, size=x.shape).astype(np.float32)
    noise_total = shifts + local
    x[valid] = np.clip(x[valid] + noise_total[valid], -1.0, 1.0)
    # present per GT
    present_gt = (~gt_mask).any(axis=1)  # [N]
    ids = np.where(present_gt)[0].tolist()
    np.random.shuffle(ids)
    k_drop = max(0, int(round(len(ids) * float(drop_frac))))
    drop_ids = set(ids[:k_drop])
    # identity mask = GT present AND not dropped
    keep_identity = present_gt.copy()
    is_drop = np.zeros((N,), dtype=bool)
    for i in drop_ids:
        keep_identity[i] = False
        is_drop[i] = True
        # near-GT noisy polyline to enable creation training within gate
        rnd = np.random.normal(scale=max(shift_sigma, point_sigma) * 0.8, size=(P, 2)).astype(np.float32)
        x[i] = np.clip(gt_pack[i] + rnd, -1.0, 1.0)
    # ghosts in empty slots (GT absent)
    empty_ids = np.where(~present_gt)[0].tolist()
    np.random.shuffle(empty_ids)
    is_ghost = np.zeros((N,), dtype=bool)
    for j in empty_ids[:max(0, int(ghosts))]:
        x[j] = _rand_ghost(P)
        is_ghost[j] = True
    return x, keep_identity, is_drop, is_ghost


def build_fixed_targets(
    x_in: torch.Tensor,         # [N,P,2] proposals on CUDA
    gt_coords: torch.Tensor,    # [N,P,2] packed GT on CUDA
    gt_mask: torch.Tensor,      # [N,P] padding mask on CUDA
    gt_present: torch.Tensor,   # [N]
    keep_identity_mask: torch.Tensor,  # [N] True=keep identity mapping (small-noise case)
    max_center_dist: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct fixed mapping targets per batch sample (single scene => 1 sample here).
    - identity for keep slots (preserve order/points)
    - greedy match for the rest (assign unmatched GT to best free slots)
    - forward/reverse choice to avoid flips
    Returns (tgt_coords [N,P,2], tgt_mask [N,P], tgt_present [N], tgt_labels [N] with -1 ignored)
    """
    device = x_in.device
    N, P, _ = x_in.shape
    # Initialize all as background
    tgt_coords = torch.zeros_like(gt_coords)
    tgt_mask_o = torch.ones_like(gt_mask, dtype=torch.bool)
    tgt_present_o = torch.zeros_like(gt_present)
    match_idx = torch.full_like(gt_present, fill_value=-1, dtype=torch.long)

    # 1) identity for keep slots (gt_present=1 and keep_identity_mask=1)
    #    Make identity orientation permutation-invariant: choose fwd/rev by lower masked L1
    keep_ids = torch.nonzero(keep_identity_mask & (gt_present > 0), as_tuple=False).view(-1)
    if int(keep_ids.numel()) > 0:
        x_sel = x_in.index_select(0, keep_ids)         # [K,P,2]
        g_sel = gt_coords.index_select(0, keep_ids)    # [K,P,2]
        m_sel = gt_mask.index_select(0, keep_ids)      # [K,P]
        v = (~m_sel).float()
        # forward masked mean L1
        diff_f = (x_sel - g_sel).abs().sum(dim=-1)     # [K,P]
        denom = v.sum(dim=1).clamp_min(1.0)            # [K]
        l1_f = (diff_f * v).sum(dim=1) / denom         # [K]
        # reverse masked mean L1
        g_rev = torch.flip(g_sel, dims=[1])
        m_rev = torch.flip(m_sel, dims=[1])
        diff_r = (x_sel - g_rev).abs().sum(dim=-1)
        v_rev = (~m_rev).float()
        l1_r = (diff_r * v_rev).sum(dim=1) / denom     # denom same count
        use_rev = (l1_r < l1_f).view(-1)               # [K]
        # assign chosen orientation
        g_final = g_sel.clone()
        m_final = m_sel.clone()
        if bool(use_rev.any()):
            g_final[use_rev] = g_rev[use_rev]
            m_final[use_rev] = m_rev[use_rev]
        tgt_coords.index_copy_(0, keep_ids, g_final)
        tgt_mask_o.index_copy_(0, keep_ids, m_final)
        tgt_present_o.index_copy_(0, keep_ids, torch.ones_like(keep_ids, dtype=tgt_present_o.dtype))
        # match identity indices
        for idx in keep_ids.tolist():
            match_idx[idx] = int(idx)

    # 2) greedy match for remaining GT (those not covered by identity)
    # mask to indicate which GT instances still need assignment
    remaining_gt = (gt_present > 0).clone()
    remaining_gt[keep_ids] = False
    if bool(remaining_gt.any()):
        # Hungarian matching with permutation-invariant regression (forward/reverse)
        gt_sel_idx = torch.nonzero(remaining_gt, as_tuple=False).view(-1)
        if int(gt_sel_idx.numel()) > 0:
            gt_sel_coords = gt_coords.index_select(0, gt_sel_idx)
            gt_sel_mask = gt_mask.index_select(0, gt_sel_idx)
            pairs, perm_choice = hungarian_match_perm(
                x_in, None, gt_sel_coords, gt_sel_mask, None, cls_weight=0.0, reg_weight=50.0, use_l1_beta=0.0
            )
            assigned_slots = set(keep_ids.tolist())
            for (pi, gj_loc) in pairs:
                pi_i = int(pi)
                if pi_i in assigned_slots:
                    continue
                gj_global = int(gt_sel_idx[int(gj_loc)].item())
                if not bool(remaining_gt[gj_global]):
                    continue
                # Apply chosen orientation (0=forward, 1=reverse)
                k = int(perm_choice[int(gj_loc)]) if (perm_choice is not None and len(perm_choice) > int(gj_loc)) else 0
                if k == 1:
                    tgt_coords[pi_i] = torch.flip(gt_coords[gj_global], dims=[0])
                    tgt_mask_o[pi_i] = torch.flip(gt_mask[gj_global], dims=[0])
                else:
                    tgt_coords[pi_i] = gt_coords[gj_global]
                    tgt_mask_o[pi_i] = gt_mask[gj_global]
                tgt_present_o[pi_i] = 1
                remaining_gt[gj_global] = False
                assigned_slots.add(pi_i)
                match_idx[pi_i] = int(gj_global)

    return tgt_coords, tgt_mask_o, tgt_present_o, match_idx

def _labels_from_budgets(budgets: Dict[int, int], N: int) -> np.ndarray:
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
    ap = argparse.ArgumentParser(description='Clean one-scene refine training (GT+noise hybrid)')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--prior-lr-mult', type=float, default=2.0, help='LR multiplier for prior/class branches to speed up early learning')
    # Device control
    ap.add_argument('--gpu', type=str, default=None, help='CUDA_VISIBLE_DEVICES value (e.g., "8" to use GPU index 8)')
    # LR scheduler (default: cosine)
    ap.add_argument('--sched', choices=['cosine', 'none'], default='cosine')
    ap.add_argument('--lr-min', type=float, default=1e-5, help='min LR for cosine')
    # EDM schedule
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.03, help='SDEdit blend weight: xK=(1-a)*x + a*N')
    # augmentation (normalized coords)
    # Lower noise to avoid cross-lane samples
    ap.add_argument('--shift-sigma', type=float, default=0.10, dest='shift_sigma', help='global per-line shift sigma (structure-preserving)')
    ap.add_argument('--point-sigma', type=float, default=0.02, dest='point_sigma', help='small local noise sigma per point (keeps smoothness)')
    ap.add_argument('--jitter', type=float, default=0.05, help='[deprecated] kept for backward compat; prefer shift/point sigma')
    ap.add_argument('--drop-frac', type=float, default=0.15)
    ap.add_argument('--ghosts', type=int, default=2)
    # loss weights
    ap.add_argument('--l1-weight', type=float, default=20.0)
    ap.add_argument('--cls-weight', type=float, default=5.0)
    ap.add_argument('--use-focal', action='store_true')
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    ap.add_argument('--step-loss-weight', type=float, default=1.0)
    ap.add_argument('--final-loss-weight', type=float, default=1.0)
    # (reverted) asymmetry loss flag removed
    # mapping gate
    ap.add_argument('--anchor-max-center-dist', type=float, default=0.05)
    # io
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_one_scene_clean')
    ap.add_argument('--viz-thr', type=float, default=0.5)
    # Orientation-agnostic vector loss (min over forward/reverse), MapTR-style LinesL1
    ap.add_argument('--vec-loss', action='store_true', default=True,
                    help='enable permutation-invariant vector loss (min over forward/reverse)')
    ap.add_argument('--save-every', type=int, default=50, help='save ckpt and viz every N epochs')
    # Gradient accumulation to stabilize per-epoch update with on-the-fly noise
    ap.add_argument('--accum-steps', type=int, default=1,
                    help='Accumulate gradients over this many micro-batches before optimizer.step()')
    # Physical batching to fully utilize GPU (parallel samples per step)
    ap.add_argument('--batch-size', type=int, default=40,
                    help='Physical batch size (number of parallel randomized samples per epoch/micro-batch)')
    # Auto inference after training
    ap.add_argument('--auto-infer', action='store_true', help='Run inference after training (GT+noise and proposal if available)')
    ap.add_argument('--infer-steps', type=int, default=10)
    ap.add_argument('--infer-sigma-min', type=float, default=0.002)
    ap.add_argument('--infer-sigma-max', type=float, default=1.5)
    ap.add_argument('--infer-second-order', action='store_true', help='Use Heun second-order update at inference')
    ap.add_argument('--infer-out-root', default='global_diffusion_map/refine/work_dirs/infer_dual_modes',
                    help='Root dir for auto inference outputs (a timestamp will be appended)')
    ap.add_argument('--agg-pred-root', required=False,
                    help='Aggregated prediction root to enable proposal-mode inference (per-scene .pkl inside)')
    args = ap.parse_args()

    # Optional GPU control before any CUDA device query
    if args.gpu is not None and len(str(args.gpu)) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    set_seed(0)
    # Enable orientation-agnostic vector loss by default (can be disabled via --no-vec-loss)
    if bool(getattr(args, 'vec_loss', True)):
        os.environ.setdefault('REFINE_VEC_LOSS', '1')
    else:
        os.environ['REFINE_VEC_LOSS'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # stats: caps + class budgets
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_points=P, num_queries=N)

    # load static GT (bounds + vectors)
    gt_pkl = osp.join(args.static_root, f'{args.scene}.pkl')
    gt = load_pickle(gt_pkl)
    bounds = gt.get('bounds')
    if bounds is None:
        raise RuntimeError('static GT lacks canonical bounds')
    # pack GT to slots (MapTR budgets order)
    gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, num_points=P, num_queries=N)
    # raster (10_render_gt.png)
    from PIL import Image
    ras = np.asarray(Image.open(osp.join(args.rendered_root, args.scene, '10_render_gt.png')).convert('RGB'), dtype=np.float32) / 255.0
    ras = torch.from_numpy(ras.transpose(2, 0, 1)).float().unsqueeze(0).to(device)  # [1,3,H,W]
    # (reverted) no raster flips applied

    # Preload GT tensors on device (reused across epoch/micro-batches)
    gt_coords_gpu = torch.from_numpy(gt_pack).float().to(device)
    gt_mask_gpu = torch.from_numpy(gt_mask).bool().to(device)
    gt_present_gpu = torch.from_numpy((~gt_mask).any(axis=1).astype(np.int64)).long().to(device)

    # model
    enc = RasterEncoder(out_dim=256).to(device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    # 为先验分支设置更高学习率（加速门控与先验编码学习）
    prior_params: List[torch.nn.Parameter] = []
    # 形状先验编码 + 门控
    prior_params += list(net.backbone.prior_mlp.parameters())
    prior_params += [net.backbone.prior_gate]
    # 类别先验编码 + 门控（若启用）
    if getattr(net.backbone, 'class_emb', None) is not None:
        prior_params += list(net.backbone.class_emb.parameters())
        prior_params += [net.backbone.class_gate]
    prior_ids = {id(p) for p in prior_params}
    # 其余参数（backbone 其他 + encoder）
    rest_backbone = [p for p in net.backbone.parameters() if id(p) not in prior_ids]
    enc_params = list(enc.parameters())
    param_groups = [
        {'params': enc_params + rest_backbone, 'lr': float(args.lr), 'weight_decay': 1e-4},
        {'params': prior_params, 'lr': float(args.lr) * float(getattr(args, 'prior_lr_mult', 2.0)), 'weight_decay': 1e-4},
    ]
    opt = torch.optim.AdamW(param_groups)
    # scheduler
    if args.sched == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=float(args.lr_min))
    else:
        sched = None
    out_dir = osp.join(args.out_root, args.scene)
    viz_dir = osp.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    for ep in range(1, int(args.epochs) + 1):
        enc.train(); base.train()
        accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
        B = max(1, int(getattr(args, 'batch_size', 1)))
        # Track epoch-averaged losses over micro-batches
        ep_loss_sum = 0.0
        ep_step_sum = 0.0
        ep_final_sum = 0.0
        # Cache for viz of the last micro-batch
        last_x_np = None  # type: ignore
        last_keep_identity_np = None  # type: ignore
        last_is_drop_np = None  # type: ignore
        last_labels_by_slot = None  # type: ignore
        preds = None  # type: ignore
        pred_coords = pred_logits = None  # type: ignore
        sigmas = None  # type: ignore
        opt.zero_grad(set_to_none=True)

        # Encode raster once and expand to B
        with torch.cuda.amp.autocast(enabled=False):
            rv_single = enc(ras)  # [1,256]
        # Expand without additional memory
        rv_expand = rv_single.expand(B, -1)

        for micro in range(accum_steps):
            # Build a physical batch of randomized samples
            batch_x: List[torch.Tensor] = []
            batch_tgt_coords: List[torch.Tensor] = []
            batch_tgt_mask: List[torch.Tensor] = []
            batch_tgt_present: List[torch.Tensor] = []
            batch_tgt_labels: List[torch.Tensor] = []
            batch_prior_labels: List[torch.Tensor] = []

            for b in range(B):
                x_np, keep_identity_np, is_drop_np, _is_ghost_np = jitter_drop_ghost(
                    gt_pack, gt_mask,
                    shift_sigma=float(getattr(args, 'shift_sigma', 0.10)),
                    point_sigma=float(getattr(args, 'point_sigma', 0.02)),
                    drop_frac=float(args.drop_frac), ghosts=int(args.ghosts))

                x_b = torch.from_numpy(x_np).float().to(device)  # [N,P,2]
                gt_coords_b = gt_coords_gpu
                gt_mask_b = gt_mask_gpu
                gt_present_b = gt_present_gpu

                with torch.no_grad():
                    keep_identity = torch.from_numpy(keep_identity_np).to(device)  # [N]
                tgt_coords_b, tgt_mask_b, tgt_present_b, match_idx = build_fixed_targets(
                    x_b, gt_coords_b, gt_mask_b, gt_present_b, keep_identity_mask=keep_identity,
                    max_center_dist=float(args.anchor_max_center_dist))

                labels_all = _labels_from_budgets(budgets, N)
                labels_by_slot = np.full((N,), -1, dtype=np.int64)
                mi = match_idx.detach().cpu().numpy()
                for i in range(N):
                    j = int(mi[i])
                    if j >= 0 and j < N:
                        labels_by_slot[i] = labels_all[j]
                tgt_labels_b = torch.from_numpy(labels_by_slot).long().to(device)
                input_prior_np = labels_by_slot.copy()
                input_prior_np[~keep_identity_np] = -1
                input_prior_labels_b = torch.from_numpy(input_prior_np).long().to(device)

                batch_x.append(x_b)
                batch_tgt_coords.append(tgt_coords_b)
                batch_tgt_mask.append(tgt_mask_b)
                batch_tgt_present.append(tgt_present_b)
                batch_tgt_labels.append(tgt_labels_b)
                batch_prior_labels.append(input_prior_labels_b)

                # Save last sample of the micro-batch for viz
                if b == B - 1:
                    last_x_np = x_np
                    last_keep_identity_np = keep_identity_np
                    last_is_drop_np = is_drop_np
                    last_labels_by_slot = labels_by_slot

            # Stack batch tensors
            x = torch.stack(batch_x, dim=0)  # [B,N,P,2]
            tgt_coords = torch.stack(batch_tgt_coords, dim=0)
            tgt_mask_o = torch.stack(batch_tgt_mask, dim=0)
            tgt_present_o = torch.stack(batch_tgt_present, dim=0)
            tgt_labels = torch.stack(batch_tgt_labels, dim=0)
            input_prior_labels = torch.stack(batch_prior_labels, dim=0)

            # SDEdit start (add noise to proposals)
            xK = torch.clamp((1.0 - float(args.alpha)) * x + float(args.alpha) * torch.randn_like(x), -1.0, 1.0)
            sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
            pred_coords, pred_logits, preds, _states = edm_unrolled_train(
                net, xK, rv_expand, sigmas, second_order=bool(args.second_order),
                cond_prior=x, input_labels=input_prior_labels)

            # Loss over batch
            step_losses = []
            for item in preds:
                pc, pl = item[0], item[1]
                sem_step = item[2] if (len(item) >= 3) else None
                out = criterion(pc, pl, tgt_coords, tgt_mask_o, tgt_present_o,
                                l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                                use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                pred_sem_logits=sem_step, tgt_sem_labels=tgt_labels, sem_weight=float(args.sem_weight))
                step_losses.append(out['loss_cls'] + out['loss_reg'] + out.get('loss_sem', pc.new_zeros([])))
            step_loss = torch.stack(step_losses).mean() if step_losses else pred_coords.new_zeros([])
            out_final = criterion(pred_coords, pred_logits, tgt_coords, tgt_mask_o, tgt_present_o,
                                  l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                                  use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                  pred_sem_logits=(preds[-1][2] if (len(preds) > 0 and len(preds[-1]) >= 3) else None),
                                  tgt_sem_labels=tgt_labels, sem_weight=float(args.sem_weight))
            final_loss = out_final['loss_cls'] + out_final['loss_reg'] + out_final.get('loss_sem', pred_coords.new_zeros([]))
            loss = float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss

            retain = (micro < accum_steps - 1)
            (loss / float(accum_steps)).backward(retain_graph=retain)
            ep_loss_sum += float(loss.item())
            ep_step_sum += float(step_loss.item())
            ep_final_sum += float(final_loss.item())

        # Update once after accumulation
        opt.step()

        # logging
        if (ep % 20 == 0) or (ep == 1):
            cur_lr = opt.param_groups[0]['lr']
            avg_loss = ep_loss_sum / float(accum_steps)
            avg_step = ep_step_sum / float(accum_steps)
            avg_final = ep_final_sum / float(accum_steps)
            print(f"[ep {ep:04d}] steps={len(sigmas) if sigmas is not None else 0} lr={cur_lr:.6g} batch={B} accum={accum_steps} loss={avg_loss:.6f} step={avg_step:.6f} final={avg_final:.6f}")
        # viz + ckpt (every N epochs)
        if (ep % int(args.save_every) == 0) or (ep == int(args.epochs)):
            with torch.no_grad():
                # Noise visualization: draw only identity+drop (用于几何学习)，隐藏纯 ghost
                Nviz_all = last_x_np.shape[0] if isinstance(last_x_np, np.ndarray) else N
                mask_all = np.ones((Nviz_all, P), dtype=bool)
                draw_ids = np.where(last_keep_identity_np | last_is_drop_np)[0].tolist() if isinstance(last_keep_identity_np, np.ndarray) else list(range(Nviz_all))
                for i in draw_ids:
                    mask_all[i, :] = False
                title_noise = f"noise ep{ep:04d} shift={getattr(args,'shift_sigma',0.10):.2f} local={getattr(args,'point_sigma',0.02):.2f} drop={args.drop_frac:.2f} ghosts={int(args.ghosts)}"
                overlay_slots_annot(osp.join(viz_dir, f'noise_{ep:04d}.png'), ras[0].detach().cpu().numpy(), bounds,
                                    slots=last_x_np, mask=mask_all, labels=last_labels_by_slot, title=title_noise,
                                    gt_slots=gt_pack, gt_mask=gt_mask)
                # Visualize last item in the batch
                pc = pred_coords[-1].detach().cpu().numpy()
                pl = pred_logits[-1].detach().cpu().numpy().reshape(-1)
                prob = 1.0 / (1.0 + np.exp(-pl))
                keep = np.where(prob >= float(args.viz_thr))[0].tolist()
                Nviz = pc.shape[0]
                mask = np.ones((Nviz, P), dtype=bool)
                for i in keep:
                    if 0 <= i < Nviz:
                        mask[i, :] = False
                # semantic labels from last step if available
                sem_labels = None
                if len(preds) > 0 and len(preds[-1]) >= 3 and preds[-1][2] is not None:
                    sem_np = preds[-1][2][-1].detach().cpu().numpy()
                    sem_labels = np.argmax(sem_np, axis=-1).astype(np.int64)
                overlay_slots_annot(osp.join(viz_dir, f'ep_{ep:04d}.png'), ras[0].detach().cpu().numpy(), bounds,
                                    slots=pc, mask=mask, labels=sem_labels, title=f'ep_{ep:04d} (thr={args.viz_thr})')
                # ckpt
                os.makedirs(out_dir, exist_ok=True)
                torch.save({'encoder': (enc.state_dict()), 'net': (base.state_dict()), 'P': P, 'N': N, 'budgets': budgets},
                           osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth'))

        # scheduler step
        if sched is not None:
            sched.step()

    print(f"[ok] clean one-scene training done. Viz: {viz_dir}")

    # Auto inference: run clean inference script in two modes if requested
    if bool(getattr(args, 'auto_infer', False)):
        # Choose final checkpoint (prefer exact final epoch; fallback to latest by name)
        ckpt_name = f'ckpt_ep_{int(args.epochs):04d}.pth'
        ckpt_path = osp.join(out_dir, ckpt_name)
        if not osp.exists(ckpt_path):
            cand = [f for f in os.listdir(out_dir) if f.startswith('ckpt_ep_') and f.endswith('.pth')]
            ckpt_path = osp.join(out_dir, sorted(cand)[-1]) if cand else ''
        if not ckpt_path or not osp.exists(ckpt_path):
            print('[warn] auto-infer skipped: no checkpoint found')
            return

        ts = time.strftime('%Y%m%d_%H%M%S')
        infer_root = f"{args.infer_out_root}_{ts}"
        os.makedirs(infer_root, exist_ok=True)
        log_path = osp.join(out_dir, 'infer_watch.log')
        env = os.environ.copy()
        cmd_base = [
            sys.executable, '-u', 'global_diffusion_map/refine/clean/infer_one_scene_clean.py',
            '--static-root', args.static_root,
            '--rendered-root', args.rendered_root,
            '--stats-json', args.stats_json,
            '--scene', args.scene,
            '--ckpt', ckpt_path,
            '--steps', str(int(args.infer_steps)),
            '--sigma-min', str(float(args.infer_sigma_min)),
            '--sigma-max', str(float(args.infer_sigma_max)),
        ]
        if bool(getattr(args, 'infer_second_order', False)):
            cmd_base.append('--second-order')
        # A) GT+noise mode
        cmd_gt = cmd_base + ['--start', 'gt_noise', '--out-root', osp.join(infer_root, 'gt_noise'), '--csv-move']
        # B) Proposal mode (only if aggregated predictions provided)
        do_prop = isinstance(args.agg_pred_root, str) and len(args.agg_pred_root) > 0
        cmd_prop = None
        if do_prop:
            cmd_prop = cmd_base + [
                '--start', 'proposal',
                '--agg-pred-root', args.agg_pred_root,
                '--out-root', osp.join(infer_root, 'proposal'), '--csv-move'
            ]
        # Run and tee outputs to a log file
        with open(log_path, 'a', encoding='utf-8') as flog:
            flog.write(f"[watch] start auto-infer at {ts}\n")
            flog.flush()
            try:
                flog.write(f"[cmd] {' '.join(cmd_gt)}\n")
                flog.flush()
                subprocess.run(cmd_gt, check=True, env=env, stdout=flog, stderr=flog)
                if cmd_prop is not None:
                    flog.write(f"[cmd] {' '.join(cmd_prop)}\n")
                    flog.flush()
                    subprocess.run(cmd_prop, check=True, env=env, stdout=flog, stderr=flog)
                flog.write('[watch] inference done\n')
            except subprocess.CalledProcessError as e:
                flog.write(f"[error] auto-infer failed with return code {e.returncode}\n")
        print(f"[ok] auto-infer finished. Logs: {log_path}  Outputs: {infer_root}")


if __name__ == '__main__':
    main()
