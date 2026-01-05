#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import os.path as osp
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps
from global_diffusion_map.refine.loss_refine import criterion, HungarianMatcher, gpu_greedy_match
from global_diffusion_map.refine.augment import augment_planA_from_gt_torch
from global_diffusion_map.refine.single_scene_overfit import RasterEncoder, overlay_on_raster, load_pickle
from global_diffusion_map.refine.single_scene_dataset import SingleSceneDataset, DynamicOverfitDataset
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser(description='EDM multistep refine training on one scene (xK+GT, no proposals)')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', required=True, help='only used for bounds fallback if static lacks bounds')
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--epochs', type=int, default=600)
    # DDP
    ap.add_argument('--ddp', action='store_true', help='enable DistributedDataParallel (torchrun)')
    ap.add_argument('--local_rank', type=int, default=0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--hybrid', action='store_true', help='use easy+dynamic (batch=2) hybrid per step')
    # EDM schedule
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=1.5)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.03, help='SDEdit blend weight for xK (xK=(1-a)*x + a*N)')
    # Loss
    ap.add_argument('--cls-weight', type=float, default=10.0)
    ap.add_argument('--use-focal', action='store_true', default=True)
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    ap.add_argument('--smooth-weight', type=float, default=0.05, help='curvature/smoothness regularization weight')
    ap.add_argument('--reg-len-exp', type=float, default=0.0, help='exponent for length-proportional weighting of regression loss (0=off, 1=proportional)')
    ap.add_argument('--smooth-inv-len-exp', type=float, default=0.0, help='exponent for inverse-length weighting of smooth loss (0=off, 1=proportional)')
    ap.add_argument('--dir-weight', type=float, default=0.2, help='directional consistency loss weight')
    ap.add_argument('--step-loss-weight', type=float, default=1.0, help='weight for per-step denoise loss')
    ap.add_argument('--final-loss-weight', type=float, default=1.0, help='weight for matched final-step loss')
    # Loss scaling (coordinate regression)
    ap.add_argument('--l1-weight', type=float, default=1.0, help='L1 regression loss weight for coordinates')
    # Anchor distance gate (normalized coords) for fixed mapping
    ap.add_argument('--anchor-max-center-dist', type=float, default=0.2,
                    help='Max normalized center distance to accept a proposal↔GT pair in fixed mapping; farther pairs are treated as background')
    # Anchor-matching validation (Crossed Wires Test)
    ap.add_argument('--anchor-test-perturb', action='store_true',
                    help='Enable input perturbation to validate anchor matching (e.g., shift one lane towards another or inject heavy noise).')
    ap.add_argument('--anchor-test-mode', choices=['shift', 'noise'], default='shift',
                    help='Perturbation mode: shift one present slot towards another, or add heavy noise on proposals.')
    ap.add_argument('--anchor-shift-frac', type=float, default=0.6,
                    help='For shift mode: fraction of vector from GT_A towards GT_B (e.g., 0.6).')
    ap.add_argument('--anchor-noise-sigma', type=float, default=0.5,
                    help='For noise mode: Gaussian noise std to add on proposals before building xK.')
    # Static challenging case (freeze specific refine/delete signals for overfit diagnosis)
    ap.add_argument('--static-case', action='store_true',
                    help='在循环外构造一次固定的“移位+幽灵”输入，用于单场景稳定过拟合验证（避免 moving anchor）。')
    ap.add_argument('--static-shift-frac', type=float, default=0.2,
                    help='将一个有效槽位沿 +x 方向平移的幅度（归一化坐标），用于验证 refine。')
    ap.add_argument('--static-ghosts', type=int, default=1,
                    help='在空槽里注入的幽灵曲线数量，用于验证 delete/抑制背景。')
    ap.add_argument('--static-ghost-scale', type=float, default=0.5,
                    help='幽灵曲线采样范围幅度（[-scale, scale]）。')
    # Visualization controls
    ap.add_argument('--viz-thr', type=float, default=0.2, help='presence 概率阈值（训练 ep_xxxx 可视化与 quick-infer 统一使用）')
    ap.add_argument('--viz-honest', action='store_true', default=True, help='训练可视化不使用 GT 匹配标签过滤，仅按概率阈值展示（暴露噪声以便诊断）')
    # LR schedulers
    ap.add_argument('--sched', choices=['cosine', 'plateau', 'none'], default='cosine', help='LR scheduler type')
    ap.add_argument('--lr-min', type=float, default=2e-5, help='min LR for cosine/plateau clamp')
    ap.add_argument('--plateau-factor', type=float, default=0.5, help='ReduceLROnPlateau factor')
    ap.add_argument('--plateau-patience', type=int, default=10, help='ReduceLROnPlateau patience (epochs)')
    # IO
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_edm_one_scene')
    ap.add_argument('--resume-ckpt', type=str, default=None)
    ap.add_argument('--amp', action='store_true', help='enable mixed precision (AMP) for speed')
    args = ap.parse_args()

    # Defaults to align one-scene with full training GPU-optimized path
    os.environ.setdefault('REFINE_VEC_LOSS', '1')          # vectorized loss on
    os.environ.setdefault('REFINE_GPU_MATCH', '1')         # use GPU greedy matcher by default
    os.environ.setdefault('REFINE_GPU_CHAMFER', '1')       # include Chamfer term in GPU match cost
    os.environ.setdefault('REFINE_GPU_AUG', '1')           # build proposals on GPU (jitter/drop/ghost)
    # 训练增强默认：禁用 jitter，drop∈[0.2,0.4]，ghost∈[2,4]（可用环境变量覆盖）
    os.environ.setdefault('REFINE_AUG_JITTER', '0.0')
    os.environ.setdefault('REFINE_AUG_DROP_LO', '0.2')
    os.environ.setdefault('REFINE_AUG_DROP_HI', '0.4')
    os.environ.setdefault('REFINE_AUG_GHOST_LO', '2')
    os.environ.setdefault('REFINE_AUG_GHOST_HI', '4')
    os.environ.setdefault('OMP_NUM_THREADS', '1')          # reduce CPU contention
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    # Reduce CPU thread contention when CPU is hot
    try:
        torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
    except Exception:
        pass

    # Preflight checks to avoid confusing '.pkl' path errors
    if (args.scene is None) or (str(args.scene).strip() == ''):
        raise ValueError("--scene is empty; please pass a valid scene id (UUID)")
    if (args.static_root is None) or (str(args.static_root).strip() == ''):
        raise ValueError("--static-root is empty; expected a directory with <scene>.pkl files")
    static_pkl_path = osp.join(args.static_root, f"{args.scene}.pkl")
    if not osp.isfile(static_pkl_path):
        raise FileNotFoundError(f"Static GT pickle not found: {static_pkl_path}")

    # DDP init
    import torch.distributed as dist
    use_ddp = bool(args.ddp) or (int(os.environ.get('WORLD_SIZE', '1')) > 1)
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank)) if use_ddp else 0
    if use_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    set_seed(0 + (local_rank if use_ddp else 0))
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_queries=N, num_points=P)

    # Datasets: easy (jitter GT) and dynamic (delete/create/refine on GT)
    ds_easy = SingleSceneDataset(
        static_root=args.static_root,
        rendered_root=args.rendered_root,
        agg_pred_root=args.agg_pred_root,
        scene=args.scene,
        caps=caps,
        class_budgets=budgets,
        length=max(2000, args.epochs * 2),
        jitter_sigma_m=0.0,
        drop_rate=0.0,
        ghosts=0,
    )
    ds_dyn = DynamicOverfitDataset(
        static_root=args.static_root,
        rendered_root=args.rendered_root,
        agg_pred_root=args.agg_pred_root,
        scene=args.scene,
        caps=caps,
        class_budgets=budgets,
        length=max(2000, args.epochs * 2),
        jitter_sigma_m=0.0,
        drop_frac_range=(0.2, 0.4),
        ghosts_range=(2, 4),
    )

    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    enc = RasterEncoder(out_dim=256).to(device)
    # Enable semantic head (3 classes: divider/ped/boundary) for one-scene training
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # Some heads may be conditionally unused (e.g., semantic head in step loss); keep True for robustness
        enc = DDP(enc, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
        net = DDP(net, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=True)
    opt = optim.AdamW(list(enc.parameters()) + list(net.parameters()), lr=args.lr, weight_decay=args.wd)
    # LR scheduler
    if args.sched == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=float(args.lr_min))
    elif args.sched == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        sched = ReduceLROnPlateau(opt, mode='min', factor=float(args.plateau_factor), patience=int(args.plateau_patience), min_lr=float(args.lr_min))
    else:
        sched = None

    # Optional resume
    out_dir = osp.join(args.out_root, args.scene)
    viz_dir = osp.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    if args.resume_ckpt and osp.isfile(args.resume_ckpt):
        try:
            data = torch.load(args.resume_ckpt, map_location=device)
            if 'encoder' in data:
                enc.load_state_dict(data['encoder'], strict=False)
            if 'net' in data:
                base.load_state_dict(data['net'], strict=False)
            print(f"[resume] loaded {args.resume_ckpt}")
        except Exception as e:
            print(f"[warn] resume failed: {e}")

    matcher = HungarianMatcher(w_center=1.0, w_dir=0.2, w_pw=0.5)

    # Prefetch a fixed batch for strict one-scene overfitting (avoid moving anchors)
    print('[info] Pre-fetching fixed batch for overfitting stability...')
    easy_fixed = ds_easy[0]
    dyn_fixed = ds_dyn[0]
    if args.hybrid:
        x_b_static = torch.stack([easy_fixed['proposal'], dyn_fixed['proposal']], dim=0).to(device)
        r_b_static = torch.stack([easy_fixed['raster'], dyn_fixed['raster']], dim=0).to(device)
        tgt_c_b_static = torch.stack([easy_fixed['tgt_coords'], dyn_fixed['tgt_coords']], dim=0).to(device)
        tgt_m_b_static = torch.stack([easy_fixed['tgt_mask'], dyn_fixed['tgt_mask']], dim=0).to(device)
        tgt_p_b_static = torch.stack([easy_fixed['tgt_present'], dyn_fixed['tgt_present']], dim=0).to(device)
    else:
        x_b_static = dyn_fixed['proposal'][None, ...].to(device)
        r_b_static = dyn_fixed['raster'][None, ...].to(device)
        tgt_c_b_static = dyn_fixed['tgt_coords'][None, ...].to(device)
        tgt_m_b_static = dyn_fixed['tgt_mask'][None, ...].to(device)
        tgt_p_b_static = dyn_fixed['tgt_present'][None, ...].to(device)

    # Optionally construct a static challenging input once (refine + ghost), then freeze across epochs
    if bool(getattr(args, 'static_case', False)):
        with torch.no_grad():
            Bfix = x_b_static.shape[0]
            for bb in range(Bfix):
                present_slots = (~tgt_m_b_static[bb]).any(dim=1)
                # A) refine: shift the first present slot along +x by static_shift_frac
                ids_present = torch.nonzero(present_slots, as_tuple=False).view(-1)
                if ids_present.numel() > 0:
                    idx_ref = int(ids_present[0].item())
                    x_b_static[bb, idx_ref, :, 0] = torch.clamp(
                        x_b_static[bb, idx_ref, :, 0] + float(getattr(args, 'static_shift_frac', 0.2)), -1.0, 1.0)
                # B) ghosts: fill first K empty slots with random polylines near center
                K = int(getattr(args, 'static_ghosts', 1))
                scale = float(getattr(args, 'static_ghost_scale', 0.5))
                ids_empty = torch.nonzero(~present_slots, as_tuple=False).view(-1)
                for k in range(min(K, int(ids_empty.numel()))):
                    idx_emp = int(ids_empty[k].item())
                    Pts = x_b_static.shape[2]
                    rnd = (torch.rand((Pts, 2), device=x_b_static.device, dtype=x_b_static.dtype) * 2.0 - 1.0) * scale
                    x_b_static[bb, idx_emp] = torch.clamp(rnd, -1.0, 1.0)

    # Optionally precompute a fixed proposal→GT mapping once (anchor), to be reused every epoch
    static_tgt_coords = static_tgt_mask = static_tgt_present = static_tgt_labels = None
    if bool(getattr(args, 'static_case', False)):
        try:
            import numpy as _np
            from global_diffusion_map.refine.loss_refine import hungarian_match_perm as _HMP
            Bfix = x_b_static.shape[0]
            static_coords_list: List[torch.Tensor] = []
            static_mask_list: List[torch.Tensor] = []
            static_pres_list: List[torch.Tensor] = []
            static_labels_list: List[torch.Tensor] = []
            # Build MapTR label order to supervise semantic when available
            budgets = budgets  # keep from above
            for b in range(Bfix):
                pc0 = x_b_static[b]
                gb = tgt_c_b_static[b]
                gmb = tgt_m_b_static[b]
                gpb = tgt_p_b_static[b]
                # labels by budgets order: divider,ped,boundary → 0,1,2 (mapped as in infer)
                order: List[int] = []
                for orig in (1, 0, 2):
                    cap = int(budgets.get(orig, 0))
                    lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                    order += [lab] * max(0, cap)
                gt_labels_b = torch.as_tensor(order, dtype=torch.long, device=device)[: gb.shape[0]]
                # Hungarian (perm-invariant) without class term for fixed mapping
                pairs0, gt_perm_choice = _HMP(pc0, None, gb, gmb, gt_labels_b, cls_weight=0.0, reg_weight=50.0)
                Nq = gb.shape[0]
                new_tgt0 = torch.zeros_like(gb)
                new_msk0 = torch.ones_like(gmb, dtype=torch.bool)
                new_pre0 = torch.zeros_like(gpb)
                new_lab0 = torch.full_like(gpb, fill_value=-1, dtype=torch.long)
                valid_idx_np = _np.where(gpb.detach().cpu().numpy() > 0)[0]
                for (pi, gj_loc) in pairs0:
                    gj_global = int(gj_loc)  # Hungarian returns global idx here
                    # distance gate to avoid far-away matches
                    try:
                        c_pi = torch.nanmean(pc0[pi], dim=0)
                        c_gj = torch.nanmean(gb[gj_global], dim=0)
                        d = torch.linalg.norm(c_pi - c_gj).item()
                        if d > float(getattr(args, 'anchor_max_center_dist', 0.2)):
                            continue
                    except Exception:
                        pass
                    new_tgt0[pi] = gb[gj_global]
                    new_msk0[pi] = gmb[gj_global]
                    new_pre0[pi] = 1
                    if 0 <= gj_global < gt_labels_b.numel():
                        new_lab0[pi] = int(gt_labels_b[gj_global].item())
                static_coords_list.append(new_tgt0)
                static_mask_list.append(new_msk0)
                static_pres_list.append(new_pre0)
                static_labels_list.append(new_lab0)
            static_tgt_coords = torch.stack(static_coords_list, dim=0).to(device)
            static_tgt_mask = torch.stack(static_mask_list, dim=0).to(device)
            static_tgt_present = torch.stack(static_pres_list, dim=0).to(device)
            static_tgt_labels = torch.stack(static_labels_list, dim=0).to(device)
            print('[info] Built static fixed mapping for single-scene overfit (anchor).')
        except Exception as e:
            print(f'[warn] static mapping failed, fallback to per-epoch mapping: {e}')

    # Decide whether to use static mapping during training epochs
    use_static_mapping = bool(getattr(args, 'static_case', False)) and (static_tgt_coords is not None)
    enc.train(); base.train()
    # AMP scaler
    use_amp = bool(getattr(args, 'amp', False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for ep in range(1, args.epochs + 1):
        # Use the fixed batch every epoch to keep anchors stable
        x_b = x_b_static
        r_b = r_b_static
        tgt_c_b = tgt_c_b_static
        tgt_m_b = tgt_m_b_static
        tgt_p_b = tgt_p_b_static

        # Optionally override CPU-side sim augmentation with GPU augmentation to reduce CPU load
        if os.environ.get('REFINE_GPU_AUG', '0') == '1':
            # 统一调用训练增强函数（禁用 jitter；drop∈[0.2,0.4]；ghost∈[2,4]，可用 env 覆盖）
            jitter_sigma = float(os.environ.get('REFINE_AUG_JITTER', '0.0'))
            drop_lo = float(os.environ.get('REFINE_AUG_DROP_LO', '0.2'))
            drop_hi = float(os.environ.get('REFINE_AUG_DROP_HI', '0.4'))
            gh_lo = int(os.environ.get('REFINE_AUG_GHOST_LO', '2'))
            gh_hi = int(os.environ.get('REFINE_AUG_GHOST_HI', '4'))
            prop = tgt_c_b.clone()
            prop, _present_prop = augment_planA_from_gt_torch(
                prop, tgt_m_b,
                jitter_sigma=jitter_sigma,
                drop_lo=drop_lo, drop_hi=drop_hi,
                ghosts_lo=gh_lo, ghosts_hi=gh_hi,
            )
            x_b = prop

        # Optional: Anchor-matching validation — perturb input proposals x_b before SDEdit
        if bool(getattr(args, 'anchor_test_perturb', False)):
            try:
                Btmp = x_b.shape[0]
                if str(getattr(args, 'anchor_test_mode', 'shift')) == 'shift':
                    frac = float(getattr(args, 'anchor_shift_frac', 0.6))
                    for bb in range(Btmp):
                        # present if any point is valid (not masked)
                        present_slots = (~tgt_m_b[bb]).any(dim=1).nonzero(as_tuple=False).view(-1)
                        if present_slots.numel() >= 2:
                            idx_a = int(present_slots[0].item())
                            idx_b = int(present_slots[1].item())
                            # shift GT_A towards GT_B by frac, then assign to x_b[idx_a]
                            shift_vec = (tgt_c_b[bb, idx_b] - tgt_c_b[bb, idx_a]) * frac
                            x_b[bb, idx_a] = torch.clamp(tgt_c_b[bb, idx_a] + shift_vec, -1.0, 1.0)
                else:
                    sigma = float(getattr(args, 'anchor_noise_sigma', 0.5))
                    x_b = torch.clamp(x_b + torch.randn_like(x_b) * sigma, -1.0, 1.0)
            except Exception:
                pass

        # Build SDEdit start xK around proposal on GPU
        xK = torch.clamp((1.0 - float(args.alpha)) * x_b + float(args.alpha) * torch.randn_like(x_b), -1.0, 1.0)

        # Raster encoding + EDM unroll; apply DDP no_sync for inner unrolled steps implicitly via accumulation (not used here)
        with torch.cuda.amp.autocast(enabled=use_amp):
            rv = enc(r_b)
            sigmas = karras_schedule(max(1, int(args.steps)), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
            pred_coords, pred_logits, _preds, _states = edm_unrolled_train(net, xK, rv, sigmas, second_order=bool(args.second_order))

        # Build fixed proposal→GT mapping for step losses (MapTR-style with permutation-invariant regression)
        # This uses the start proposal x_b for each sample and keeps mapping constant across steps.
        try:
            print(f"[dbg] use_static_mapping={use_static_mapping}")
        except Exception:
            pass
        if not use_static_mapping:
            step_tgt_coords_list: List[torch.Tensor] = []
            step_tgt_mask_list: List[torch.Tensor] = []
            step_tgt_present_list: List[torch.Tensor] = []
            step_tgt_labels_list: List[torch.Tensor] = []  # for semantic supervision aligned to step mapping
            B = x_b.shape[0]
            # MapTR-style matcher weights (classification vs regression)
            cls_w = float(os.environ.get('REFINE_MATCH_CLS_W', '5.0'))
            reg_w = float(os.environ.get('REFINE_MATCH_REG_W', '50.0'))
            for b in range(B):
                pc0 = x_b[b]
                gb = tgt_c_b[b]
                gmb = tgt_m_b[b]
                gpb = tgt_p_b[b]
                # labels for GT (MapTR labeling order)
                import numpy as _np
                order = []
                for orig in (1, 0, 2):
                    cap = int(budgets.get(orig, 0))
                    lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                    order += [lab] * max(0, cap)
                gt_labels_b = torch.as_tensor(order, dtype=torch.long, device=device)[: gb.shape[0]]
                # Choose mapping mode
                use_perm = (os.environ.get('REFINE_PERM_MATCH', '1') == '1')
                if use_perm:
                    # per-sample permutation-invariant Hungarian (MapTR-style)
                    try:
                        # Prefer per-step semantics if available; fall back to slot labels
                        sem_logits_last = None
                        if isinstance(_preds, (list, tuple)) and len(_preds) > 0 and (len(_preds[-1]) >= 3) and (_preds[-1][2] is not None):
                            sem_logits_last = _preds[-1][2][b]  # [N,3]
                        from global_diffusion_map.refine.loss_refine import hungarian_match_perm as _HMP
                        pairs0, gt_perm_choice = _HMP(
                            pc0, sem_logits_last, gb, gmb, gt_labels_b,
                            cls_weight=cls_w, reg_weight=reg_w,
                        )
                    except Exception:
                        pairs0 = []
                        gt_perm_choice = _np.full((gb.shape[0],), -1, dtype=_np.int64)
                else:
                    # GPU greedy (previous) mapping — faster但非严格 Hungarian/Permutation-invariant
                    wc = float(os.environ.get('REFINE_MATCH_WC', '3.0'))
                    wd = float(os.environ.get('REFINE_MATCH_WDIR', '0.1'))
                    wp = float(os.environ.get('REFINE_MATCH_WPW', '0.3'))
                    from global_diffusion_map.refine.loss_refine import gpu_greedy_match as _GM
                    pairs0 = _GM(pc0, gb, gmb, gpb, w_center=wc, w_dir=wd, w_pw=wp, use_chamfer=(os.environ.get('REFINE_GPU_CHAMFER','1')=='1'))
                    gt_perm_choice = _np.full((gb.shape[0],), -1, dtype=_np.int64)  # no perm info for greedy
                # materialize fixed-mapped targets for this sample
                Nq, Pts = gb.shape[0], gb.shape[1]
                new_tgt0 = torch.zeros_like(gb)
                new_msk0 = torch.ones_like(gmb, dtype=torch.bool)
                new_pre0 = torch.zeros_like(gpb)
                new_lab0 = torch.full_like(gpb, fill_value=-1, dtype=torch.long)
                # valid→全局索引（当使用 GPU 贪心匹配时，列索引基于“有效GT子集”）
                valid_idx_np = _np.where(gpb.detach().cpu().numpy() > 0)[0]
                # order GT by budgets (MapTR) for label supervision consistency (not strictly needed here)
                for (pi, gj_loc) in pairs0:
                    # choose permutation for this matched GT if available
                    perm_k = int(gt_perm_choice[gj_loc]) if isinstance(gt_perm_choice, _np.ndarray) else -1
                    # map local valid index→global GT index when using greedy matcher (Hungarian-perm gives global)
                    gj_global = int(valid_idx_np[gj_loc]) if (use_perm == False and valid_idx_np.size > 0) else int(gj_loc)
                    # distance gating on normalized centers to prevent far-away forced matches
                    try:
                        c_pi = torch.nanmean(pc0[pi], dim=0)
                        c_gj = torch.nanmean(gb[gj_global], dim=0)
                        d = torch.linalg.norm(c_pi - c_gj).item()
                        if d > float(getattr(args, 'anchor_max_center_dist', 0.2)):
                            continue
                    except Exception:
                        pass
                    if perm_k == 1:
                        new_tgt0[pi] = torch.flip(gb[gj_global], dims=[0])
                        new_msk0[pi] = torch.flip(gmb[gj_global], dims=[0])
                    else:
                        new_tgt0[pi] = gb[gj_global]
                        new_msk0[pi] = gmb[gj_global]
                    new_pre0[pi] = 1
                    # semantic label taken from GT label order
                    if 0 <= gj_global < gt_labels_b.numel():
                        new_lab0[pi] = int(gt_labels_b[gj_global].item())
                step_tgt_coords_list.append(new_tgt0)
                step_tgt_mask_list.append(new_msk0)
                step_tgt_present_list.append(new_pre0)
                step_tgt_labels_list.append(new_lab0)
            step_tgt_coords = torch.stack(step_tgt_coords_list, dim=0).to(device)
            step_tgt_mask = torch.stack(step_tgt_mask_list, dim=0).to(device)
            step_tgt_present = torch.stack(step_tgt_present_list, dim=0).to(device)
            step_tgt_labels = torch.stack(step_tgt_labels_list, dim=0).to(device)
        else:
            step_tgt_coords = static_tgt_coords
            step_tgt_mask = static_tgt_mask
            step_tgt_present = static_tgt_present
            step_tgt_labels = static_tgt_labels

        # Per-step denoise loss (average over steps) against fixed-mapped targets
        step_losses = []
        for item in _preds:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                pc, pl = item[0], item[1]
                sem_step = item[2] if (len(item) >= 3) else None
            else:
                continue
            with torch.cuda.amp.autocast(enabled=use_amp):
                ls = criterion(pc, pl, step_tgt_coords, step_tgt_mask, step_tgt_present,
                           l1_weight=float(getattr(args, 'l1_weight', 1.0)), cls_weight=float(args.cls_weight),
                           use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                           pred_sem_logits=sem_step, tgt_sem_labels=step_tgt_labels, sem_weight=float(args.sem_weight),
                           smooth_weight=float(args.smooth_weight),
                           reg_len_exp=float(args.reg_len_exp), smooth_inv_len_exp=float(args.smooth_inv_len_exp),
                           dir_weight=float(args.dir_weight))
            # Include semantic classification loss so the semantic head actually learns
            step_losses.append(
                ls['loss_cls']
                + ls['loss_reg']
                + ls.get('loss_sem', pc.new_zeros([]))
                + ls.get('loss_smooth', pc.new_zeros([]))
            )

        # Debug: print ranges to detect coordinate issues / mirroring (rank 0 only)
        try:
            import torch.distributed as dist
            is_rank0 = (not use_ddp) or (dist.get_rank() == 0)
        except Exception:
            is_rank0 = True
        if is_rank0 and (ep in (1, 10)):
            pc_dbg = _preds[-1][0].detach() if _preds else pred_coords.detach()
            def _rng(t):
                return (float(t.min().item()), float(t.max().item()))
            xr0 = _rng(x_b[...,0]); yr0 = _rng(x_b[...,1])
            tr0 = _rng(tgt_c_b[...,0]); ur0 = _rng(tgt_c_b[...,1])
            pr0 = _rng(pc_dbg[...,0]); qr0 = _rng(pc_dbg[...,1])
            # Safe debug for weights (may be undefined if using static mapping path)
            _cls_w_dbg = os.environ.get('REFINE_MATCH_CLS_W', '5.0')
            _reg_w_dbg = os.environ.get('REFINE_MATCH_REG_W', '50.0')
            print(f"[dbg ep{ep}] x_b.x{xr0} x_b.y{yr0} tgt.x{tr0} tgt.y{ur0} pred.x{pr0} pred.y{qr0} reg_w={_reg_w_dbg} cls_w={_cls_w_dbg} permute=True")
        if step_losses:
            step_loss = torch.stack(step_losses).mean()
        else:
            with torch.cuda.amp.autocast(enabled=use_amp):
                ls = criterion(pred_coords, pred_logits, tgt_c_b, tgt_m_b, tgt_p_b,
                           l1_weight=1.0, cls_weight=float(args.cls_weight),
                           use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                           sem_weight=1.0, smooth_weight=float(args.smooth_weight),
                           reg_len_exp=float(args.reg_len_exp), smooth_inv_len_exp=float(args.smooth_inv_len_exp),
                           dir_weight=float(args.dir_weight))
            step_loss = (
                ls['loss_cls']
                + ls['loss_reg']
                + ls.get('loss_sem', pred_coords.new_zeros([]))
                + ls.get('loss_smooth', pred_coords.new_zeros([]))
            )

        # Final loss uses the same fixed-mapped targets as step losses (anchor matching)
        tgt_coords_m = step_tgt_coords
        tgt_mask_m = step_tgt_mask
        tgt_present_m = step_tgt_present
        tgt_labels_b = step_tgt_labels

        # Extract final-step semantic logits if present from unrolled steps
        pred_sem_logits = None
        if isinstance(_preds, (list, tuple)) and len(_preds) > 0:
            last = _preds[-1]
            if isinstance(last, (list, tuple)) and len(last) >= 3:
                pred_sem_logits = last[2]

        with torch.cuda.amp.autocast(enabled=use_amp):
            losses_final = criterion(pred_coords, pred_logits, tgt_coords_m, tgt_mask_m, tgt_present_m,
                                 l1_weight=float(getattr(args, 'l1_weight', 1.0)), cls_weight=float(args.cls_weight),
                                 use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                 pred_sem_logits=pred_sem_logits, tgt_sem_labels=tgt_labels_b, sem_weight=float(args.sem_weight),
                                 smooth_weight=float(args.smooth_weight),
                                 reg_len_exp=float(args.reg_len_exp), smooth_inv_len_exp=float(args.smooth_inv_len_exp),
                                 dir_weight=float(args.dir_weight))
        final_loss = (
            losses_final['loss_cls']
            + losses_final['loss_reg']
            + losses_final.get('loss_sem', pred_coords.new_zeros([]))
            + losses_final.get('loss_smooth', pred_coords.new_zeros([]))
        )

        # Total loss
        loss = float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss
        opt.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward(); opt.step()

        import torch.distributed as dist
        if (ep % 20 == 0 or ep == 1) and ((not use_ddp) or (dist.get_rank() == 0)):
            cur_lr = opt.param_groups[0]['lr']
            print(f"[ep {ep:04d}] steps={len(sigmas)} lr={cur_lr:.6g} loss={loss.item():.6f} step={step_loss.item():.6f} final={final_loss.item():.6f}")
        if (ep % 100 == 0 or ep == args.epochs) and ((not use_ddp) or (dist.get_rank() == 0)):
            with torch.no_grad():
                bviz = 0  # 可视化第一个样本
                pc = pred_coords[bviz].detach().cpu().numpy()   # [N,P,2]
                pl = pred_logits[bviz].detach().cpu().numpy().reshape(-1)  # [N]
                prob = 1.0 / (1.0 + np.exp(-pl))
                thr = float(getattr(args, 'viz_thr', 0.2))
                base_keep = np.where(prob >= thr)[0].tolist()
                # 诚实可视化：不依赖 GT 匹配标签过滤（暴露模型真实噪声）
                if not bool(getattr(args, 'viz_honest', False)):
                    # 兼容旧行为：可选地仅保留匹配到 GT 的槽位
                    try:
                        step_lbl = step_tgt_labels[bviz].detach().cpu().numpy()  # [N]
                        base_keep = [i for i in base_keep if (i < step_lbl.shape[0] and int(step_lbl[i]) != -1)]
                    except Exception:
                        pass
                gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
                bounds = gt.get('bounds')
                # 使用与 quick-infer 相同的固定 batch 栅格，确保与 epXXXX/infer 对齐
                ras = r_b_static[0].detach().cpu().numpy()
                # 语义着色：使用最后一步语义头预测为槽位着色
                sem_labels = None
                try:
                    if pred_sem_logits is not None:
                        sem_np = pred_sem_logits[bviz].detach().cpu().numpy()  # [N, C]
                        sem_labels = np.argmax(sem_np, axis=-1).astype(np.int64)
                except Exception:
                    sem_labels = None
                # 构造 mask，仅绘制 keep 槽位；其余槽位在可视化中屏蔽，降低中心噪点
                Nviz = pc.shape[0]
                mask = np.ones((Nviz, pc.shape[1]), dtype=bool)
                for i in base_keep:
                    if 0 <= i < Nviz:
                        mask[i, :] = False
                # 使用带类别着色与编号的可视化，且不再叠加 Canny 边缘，减少“中心噪声”观感
                from global_diffusion_map.refine.single_scene_overfit import overlay_slots_annot
                overlay_slots_annot(
                    osp.join(viz_dir, f'ep_{ep:04d}.png'),
                    ras, bounds,
                    slots=pc, mask=mask, labels=sem_labels,
                    title=f'ep_{ep:04d} (thr=0.2)')
                # Save checkpoint
                enc_sd = enc.module.state_dict() if hasattr(enc, 'module') else enc.state_dict()
                net_sd = base.module.state_dict() if hasattr(base, 'module') else base.state_dict()
                ckpt = {
                    'encoder': enc_sd,
                    'net': net_sd,
                    'P': P, 'N': N,
                    'budgets': budgets,
                }
                os.makedirs(out_dir, exist_ok=True)
                torch.save(ckpt, osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth'))
        # Step LR scheduler
        if sched is not None:
            if args.sched == 'plateau':
                sched.step(loss.detach())
            else:
                sched.step()

    print(f"[ok] EDM one-scene training done. Visualizations under {viz_dir}.")

    # Quick inference sanity: run a fresh multistep denoise from xK (same scene)
    enc.eval(); base.eval()
    with torch.no_grad():
        # 使用固定 batch 的栅格与 proposal，确保与 epXXXX 可视化对齐
        r = r_b_static
        x = x_b_static
        if float(args.alpha) > 0.0:
            xK = torch.clamp((1.0 - float(args.alpha)) * x + float(args.alpha) * torch.randn_like(x), -1.0, 1.0)
        else:
            xK = x.clone()
        rv = enc(r)
        sigmas = karras_schedule(max(1, int(args.steps)), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
        pred_coords, pred_logits, _preds_inf, _ = edm_unrolled_train(net, xK, rv, sigmas, second_order=bool(args.second_order))
        pc = pred_coords[0].detach().cpu().numpy()
        pl = pred_logits[0].detach().cpu().numpy().reshape(-1)
        prob = 1.0 / (1.0 + np.exp(-pl))
        thr = float(getattr(args, 'viz_thr', 0.2))
        keep = np.where(prob >= thr)[0].tolist()
        # 仅保留与 GT 匹配的槽（根据固定映射标签）
        try:
            present_slots = (~tgt_m_b_static[0]).any(dim=1).cpu().numpy()
            keep = [i for i in keep if (i < present_slots.shape[0] and bool(present_slots[i]))]
        except Exception:
            pass
        gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
        bounds = gt.get('bounds')
        # 使用固定 batch 的 raster 可视化，保持与 epXXXX 对齐
        ras = r_b_static[0].detach().cpu().numpy()
        # 语义类别用于着色
        sem_labels = None
        try:
            if isinstance(_preds_inf, (list, tuple)) and len(_preds_inf) > 0 and isinstance(_preds_inf[-1], (list, tuple)) and len(_preds_inf[-1]) >= 3:
                sem_logits_inf = _preds_inf[-1][2]
                if sem_logits_inf is not None:
                    sem_labels = np.argmax(sem_logits_inf[0].detach().cpu().numpy(), axis=-1).astype(np.int64)
        except Exception:
            sem_labels = None
        # mask 仅保留 keep 槽位
        Nviz = pc.shape[0]
        mask = np.ones((Nviz, pc.shape[1]), dtype=bool)
        for i in keep:
            if 0 <= i < Nviz:
                mask[i, :] = False
        if (not use_ddp) or (dist.get_rank() == 0):
            from global_diffusion_map.refine.single_scene_overfit import overlay_slots_annot
            overlay_slots_annot(
                osp.join(out_dir, 'infer_refine_overlay.png'),
                ras, bounds,
                slots=pc, mask=mask, labels=sem_labels,
                title='Refined (thr=0.2) + sem cls')
            print(f"[ok] inference overlay: {osp.join(out_dir, 'infer_refine_overlay.png')}")


if __name__ == '__main__':
    main()
