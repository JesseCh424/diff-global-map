#!/usr/bin/env python
from __future__ import annotations

"""
One-scene refine training (EDM) — Stable Memory + Streaming Step Loss.

Purpose
- 保留 GPU5 的“快下降”路径（重匹配 + step+final 双监督），同时用“流式 step loss”彻底避免
  存储每步的 preds 列表，进一步抑制显存缓慢增长（缓存逐步上限化）。

策略
- Split-Backward：编码器与去噪器解耦；步内图立即释放。
- Static Buffers：预分配噪声/xK/中间张量；避免分配器抖动与碎片化。
- Streaming EDM：在 unrolled 过程中就地累加 step loss，不收集 preds 列表。
- Allocator Hygiene：PYTORCH_CUDA_ALLOC_CONF + 禁用 cudnn.benchmark。
"""

import os
# Use legacy cudaMalloc backend with aggressive GC to minimize driver-used growth
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMalloc,max_split_size_mb:8,garbage_collection_threshold:0.0"

import argparse
import os.path as osp
import gc
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Disable cuDNN completely in this trainer to avoid internal workspaces
torch.backends.cudnn.enabled = False

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.loss_refine import criterion, hungarian_match_perm, gpu_greedy_match
from global_diffusion_map.refine.single_scene_overfit import overlay_slots_annot, load_pickle
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class StaticDataBuffer:
    def __init__(self, B: int, N: int, P: int, device: torch.device | str) -> None:
        self.B, self.N, self.P = B, N, P
        self.x_buf = torch.zeros(B, N, P, 2, device=device)
        self.mask_buf = torch.zeros(B, N, P, dtype=torch.bool, device=device)
        self.shifts = torch.zeros(B, N, 1, 2, device=device)
        self.local = torch.zeros(B, N, P, 2, device=device)
        self.rnd_drop = torch.zeros(B, N, device=device)
        self.ghost_base = torch.zeros(B, N, P, 2, device=device)
        self.diff_noise = torch.zeros(B, N, P, 2, device=device)
        self.xK = torch.zeros(B, N, P, 2, device=device)
        # preallocate encoder expansion buffer (contiguous)
        self.rv_expand: Optional[torch.Tensor] = None

    def jitter_drop_ghost(self, gt_coords: torch.Tensor, gt_mask: torch.Tensor,
                           shift_sigma: float, point_sigma: float, drop_frac: float, ghosts: int
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.x_buf.copy_(gt_coords.expand(self.B, self.N, self.P, 2))
        self.mask_buf.copy_(gt_mask.expand(self.B, self.N, self.P))
        valid = ~self.mask_buf
        self.shifts.normal_().mul_(shift_sigma)
        self.local.normal_().mul_(point_sigma)
        noise_view = self.local + self.shifts
        x_valid = self.x_buf[valid]
        x_valid.add_(noise_view[valid]).clamp_(-1.0, 1.0)
        self.x_buf[valid] = x_valid
        present = valid.any(dim=2)
        self.rnd_drop.uniform_()
        is_drop = present & (self.rnd_drop < drop_frac)
        keep_identity = present & (~is_drop)
        if is_drop.any():
            self.local.normal_().mul_(max(shift_sigma, point_sigma) * 0.8)
            base = gt_coords.expand(self.B, self.N, self.P, 2)
            self.x_buf[is_drop] = (base[is_drop] + self.local[is_drop]).clamp_(-1.0, 1.0)
        k = int(min(max(0, ghosts), self.N))
        if k > 0:
            empty = ~present
            self.rnd_drop.uniform_(); self.rnd_drop[present] = -1.0
            _, idx = torch.topk(self.rnd_drop, k=k, dim=1)
            is_ghost = torch.zeros_like(present)
            is_ghost.scatter_(1, idx, True); is_ghost &= empty
            self.ghost_base.uniform_(-1.0, 1.0).mul_(0.6)
            for t in range(1, self.P):
                self.ghost_base[:, :, t].mul_(0.3).add_(self.ghost_base[:, :, t-1], alpha=0.7)
            self.x_buf[is_ghost] = self.ghost_base[is_ghost].clamp_(-1.0, 1.0)
        return self.x_buf, keep_identity


def build_fixed_targets(x_in: torch.Tensor, gt_coords: torch.Tensor, gt_mask: torch.Tensor,
                        gt_present: torch.Tensor, keep_identity_mask: torch.Tensor,
                        use_greedy: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x_in.device
    N, P, _ = x_in.shape
    tgt_coords = torch.zeros_like(gt_coords)
    tgt_mask_o = torch.ones_like(gt_mask, dtype=torch.bool)
    tgt_present_o = torch.zeros_like(gt_present)
    match_idx = torch.full_like(gt_present, -1)

    keep_ids = torch.nonzero(keep_identity_mask & (gt_present > 0), as_tuple=False).view(-1)
    if keep_ids.numel() > 0:
        x_sel = x_in.index_select(0, keep_ids)
        g_sel = gt_coords.index_select(0, keep_ids)
        m_sel = gt_mask.index_select(0, keep_ids)
        v = (~m_sel).float(); denom = v.sum(dim=1).clamp_min(1.0)
        l1_f = ((x_sel - g_sel).abs().sum(dim=-1) * v).sum(dim=1) / denom
        g_rev = g_sel.flip(1); m_rev = m_sel.flip(1)
        l1_r = ((x_sel - g_rev).abs().sum(dim=-1) * (~m_rev).float()).sum(dim=1) / denom
        use_rev = l1_r < l1_f
        g_final = torch.where(use_rev.view(-1, 1, 1), g_rev, g_sel)
        m_final = torch.where(use_rev.view(-1, 1), m_rev, m_sel)
        tgt_coords.index_copy_(0, keep_ids, g_final)
        tgt_mask_o.index_copy_(0, keep_ids, m_final)
        tgt_present_o.index_copy_(0, keep_ids, torch.ones_like(keep_ids, dtype=tgt_present_o.dtype))
        match_idx.index_copy_(0, keep_ids, keep_ids)

    remaining_gt = (gt_present > 0).clone(); remaining_gt[keep_ids] = False
    if bool(remaining_gt.any()):
        gt_sel_idx = torch.nonzero(remaining_gt, as_tuple=False).view(-1)
        if gt_sel_idx.numel() > 0:
            gt_sel_coords = gt_coords.index_select(0, gt_sel_idx)
            gt_sel_mask = gt_mask.index_select(0, gt_sel_idx)
            assigned_slots = set(keep_ids.tolist())
            all_idx = torch.arange(x_in.size(0), device=x_in.device)
            free_mask = torch.ones_like(all_idx, dtype=torch.bool)
            if keep_ids.numel() > 0:
                free_mask[keep_ids] = False
            pred_free_idx = all_idx[free_mask]
            if use_greedy:
                x_free = x_in.index_select(0, pred_free_idx)
                present_rem = torch.ones((gt_sel_idx.numel(),), dtype=torch.bool, device=x_in.device)
                pairs = gpu_greedy_match(x_free, gt_sel_coords, gt_sel_mask, present_rem,
                                         w_center=1.0, w_dir=0.2, w_pw=0.5, use_chamfer=True,
                                         max_center_dist=None)
                for (pi_free, gj_loc) in pairs:
                    pi_i = int(pred_free_idx[int(pi_free)].item())
                    if pi_i in assigned_slots:
                        continue
                    gj_global = int(gt_sel_idx[int(gj_loc)].item())
                    if not bool(remaining_gt[gj_global]):
                        continue
                    g_sel = gt_coords[gj_global]
                    m_sel = gt_mask[gj_global]
                    v = (~m_sel).float(); denom = v.sum(dim=0).clamp_min(1.0)
                    l1_f = ((x_in[pi_i] - g_sel).abs().sum(dim=-1) * v).sum() / denom
                    g_rev = g_sel.flip(0); m_rev = m_sel.flip(0)
                    v_rev = (~m_rev).float(); l1_r = ((x_in[pi_i] - g_rev).abs().sum(dim=-1) * v_rev).sum() / denom
                    if bool(l1_r < l1_f):
                        tgt_coords[pi_i] = g_rev; tgt_mask_o[pi_i] = m_rev
                    else:
                        tgt_coords[pi_i] = g_sel; tgt_mask_o[pi_i] = m_sel
                    tgt_present_o[pi_i] = 1
                    remaining_gt[gj_global] = False
                    assigned_slots.add(pi_i)
                    match_idx[pi_i] = int(gj_global)
            else:
                pairs, perm_choice = hungarian_match_perm(
                    x_in, None, gt_sel_coords, gt_sel_mask, None, cls_weight=0.0, reg_weight=50.0, use_l1_beta=0.0
                )
                for (pi, gj_loc) in pairs:
                    pi_i = int(pi)
                    if pi_i in assigned_slots:
                        continue
                    gj_global = int(gt_sel_idx[int(gj_loc)].item())
                    if not bool(remaining_gt[gj_global]):
                        continue
                    k = int(perm_choice[int(gj_loc)]) if (perm_choice is not None and len(perm_choice) > int(gj_loc)) else 0
                    if k == 1:
                        tgt_coords[pi_i] = gt_coords[gj_global].flip(0)
                        tgt_mask_o[pi_i] = gt_mask[gj_global].flip(0)
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
    if m > 0: out[:m] = np.asarray(order[:m], dtype=np.int64)
    return out


class PolyDiffuseImageEncoder256(nn.Module):
    def __init__(self, cfg_path: str, pretrained_ckpt: str, device: str = 'cuda') -> None:
        super().__init__()
        from mmcv import Config
        from mmdet.models import build_backbone, build_neck
        cfg = Config.fromfile(cfg_path)
        self.backbone = build_backbone(cfg.model.get('img_backbone'))
        self.neck = build_neck(cfg.model.get('img_neck'))
        state = torch.load(pretrained_ckpt, map_location='cpu')
        sd = state.get('state_dict', state)
        self.backbone.load_state_dict({k.replace('img_backbone.', ''): v for k, v in sd.items() if 'img_backbone.' in k}, strict=False)
        self.neck.load_state_dict({k.replace('img_neck.', ''): v for k, v in sd.items() if 'img_neck.' in k}, strict=False)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            feats = self.neck(self.backbone(dummy))
            dim = feats[0].shape[1]
        self.proj = nn.Identity() if dim == 256 else nn.Linear(dim, 256)
        self.to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.neck(self.backbone(x))
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        return self.proj(torch.stack(pooled, dim=0).mean(0))


def edm_unrolled_stream(
    net: EDMPrecondRefine,
    x_start: torch.Tensor,            # [B,N,P,2]
    raster_vec: torch.Tensor,         # [B,C]
    sigmas: torch.Tensor,             # [S]
    second_order: bool,
    cond_prior: torch.Tensor | None,
    input_labels: torch.Tensor | None,
    # streaming: accumulate step loss externally
    step_loss_acc: Optional[List[torch.Tensor]],  # pass [] to accumulate, or None to skip
    # closure to compute per-step loss (pc, pl) -> scalar
    step_loss_fn,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    B = x_start.shape[0]
    x_t = x_start
    pred_coords: torch.Tensor | None = None
    pred_logits: torch.Tensor | None = None
    pred_sem: torch.Tensor | None = None

    for si, sigma in enumerate(sigmas):
        sigma_b = torch.full((B,), float(sigma), device=x_start.device, dtype=torch.float32)
        out = net(x_t, sigma_b, raster_vec, x_prior=cond_prior, input_labels=input_labels)
        if isinstance(out, (tuple, list)):
            pred_coords, pred_logits = out[0], out[1]
            pred_sem = out[2] if len(out) > 2 else None
        else:
            pred_coords, pred_logits = out, None
            pred_sem = None
        if step_loss_acc is not None:
            step_loss_acc.append(step_loss_fn(pred_coords, pred_logits, pred_sem))

        if si < len(sigmas) - 1:
            sigma_next = float(sigmas[si + 1])
            t_hat = float(sigma)
            d_cur = (x_t - pred_coords).detach() / max(t_hat, 1e-6)
            x_next = x_t + (sigma_next - t_hat) * d_cur
            if second_order:
                sigma_b2 = torch.full((B,), sigma_next, device=x_start.device, dtype=torch.float32)
                with torch.no_grad():
                    out2 = net(x_next, sigma_b2, raster_vec, x_prior=cond_prior, input_labels=input_labels)
                    pc2 = out2[0] if isinstance(out2, (tuple, list)) else out2
                d_prime = (x_next - pc2).detach() / max(sigma_next, 1e-6)
                x_next = x_t + (sigma_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            x_t = x_next.clamp(-1.0, 1.0)
    return pred_coords, pred_logits, pred_sem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_one_scene_polydiff_stream')
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--accum-steps', type=int, default=1)
    ap.add_argument('--freeze-encoder', action='store_true', default=True)
    ap.add_argument('--polydiff-cfg', default='official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py')
    ap.add_argument('--pretrained-maptr-ckpt', default='global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth')
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.4)
    ap.add_argument('--alpha', type=float, default=0.03)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--matcher', choices=['greedy', 'hungarian'], default='greedy')
    ap.add_argument('--shift-sigma', type=float, default=0.10)
    ap.add_argument('--point-sigma', type=float, default=0.02)
    ap.add_argument('--drop-frac', type=float, default=0.15)
    ap.add_argument('--ghosts', type=int, default=2)
    ap.add_argument('--l1-weight', type=float, default=20.0)
    ap.add_argument('--cls-weight', type=float, default=5.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    ap.add_argument('--step-loss-weight', type=float, default=1.0)
    ap.add_argument('--final-loss-weight', type=float, default=1.0)
    ap.add_argument('--use-focal', action='store_true')
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--prior-lr-mult', type=float, default=3.0)
    ap.add_argument('--sched', choices=['cosine', 'none'], default='cosine')
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--resume-ckpt', default=None)
    ap.add_argument('--resume-epoch', type=int, default=-1)
    ap.add_argument('--log-every', type=int, default=1)
    ap.add_argument('--viz-thr', type=float, default=0.5)
    # Checkpoint saving
    ap.add_argument('--save-every', type=int, default=50, help='Save checkpoint every N epochs (0=disable)')
    args = ap.parse_args()

    set_seed(0)
    device = torch.device('cuda')
    # additionally cap allocator cache per-process to curb reserved growth
    try:
        torch.cuda.set_per_process_memory_fraction(0.2)
    except Exception:
        pass

    import json
    with open(args.stats_json, 'r') as f: stats = json.load(f)
    P = int(stats.get('M', 20)); N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 12, 1: 30, 2: 22}).items()}
    RefineCaps(num_points=P, num_queries=N)

    gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
    bounds = gt['bounds']
    gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, P, N)
    gt_coords_gpu = torch.from_numpy(gt_pack).float().to(device)
    gt_mask_gpu = torch.from_numpy(gt_mask).bool().to(device)
    gt_present_gpu = torch.from_numpy((~gt_mask).any(axis=1).astype(np.int64)).long().to(device)

    from PIL import Image
    import torchvision.transforms.functional as TF
    raw = Image.open(osp.join(args.rendered_root, args.scene, '10_render_gt.png')).convert('RGB')
    w, h = raw.size
    nw, nh = max(32, (w//32)*32), max(32, (h//32)*32)
    if nw!=w or nh!=h: raw = raw.resize((nw, nh), Image.BILINEAR)
    img_t = TF.normalize(TF.to_tensor(raw), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ras_t = img_t.unsqueeze(0).to(device)

    enc = PolyDiffuseImageEncoder256(args.polydiff_cfg, args.pretrained_maptr_ckpt, device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    buffer = StaticDataBuffer(args.batch_size, N, P, device)

    params = []
    if args.freeze_encoder:
        enc.eval(); [p.requires_grad_(False) for p in enc.parameters()]
    else:
        params += [{'params': enc.parameters(), 'lr': args.lr * 0.1}]
    prior_params = list(net.backbone.prior_mlp.parameters()) + [net.backbone.prior_gate]
    if hasattr(net.backbone, 'class_emb'):
        prior_params += list(net.backbone.class_emb.parameters()) + [net.backbone.class_gate]
    prior_ids = {id(p) for p in prior_params}
    main_params = [p for p in net.parameters() if id(p) not in prior_ids]
    params += [{'params': main_params, 'lr': args.lr}, {'params': prior_params, 'lr': args.lr * args.prior_lr_mult}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr_min) if args.sched == 'cosine' else None
    scaler = torch.cuda.amp.GradScaler()

    start_ep = 1
    if args.resume_ckpt:
        d = torch.load(args.resume_ckpt, map_location=device)
        if 'net' in d: net.backbone.load_state_dict(d['net'], strict=False)
        if 'encoder' in d: enc.load_state_dict(d['encoder'], strict=False)
        start_ep = args.resume_epoch + 1
        print(f"Resumed from {args.resume_ckpt}")

    labels_base = torch.from_numpy(_labels_from_budgets(budgets, N)).long().to(device)
    out_dir = osp.join(args.out_root, args.scene)
    viz_dir = osp.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    print(f"[start] streaming step-loss, alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    for ep in range(start_ep, args.epochs + 1):
        net.train(); enc.train() if not args.freeze_encoder else enc.eval()
        opt.zero_grad(set_to_none=True)
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        enc_ctx = torch.no_grad() if args.freeze_encoder else torch.enable_grad()
        with enc_ctx:
            rv_single = enc(ras_t)
        if not args.freeze_encoder:
            rv_use = rv_single.detach(); rv_use.requires_grad_(True)
        else:
            rv_use = rv_single
        # expand into preallocated contiguous buffer to avoid implicit copies
        if buffer.rv_expand is None or buffer.rv_expand.shape != (args.batch_size, rv_use.shape[-1]):
            buffer.rv_expand = torch.empty(args.batch_size, rv_use.shape[-1], device=device)
        buffer.rv_expand.copy_(rv_use.expand(args.batch_size, -1))

        # Precompute sigmas once per epoch on device
        sigmas = karras_schedule(args.steps, args.sigma_min, args.sigma_max, args.rho).to(device)

        l_sum = s_sum = f_sum = 0.0
        args.need_viz = (ep % 50 == 0 or ep == args.epochs)

        # One-time warmup on first epoch to materialize all workspaces/caches upfront
        if ep == start_ep:
            with torch.no_grad():
                for _ in range(16):
                    x_w, keep_identity_w = buffer.jitter_drop_ghost(
                        gt_coords_gpu, gt_mask_gpu, args.shift_sigma, args.point_sigma, args.drop_frac, args.ghosts
                    )
                    if buffer.rv_expand is None:
                        buffer.rv_expand = torch.empty(args.batch_size, rv_use.shape[-1], device=device)
                        buffer.rv_expand.copy_(rv_use.expand(args.batch_size, -1))
                    buffer.diff_noise.normal_(); buffer.xK.copy_(x_w).mul_(1.0 - args.alpha).add_(buffer.diff_noise, alpha=args.alpha).clamp_(-1,1)
                    pc_w, pl_w, _ = edm_unrolled_stream(
                        net, buffer.xK, buffer.rv_expand, sigmas, args.second_order,
                        cond_prior=x_w, input_labels=None, step_loss_acc=None, step_loss_fn=None
                    )
                    # touch criterion once to allocate any internal buffers
                    _ = criterion(pc_w, pl_w, torch.zeros_like(buffer.xK), buffer.mask_buf[:args.batch_size],
                                  torch.zeros(args.batch_size, N, dtype=torch.long, device=device))
                try:
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
                except Exception:
                    pass

        for micro in range(args.accum_steps):
            if micro < args.accum_steps - 1: args.need_viz = False
            elif (ep % 50 == 0 or ep == args.epochs): args.need_viz = True

            # Data gen
            x, keep_identity = buffer.jitter_drop_ghost(
                gt_coords_gpu, gt_mask_gpu, args.shift_sigma, args.point_sigma, args.drop_frac, args.ghosts
            )
            # Targets per batch item (preallocated buffers)
            if 'tgt_coords_buf' not in locals():
                tgt_coords_buf = torch.empty(args.batch_size, N, P, 2, device=device)
                tgt_mask_o = torch.empty(args.batch_size, N, P, dtype=torch.bool, device=device)
                tgt_present_o = torch.empty(args.batch_size, N, dtype=torch.long, device=device)
                tgt_labels = torch.empty(args.batch_size, N, dtype=torch.long, device=device)
                input_prior = torch.empty(args.batch_size, N, dtype=torch.long, device=device)
            for b in range(args.batch_size):
                keep_b = keep_identity[b]
                # Build targets fully on CPU to avoid transient GPU allocations during matching
                with torch.no_grad():
                    xb_cpu = x[b].detach().cpu()
                    gt_coords_cpu = gt_coords_gpu.detach().cpu()
                    gt_mask_cpu = gt_mask_gpu.detach().cpu()
                    gt_present_cpu = gt_present_gpu.detach().cpu()
                    tgt_c_cpu, tgt_m_cpu, tgt_p_cpu, match_idx = build_fixed_targets(
                        xb_cpu, gt_coords_cpu, gt_mask_cpu, gt_present_cpu, keep_b.cpu(), use_greedy=(args.matcher=='greedy')
                    )
                    # move results back to GPU buffers
                    tgt_c = tgt_c_cpu.to(device, non_blocking=True)
                    tgt_m = tgt_m_cpu.to(device, non_blocking=True)
                    tgt_p = tgt_p_cpu.to(device, non_blocking=True)
                labels_slot = torch.full((N,), -1, dtype=torch.long, device=device)
                valid = (match_idx >= 0) & (match_idx < N)
                if valid.any():
                    labels_slot[valid] = labels_base[match_idx[valid].to(device)]
                tmp_prior = labels_slot.clone(); tmp_prior[~keep_b] = -1
                tgt_coords_buf[b].copy_(tgt_c)
                tgt_mask_o[b].copy_(tgt_m)
                tgt_present_o[b].copy_(tgt_p)
                tgt_labels[b].copy_(labels_slot)
                input_prior[b].copy_(tmp_prior)

            # xK via buffer
            buffer.diff_noise.normal_()
            buffer.xK.copy_(x).mul_(1.0 - args.alpha).add_(buffer.diff_noise, alpha=args.alpha).clamp_(-1, 1)

            def step_loss_fn(pc, pl, sem_logits):
                out = criterion(pc, pl, tgt_coords_buf, tgt_mask_o, tgt_present_o,
                                l1_weight=args.l1_weight, cls_weight=args.cls_weight,
                                use_focal=args.use_focal, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                                pred_sem_logits=sem_logits, tgt_sem_labels=tgt_labels, sem_weight=args.sem_weight)
                return out['loss_cls'] + out['loss_reg'] + out.get('loss_sem', 0.0)

            # Streaming unrolled EDM (no preds list)
            step_losses: List[torch.Tensor] = [] if args.step_loss_weight > 0 else None
            pc, pl, psem = edm_unrolled_stream(
                net, buffer.xK, buffer.rv_expand, sigmas, args.second_order,
                cond_prior=x, input_labels=input_prior,
                step_loss_acc=step_losses, step_loss_fn=step_loss_fn
            )
            if args.step_loss_weight > 0 and step_losses is not None and len(step_losses) > 0:
                step_loss = torch.stack(step_losses).mean()
            else:
                step_loss = torch.tensor(0.0, device=device)
            out_final = criterion(pc, pl, tgt_coords_buf, tgt_mask_o, tgt_present_o,
                                  l1_weight=args.l1_weight, cls_weight=args.cls_weight,
                                  use_focal=args.use_focal, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                                  pred_sem_logits=psem, tgt_sem_labels=tgt_labels, sem_weight=args.sem_weight)
            final_loss = out_final['loss_cls'] + out_final['loss_reg'] + out_final.get('loss_sem', 0.0)
            total_loss = (args.step_loss_weight * step_loss + args.final_loss_weight * final_loss) / float(args.accum_steps)

            scaler.scale(total_loss).backward()
            l_sum += float(total_loss.item()) * float(args.accum_steps)
            s_sum += float(step_loss.item()) if isinstance(step_loss, torch.Tensor) else 0.0
            f_sum += float(final_loss.item())

            # release references promptly (buffers persist)
            del pc, pl, psem, total_loss, final_loss, step_loss
            if step_losses is not None:
                step_losses.clear()
            # proactively trim minor caches to keep reserved flat
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()

        if not args.freeze_encoder and rv_use.grad is not None:
            rv_single.backward(rv_use.grad)
        opt.step()
        if sched: sched.step()
        # keep buffer.rv_expand for reuse
        if not args.freeze_encoder: del rv_use

        if ep % args.log_every == 0 or ep == 1:
            print(f"[ep {ep:04d}] steps={args.steps} loss={l_sum/args.accum_steps:.4f} step={s_sum/args.accum_steps:.4f} final={f_sum/args.accum_steps:.4f}")

        # Save checkpoint periodically and at the end
        try:
            if (int(getattr(args, 'save_every', 0)) > 0 and (ep % int(args.save_every) == 0)) or (ep == int(args.epochs)):
                ckpt_path = osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth')
                os.makedirs(out_dir, exist_ok=True)
                torch.save({
                    'encoder': enc.state_dict(),
                    'net': base.state_dict(),
                    'P': P,
                    'N': N,
                    'budgets': budgets,
                }, ckpt_path)
                print(f"[ckpt] saved: {ckpt_path}")
        except Exception as e:
            print(f"[warn] checkpoint save failed: {e}")

        # trim cache every epoch to keep reserved flat
        gc.collect();
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()

    print(f"[ok] done: {out_dir}")


if __name__ == '__main__':
    main()
