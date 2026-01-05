#!/usr/bin/env python
from __future__ import annotations

"""
One-scene refine training (EDM) using the official PolyDiffuse image encoder (MapTR ResNet‑50 + FPN).

Highlights
- Reuses the existing refine pipeline (pack GT to slots, jitter/drop/ghost, EDM schedule, SlotMLP denoiser).
- Replaces the simple RasterEncoder with a PolyDiffuse encoder built from the MapTR config:
  * Builds `img_backbone` + `img_neck` via mmdet3d using
    `official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py`.
  * Loads `img_backbone` and `img_neck` weights from the MapTR checkpoint.
  * Exposes a 256‑dim global raster embedding by pooling FPN features.
- BN hygiene: keep BN layers in eval mode to stabilize with small batch.
- Optimizer groups: 0.1× LR for loaded backbone/neck, base LR for new heads, and higher LR for prior gates.

Notes
- Bounds, canvases, and data sources follow MapTracker policy (static GT pkls provide canonical bounds).
- This script focuses on training; for inference with the same encoder, create a matching infer script or integrate an encoder flag.

Anti-fragmentation (allocator) hygiene
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8` before importing torch
  (can be overridden by environment).
- Pre-allocate diffusion noise buffer and reuse every micro-step to reduce allocator churn.
"""

import os as _os
# Set allocator config before importing torch (can be overridden by user env)
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")

import argparse
import os
import os.path as osp
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Official‑style numerics: disable TF32 to match PolyDiffuse defaults
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Prefer stable cudnn plan cache to avoid growth across shape variants
torch.backends.cudnn.benchmark = False

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Refine pipeline building blocks
from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.loss_refine import criterion, hungarian_match_perm, gpu_greedy_match
from global_diffusion_map.refine.single_scene_overfit import (
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
    for k in range(1, P):
        pts[k] = 0.7 * pts[k] + 0.3 * pts[k - 1]
    return np.clip(pts, -1.0, 1.0)


def jitter_drop_ghost(
    gt_pack: np.ndarray,   # [N,P,2] normalized
    gt_mask: np.ndarray,   # [N,P] True=pad
    shift_sigma: float = 0.10,
    point_sigma: float = 0.02,
    drop_frac: float = 0.15,
    ghosts: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, P, _ = gt_pack.shape
    x = gt_pack.copy()
    valid = ~gt_mask
    shifts = np.random.normal(scale=shift_sigma, size=(N, 1, 2)).astype(np.float32)
    local = np.random.normal(scale=point_sigma, size=x.shape).astype(np.float32)
    noise_total = shifts + local
    x[valid] = np.clip(x[valid] + noise_total[valid], -1.0, 1.0)
    present_gt = (~gt_mask).any(axis=1)
    ids = np.where(present_gt)[0].tolist(); np.random.shuffle(ids)
    k_drop = max(0, int(round(len(ids) * float(drop_frac))))
    drop_ids = set(ids[:k_drop])
    keep_identity = present_gt.copy()
    is_drop = np.zeros((N,), dtype=bool)
    for i in drop_ids:
        keep_identity[i] = False
        is_drop[i] = True
        rnd = np.random.normal(scale=max(shift_sigma, point_sigma) * 0.8, size=(P, 2)).astype(np.float32)
        x[i] = np.clip(gt_pack[i] + rnd, -1.0, 1.0)
    empty_ids = np.where(~present_gt)[0].tolist(); np.random.shuffle(empty_ids)
    is_ghost = np.zeros((N,), dtype=bool)
    for j in empty_ids[:max(0, int(ghosts))]:
        x[j] = _rand_ghost(P)
        is_ghost[j] = True
    return x, keep_identity, is_drop, is_ghost

def jitter_drop_ghost_batch(
    gt_coords: torch.Tensor,   # [1,N,P,2] on device
    gt_mask: torch.Tensor,     # [1,N,P] on device
    batch_size: int,
    shift_sigma: float = 0.10,
    point_sigma: float = 0.02,
    drop_frac: float = 0.15,
    ghosts: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GPU generation for the entire batch.
    Returns (x [B,N,P,2], keep_identity [B,N])."""
    device = gt_coords.device
    _, N, P, _ = gt_coords.shape
    x = gt_coords.repeat(batch_size, 1, 1, 1).clone()
    mask = gt_mask.repeat(batch_size, 1, 1)
    valid = ~mask
    # Jitter
    shifts = torch.randn(batch_size, N, 1, 2, device=device) * float(shift_sigma)
    local = torch.randn(batch_size, N, P, 2, device=device) * float(point_sigma)
    noise = shifts + local  # small temporary; kept to retain original behavior
    x[valid] = torch.clamp(x[valid] + noise[valid], -1.0, 1.0)
    # Presence and drops
    present = valid.any(dim=2)  # [B,N]
    rnd = torch.rand(batch_size, N, device=device)
    is_drop = present & (rnd < float(drop_frac))
    keep_identity = present & (~is_drop)
    if bool(is_drop.any()):
        drop_noise = torch.randn_like(x) * (max(float(shift_sigma), float(point_sigma)) * 0.8)
        base = gt_coords.repeat(batch_size, 1, 1, 1)
        x[is_drop] = torch.clamp(base[is_drop] + drop_noise[is_drop], -1.0, 1.0)
    # Ghosts into empty slots
    k = int(min(max(0, int(ghosts)), N))
    if k > 0:
        empty = ~present
        scores = torch.rand(batch_size, N, device=device)
        scores[present] = -1.0
        _, idx = torch.topk(scores, k=k, dim=1)
        is_ghost = torch.zeros_like(present)
        is_ghost.scatter_(1, idx, True)
        is_ghost = is_ghost & empty
        ghost = (torch.rand(batch_size, N, P, 2, device=device) * 2.0 - 1.0) * 0.6
        for t in range(1, P):
            ghost[:, :, t] = 0.7 * ghost[:, :, t] + 0.3 * ghost[:, :, t - 1]
        x[is_ghost] = torch.clamp(ghost[is_ghost], -1.0, 1.0)
    return x, keep_identity


def build_fixed_targets(
    x_in: torch.Tensor,         # [N,P,2]
    gt_coords: torch.Tensor,    # [N,P,2]
    gt_mask: torch.Tensor,      # [N,P]
    gt_present: torch.Tensor,   # [N]
    keep_identity_mask: torch.Tensor,  # [N]
    use_greedy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x_in.device
    N, P, _ = x_in.shape
    tgt_coords = torch.zeros_like(gt_coords)
    tgt_mask_o = torch.ones_like(gt_mask, dtype=torch.bool)
    tgt_present_o = torch.zeros_like(gt_present)
    match_idx = torch.full_like(gt_present, fill_value=-1, dtype=torch.long)

    keep_ids = torch.nonzero(keep_identity_mask & (gt_present > 0), as_tuple=False).view(-1)
    if int(keep_ids.numel()) > 0:
        x_sel = x_in.index_select(0, keep_ids)
        g_sel = gt_coords.index_select(0, keep_ids)
        m_sel = gt_mask.index_select(0, keep_ids)
        v = (~m_sel).float()
        diff_f = (x_sel - g_sel).abs().sum(dim=-1)
        denom = v.sum(dim=1).clamp_min(1.0)
        l1_f = (diff_f * v).sum(dim=1) / denom
        g_rev = torch.flip(g_sel, dims=[1])
        m_rev = torch.flip(m_sel, dims=[1])
        diff_r = (x_sel - g_rev).abs().sum(dim=-1)
        v_rev = (~m_rev).float()
        l1_r = (diff_r * v_rev).sum(dim=1) / denom
        use_rev = (l1_r < l1_f).view(-1)
        g_final = g_sel.clone(); m_final = m_sel.clone()
        if bool(use_rev.any()):
            g_final[use_rev] = g_rev[use_rev]
            m_final[use_rev] = m_rev[use_rev]
        tgt_coords.index_copy_(0, keep_ids, g_final)
        tgt_mask_o.index_copy_(0, keep_ids, m_final)
        tgt_present_o.index_copy_(0, keep_ids, torch.ones_like(keep_ids, dtype=tgt_present_o.dtype))
        for idx in keep_ids.tolist():
            match_idx[idx] = int(idx)

    remaining_gt = (gt_present > 0).clone(); remaining_gt[keep_ids] = False
    if bool(remaining_gt.any()):
        gt_sel_idx = torch.nonzero(remaining_gt, as_tuple=False).view(-1)
        if int(gt_sel_idx.numel()) > 0:
            gt_sel_coords = gt_coords.index_select(0, gt_sel_idx)
            gt_sel_mask = gt_mask.index_select(0, gt_sel_idx)
            assigned_slots = set(keep_ids.tolist())
            # Free pred slots indices
            all_idx = torch.arange(x_in.size(0), device=x_in.device)
            free_mask = torch.ones_like(all_idx, dtype=torch.bool)
            if int(keep_ids.numel()) > 0:
                free_mask[keep_ids] = False
            pred_free_idx = all_idx[free_mask]
            if use_greedy:
                # Greedy GPU match between free preds and remaining GT
                x_free = x_in.index_select(0, pred_free_idx)
                present_rem = torch.ones((gt_sel_idx.numel(),), dtype=torch.bool, device=x_in.device)
                pairs = gpu_greedy_match(x_free, gt_sel_coords, gt_sel_mask, present_rem,
                                         w_center=1.0, w_dir=0.2, w_pw=0.5, use_chamfer=True,
                                         max_center_dist=None)
                # For each pair, choose orientation by masked L1
                for (pi_free, gj_loc) in pairs:
                    pi_i = int(pred_free_idx[int(pi_free)].item())
                    if pi_i in assigned_slots:
                        continue
                    gj_global = int(gt_sel_idx[int(gj_loc)].item())
                    if not bool(remaining_gt[gj_global]):
                        continue
                    g_sel = gt_coords[gj_global]
                    m_sel = gt_mask[gj_global]
                    v = (~m_sel).float()
                    # compare forward vs reverse masked L1
                    diff_f = (x_in[pi_i] - g_sel).abs().sum(dim=-1)
                    denom = v.sum(dim=0).clamp_min(1.0)
                    l1_f = (diff_f * v).sum() / denom
                    g_rev = torch.flip(g_sel, dims=[0])
                    m_rev = torch.flip(m_sel, dims=[0])
                    v_rev = (~m_rev).float()
                    diff_r = (x_in[pi_i] - g_rev).abs().sum(dim=-1)
                    l1_r = (diff_r * v_rev).sum() / denom
                    use_rev = bool(l1_r < l1_f)
                    if use_rev:
                        tgt_coords[pi_i] = g_rev
                        tgt_mask_o[pi_i] = m_rev
                    else:
                        tgt_coords[pi_i] = g_sel
                        tgt_mask_o[pi_i] = m_sel
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
    for orig in (1, 0, 2):  # MapTR ids: divider=1→lab0, ped=0→lab1, boundary=2→lab2
        cap = int(budgets.get(orig, 0))
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        order += [lab] * max(0, cap)
    out = np.full((N,), -1, dtype=np.int64)
    m = min(N, len(order))
    if m > 0:
        out[:m] = np.asarray(order[:m], dtype=np.int64)
    return out


class PolyDiffuseImageEncoder256(nn.Module):
    """ResNet‑50 + FPN (from official PolyDiffuse MapTR config) pooled to 256‑d.

    - BN layers are kept in eval() to avoid small-batch instability.
    - If FPN out_channels != 256, a Linear projection aligns to 256.
    """
    def __init__(self, cfg_path: str, pretrained_ckpt: str, device: str = 'cuda') -> None:
        super().__init__()
        # Defer heavy imports to runtime to keep this module lightweight
        from mmcv import Config
        from mmdet.models import build_backbone, build_neck

        cfg = Config.fromfile(cfg_path)
        bb_cfg = cfg.model.get('img_backbone')
        neck_cfg = cfg.model.get('img_neck')
        if bb_cfg is None or neck_cfg is None:
            raise RuntimeError('Config missing img_backbone/img_neck entries')

        # Build backbone and neck directly from mmdet registries
        self.backbone = build_backbone(bb_cfg)
        self.neck = build_neck(neck_cfg)

        # Load only img_backbone/img_neck weights from MapTR checkpoint
        state = torch.load(pretrained_ckpt, map_location='cpu')
        sd = state.get('state_dict', state)
        bb_sd = {k.split('img_backbone.', 1)[1]: v for k, v in sd.items() if k.startswith('img_backbone.')}
        nk_sd = {k.split('img_neck.', 1)[1]: v for k, v in sd.items() if k.startswith('img_neck.')}
        try:
            self.backbone.load_state_dict(bb_sd, strict=False)
        except Exception:
            pass
        try:
            self.neck.load_state_dict(nk_sd, strict=False)
        except Exception:
            pass
        # Determine out dim from FPN
        test_in = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.neck(self.backbone(test_in))
            # feats is a list of [B,C,H,W]; pool to [B,C]
            pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
            C = pooled[0].shape[1]
        self.proj = nn.Identity() if C == 256 else nn.Linear(C, 256)
        # Move to device
        self.to(device)

    def train(self, mode: bool = True):
        # Keep BN in eval mode for stability
        super().train(mode)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d,)):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,3,H,W]
        feats = self.neck(self.backbone(x))
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        f = torch.stack(pooled, dim=0).mean(0)  # [B,C]
        return self.proj(f)  # [B,256]


def main() -> None:
    ap = argparse.ArgumentParser(description='One-scene refine training with PolyDiffuse encoder (MapTR R50+FPN)')
    # Data
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    # Train
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--prior-lr-mult', type=float, default=2.0)
    ap.add_argument('--sched', choices=['cosine', 'none'], default='cosine')
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--accum-steps', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=8, help='Physical batch size for parallel randomized samples')
    ap.add_argument('--matcher', choices=['greedy', 'hungarian'], default='greedy', help='Assignment for non-identity slots')
    ap.add_argument('--log-every', type=int, default=1, help='Print progress every N epochs')
    # EDM (multi-step schedule)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.03)
    ap.add_argument('--reencode-per-micro', action='store_true',
                    help='Encode raster features inside each micro-batch to increase GPU utilization')
    ap.add_argument('--enc-batch', type=int, default=8,
                    help='Per-micro encoder batch when reencoding (controls encoder memory footprint)')
    # Corruption
    ap.add_argument('--shift-sigma', type=float, default=0.10)
    ap.add_argument('--point-sigma', type=float, default=0.02)
    ap.add_argument('--drop-frac', type=float, default=0.15)
    ap.add_argument('--ghosts', type=int, default=2)
    # Loss
    ap.add_argument('--l1-weight', type=float, default=20.0)
    ap.add_argument('--cls-weight', type=float, default=5.0)
    ap.add_argument('--use-focal', action='store_true')
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    ap.add_argument('--step-loss-weight', type=float, default=1.0)
    ap.add_argument('--final-loss-weight', type=float, default=1.0)
    ap.add_argument('--anchor-max-center-dist', type=float, default=0.05)
    # PolyDiffuse encoder
    ap.add_argument('--polydiff-cfg', default='official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py')
    ap.add_argument('--pretrained-maptr-ckpt', default='global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth')
    # IO
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_one_scene_polydiff')
    ap.add_argument('--viz-thr', type=float, default=0.5)
    
    # Resume (optional)
    ap.add_argument('--resume-ckpt', default=None, help='Path to ckpt_ep_xxxx.pth to resume weights from')
    ap.add_argument('--resume-epoch', type=int, default=-1, help='Epoch number of resume ckpt; training starts from resume+1')
    args = ap.parse_args()

    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Stabilize CUDA allocator behavior and matmul kernels
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    # Enable TF32 on Ampere+ to increase matmul/convolution throughput
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.allow_tf32 = True

    # Stats/caps + budgets
    import json
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    _ = RefineCaps(num_points=P, num_queries=N)

    # Load GT and raster
    gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
    bounds = gt.get('bounds')
    if bounds is None:
        raise RuntimeError('static GT lacks canonical bounds')
    gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, num_points=P, num_queries=N)
    # Preload GT tensors to device (reuse across batches)
    gt_coords_gpu = torch.from_numpy(gt_pack).float().to(device)
    gt_mask_gpu = torch.from_numpy(gt_mask).bool().to(device)
    gt_present_gpu = torch.from_numpy((~gt_mask).any(axis=1).astype(np.int64)).long().to(device)
    # Image load + normalization (ImageNet) + size to multiple of 32 for FPN
    from PIL import Image
    import torchvision.transforms.functional as TF
    raw_img = Image.open(osp.join(args.rendered_root, args.scene, '10_render_gt.png')).convert('RGB')
    w, h = raw_img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    if new_w <= 0 or new_h <= 0:
        new_w, new_h = max(32, w), max(32, h)
    if new_w != w or new_h != h:
        raw_img = raw_img.resize((new_w, new_h), Image.BILINEAR)
    img_t = TF.to_tensor(raw_img)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_t = TF.normalize(img_t, mean, std)
    ras_t = img_t.unsqueeze(0).to(device)  # [1,3,H,W]
    # Build PolyDiffuse encoder + denoiser model
    enc = PolyDiffuseImageEncoder256(args.polydiff_cfg, args.pretrained_maptr_ckpt, device=device)

    # Build PolyDiffuse encoder + denoiser model (encoder already constructed above)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)

    # LR groups: optionally freeze encoder to limit memory; otherwise, 0.1× for loaded parts
    prior_params: List[torch.nn.Parameter] = []
    prior_params += list(net.backbone.prior_mlp.parameters())
    prior_params += [net.backbone.prior_gate]
    if getattr(net.backbone, 'class_emb', None) is not None:
        prior_params += list(net.backbone.class_emb.parameters())
        prior_params += [net.backbone.class_gate]
    prior_ids = {id(p) for p in prior_params}
    rest_backbone = [p for p in net.backbone.parameters() if id(p) not in prior_ids]
    param_groups: List[dict] = []
    if bool(getattr(args, 'freeze_encoder', True)):
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        param_groups += [
            {'params': rest_backbone, 'lr': float(args.lr), 'weight_decay': 1e-4},
            {'params': prior_params, 'lr': float(args.lr) * float(args.prior_lr_mult), 'weight_decay': 1e-4},
        ]
    else:
        enc_loaded = list(enc.backbone.parameters()) + list(enc.neck.parameters())
        enc_head = list(enc.proj.parameters())
        param_groups += [
            {'params': enc_loaded, 'lr': float(args.lr) * 0.1, 'weight_decay': 1e-4},
            {'params': rest_backbone, 'lr': float(args.lr), 'weight_decay': 1e-4},
            {'params': enc_head, 'lr': float(args.lr), 'weight_decay': 1e-4},
            {'params': prior_params, 'lr': float(args.lr) * float(args.prior_lr_mult), 'weight_decay': 1e-4},
        ]
    opt = torch.optim.AdamW(param_groups)
    if args.sched == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=float(args.lr_min))
    else:
        sched = None

    out_dir = osp.join(args.out_root, args.scene)
    viz_dir = osp.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    accum_steps = max(1, int(args.accum_steps))
    B = max(1, int(args.batch_size))
    scaler = torch.cuda.amp.GradScaler()
    # Strict alignment with official: FP32 by default; avoid AMP/scaler to keep allocator stable
    start_epoch = int(getattr(args, 'resume_epoch', -1)) + 1 if int(getattr(args, 'resume_epoch', -1)) >= 0 else 1
    # Optionally resume weights
    if getattr(args, 'resume_ckpt', None):
        try:
            data = torch.load(args.resume_ckpt, map_location=device)
            if 'encoder' in data:
                enc.load_state_dict(data['encoder'], strict=False)
            if 'net' in data:
                base.load_state_dict(data['net'], strict=False)
            print(f"[resume] loaded: {args.resume_ckpt} (epoch={getattr(args,'resume_epoch',-1)})")
        except Exception as e:
            print(f"[warn] resume failed: {e}")

    # Pre-allocate diffusion noise buffer to reduce allocator churn
    noise_buf = torch.empty((B, N, P, 2), device=device)

    for ep in range(start_epoch, int(args.epochs) + 1):
        enc.train(); base.train()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        ep_loss_sum = 0.0; ep_step_sum = 0.0; ep_final_sum = 0.0
        opt.zero_grad(set_to_none=True)

        # Encode raster once and expand to batch
        with torch.cuda.amp.autocast():
            rv_single_epoch = enc(ras_t)  # [1,256]
        rv_expand_epoch = rv_single_epoch.expand(B, -1)

        last_x_np = None; last_keep_identity_np = None; last_is_drop_np = None; last_labels_by_slot = None
        preds = None; pred_coords = None; pred_logits = None

        for micro in range(accum_steps):
            # Generate entire batch on GPU
            x, keep_identity = jitter_drop_ghost_batch(
                gt_coords_gpu.unsqueeze(0), gt_mask_gpu.unsqueeze(0),
                batch_size=B,
                shift_sigma=float(args.shift_sigma), point_sigma=float(args.point_sigma),
                drop_frac=float(args.drop_frac), ghosts=int(args.ghosts))

            # Build targets per batch item (GPU-friendly loop)
            batch_tgt_coords: List[torch.Tensor] = []
            batch_tgt_mask: List[torch.Tensor] = []
            batch_tgt_present: List[torch.Tensor] = []
            batch_tgt_labels: List[torch.Tensor] = []
            batch_prior_labels: List[torch.Tensor] = []
            labels_all_np = _labels_from_budgets(budgets, N)
            labels_base = torch.from_numpy(labels_all_np).long().to(device)
            for b in range(B):
                keep_b = keep_identity[b]
                tgt_c_b, tgt_m_b, tgt_p_b, match_idx = build_fixed_targets(
                    x[b], gt_coords_gpu, gt_mask_gpu, gt_present_gpu, keep_identity_mask=keep_b,
                    use_greedy=(args.matcher == 'greedy'))
                labels_slot = torch.full((N,), -1, dtype=torch.long, device=device)
                valid_match = (match_idx >= 0) & (match_idx < N)
                if bool(valid_match.any()):
                    labels_slot[valid_match] = labels_base[match_idx[valid_match]]
                input_prior = labels_slot.clone()
                input_prior[~keep_b] = -1
                batch_tgt_coords.append(tgt_c_b)
                batch_tgt_mask.append(tgt_m_b)
                batch_tgt_present.append(tgt_p_b)
                batch_tgt_labels.append(labels_slot)
                batch_prior_labels.append(input_prior)

            tgt_coords = torch.stack(batch_tgt_coords, dim=0)
            tgt_mask_o = torch.stack(batch_tgt_mask, dim=0)
            tgt_present_o = torch.stack(batch_tgt_present, dim=0)
            tgt_labels = torch.stack(batch_tgt_labels, dim=0)
            input_prior_labels = torch.stack(batch_prior_labels, dim=0)

            # Cache last sample for viz
            last_x_np = x[-1].detach().cpu().numpy()
            last_keep_identity_np = keep_identity[-1].detach().cpu().numpy()
            last_is_drop_np = (~last_keep_identity_np).astype(np.bool_)
            last_labels_by_slot = tgt_labels[-1].detach().cpu().numpy()

            # SDEdit start around x
            # Use static noise buffer to avoid per-step allocation
            noise_buf.normal_()
            xK = torch.clamp((1.0 - float(args.alpha)) * x + float(args.alpha) * noise_buf, -1.0, 1.0)
            sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
            with torch.cuda.amp.autocast():
                if bool(getattr(args, 'reencode_per_micro', False)):
                    # Chunked encoder batches to avoid OOM while improving utilization
                    enc_bs = max(1, int(getattr(args, 'enc_batch', 8)))
                    chunks = []
                    cur = 0
                    while cur < B:
                        k = min(enc_bs, B - cur)
                        # Use repeat to materialize a contiguous batch [k,3,H,W]
                        rv_chunk = enc(ras_t.repeat(k, 1, 1, 1))  # [k,256]
                        chunks.append(rv_chunk)
                        cur += k
                    rv_expand = torch.cat(chunks, dim=0)  # [B,256]
                else:
                    rv_expand = rv_expand_epoch
                use_step_loss = float(args.step_loss_weight) > 0.0
                pred_coords, pred_logits, preds, _states = edm_unrolled_train(
                    net, xK, rv_expand, sigmas, second_order=bool(args.second_order),
                    cond_prior=x, input_labels=input_prior_labels, collect_states=False,
                    collect_preds=use_step_loss)

                if use_step_loss:
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
                else:
                    step_loss = pred_coords.new_zeros([])
                out_final = criterion(pred_coords, pred_logits, tgt_coords, tgt_mask_o, tgt_present_o,
                                      l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                                      use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                      pred_sem_logits=(preds[-1][2] if (len(preds) > 0 and len(preds[-1]) >= 3) else None),
                                      tgt_sem_labels=tgt_labels, sem_weight=float(args.sem_weight))
                final_loss = out_final['loss_cls'] + out_final['loss_reg'] + out_final.get('loss_sem', pred_coords.new_zeros([]))
            # Snapshot small CPU-only viz labels to avoid holding graph refs
            viz_sem_labels_np = None
            try:
                if len(preds) > 0 and len(preds[-1]) >= 3 and preds[-1][2] is not None:
                    sem_np = preds[-1][2][-1].detach().cpu().numpy()
                    viz_sem_labels_np = np.argmax(sem_np, axis=-1).astype(np.int64)
            except Exception:
                viz_sem_labels_np = None
            preds = None  # drop graph refs before backward
            loss = float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss

            # When rv is shared across micro-batches, retain until last micro
            retain = (micro < accum_steps - 1)
            scaler.scale(loss / float(accum_steps)).backward(retain_graph=retain)
            ep_loss_sum += float(loss.item())
            ep_step_sum += float(step_loss.item())
            ep_final_sum += float(final_loss.item())

            # Inner-loop cleanup to avoid incremental growth
            try:
                # Drop large per-micro tensors ASAP
                del tgt_coords, tgt_mask_o, tgt_present_o, tgt_labels, input_prior_labels
                del x, xK
                import gc; gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

        scaler.step(opt)
        scaler.update()
        # Clear grads immediately after step to release memory
        try:
            opt.zero_grad(set_to_none=True)
        except Exception:
            pass
        if sched is not None:
            sched.step()

        # Drop large graph-carrying tensors eagerly unless needed for viz
        keep_for_viz = (not bool(getattr(args, 'no_viz', True))) and ((ep % 50 == 0) or (ep == int(args.epochs)))
        if not keep_for_viz:
            pred_coords = None
            pred_logits = None

        if (ep % int(args.log_every) == 0) or (ep == 1):
            cur_lr = opt.param_groups[0]['lr']
            avg_loss = ep_loss_sum / float(accum_steps)
            avg_step = ep_step_sum / float(accum_steps)
            avg_final = ep_final_sum / float(accum_steps)
            print(f"[ep {ep:04d}] steps={int(args.steps)} lr={cur_lr:.6g} batch={B} accum={accum_steps} loss={avg_loss:.6f} step={avg_step:.6f} final={avg_final:.6f}")

        # periodic viz + ckpt
        if (ep % 50 == 0) or (ep == int(args.epochs)):
            os.makedirs(out_dir, exist_ok=True)
            with torch.no_grad():
                # viz noise + refined overlay
                if last_x_np is not None and last_keep_identity_np is not None and last_is_drop_np is not None and last_labels_by_slot is not None:
                    Nviz_all = last_x_np.shape[0]
                    mask_all = np.ones((Nviz_all, P), dtype=bool)
                    draw_ids = np.where(last_keep_identity_np | last_is_drop_np)[0].tolist()
                    for i in draw_ids:
                        mask_all[i, :] = False
                    title_noise = f"noise ep{ep:04d} shift={args.shift_sigma:.2f} local={args.point_sigma:.2f} drop={args.drop_frac:.2f} ghosts={int(args.ghosts)}"
                    # denormalize raster back to [0,1] for correct colors in viz
                    mean_v = torch.tensor([0.485, 0.456, 0.406], device=ras_t.device).view(3,1,1)
                    std_v = torch.tensor([0.229, 0.224, 0.225], device=ras_t.device).view(3,1,1)
                    ras_viz_np = (ras_t[0] * std_v + mean_v).clamp(0.0, 1.0).detach().cpu().numpy()
                    overlay_slots_annot(osp.join(viz_dir, f'noise_{ep:04d}.png'), ras_viz_np, bounds,
                                        slots=last_x_np, mask=mask_all, labels=last_labels_by_slot, title=title_noise,
                                        gt_slots=gt_pack, gt_mask=gt_mask)
                if pred_coords is not None and pred_logits is not None:
                    pc = pred_coords[-1].detach().cpu().numpy()
                    pl = pred_logits[-1].detach().cpu().numpy().reshape(-1)
                    prob = 1.0 / (1.0 + np.exp(-pl))
                    keep = np.where(prob >= float(args.viz_thr))[0].tolist()
                    mask = np.ones((pc.shape[0], P), dtype=bool)
                    for i in keep:
                        if 0 <= i < mask.shape[0]:
                            mask[i, :] = False
                    sem_labels = viz_sem_labels_np
                    # reuse denormalized viz image
                    overlay_slots_annot(osp.join(viz_dir, f'ep_{ep:04d}.png'), ras_viz_np, bounds,
                                        slots=pc, mask=mask, labels=sem_labels, title=f'ep_{ep:04d} (thr={args.viz_thr})')
            # ckpt (save encoder too)
            torch.save({'encoder': enc.state_dict(), 'net': (base.state_dict()), 'P': P, 'N': N, 'budgets': budgets},
                       osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth'))
            # pro‑actively drop references to free cached graph tensors
            pred_coords = None; pred_logits = None
            try:
                del rv_expand
            except Exception:
                pass
            try:
                import gc; gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        # End-of-epoch cleanup
        try:
            import gc; gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Memory footprint log
        try:
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserv = torch.cuda.memory_reserved() / (1024**2)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"[mem ep {ep:04d}] alloc_MB={alloc:.1f} reserved_MB={reserv:.1f} peak_MB={peak:.1f}")
        except Exception:
            pass

    print(f"[ok] training (PolyDiffuse encoder) done. Viz: {viz_dir}")


if __name__ == '__main__':
    main()
