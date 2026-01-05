#!/usr/bin/env python
from __future__ import annotations

"""
One-scene refine training (EDM) — Stable Memory + Full Matching (GPU5 behavior, no creep).

Goals
- 保留 GPU5 版本的“快下降”特性：开启重匹配（greedy/hungarian）、step+final 双监督、steps>1。
- 引入稳定显存策略：Split-Backward（编码器与去噪器解耦）、静态缓冲区、关闭 cuDNN benchmark、分配器参数。

使用说明
- 与 train_one_scene_polydiffuse_encoder.py 等价的 loss/匹配逻辑，但内存更稳定。
- 默认冻结 encoder；如需训练 encoder，可加 --no-freeze-encoder（仍保持 split-backward）。
"""

import os
# 限制块拆分大小，降低碎片化；阈值可按需微调
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")

import argparse
import os.path as osp
import gc
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 数值/性能：TF32 打开；禁用 benchmark，避免算法缓存增长
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.loss_refine import criterion, hungarian_match_perm, gpu_greedy_match
from global_diffusion_map.refine.single_scene_overfit import overlay_slots_annot, load_pickle
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class StaticDataBuffer:
    """预分配数据生成/扩散所需的所有大张量，避免热循环反复申请导致碎片化。"""
    def __init__(self, batch_size: int, N: int, P: int, device: torch.device | str) -> None:
        self.B, self.N, self.P = batch_size, N, P
        self.device = device
        self.x_buf = torch.zeros(batch_size, N, P, 2, device=device)
        self.mask_buf = torch.zeros(batch_size, N, P, dtype=torch.bool, device=device)
        self.shifts = torch.zeros(batch_size, N, 1, 2, device=device)
        self.local = torch.zeros(batch_size, N, P, 2, device=device)
        self.rnd_drop = torch.zeros(batch_size, N, device=device)
        self.ghost_base = torch.zeros(batch_size, N, P, 2, device=device)
        # diffusion
        self.diff_noise = torch.zeros(batch_size, N, P, 2, device=device)
        self.xK = torch.zeros(batch_size, N, P, 2, device=device)

    def generate(self, gt_coords: torch.Tensor, gt_mask: torch.Tensor,
                 shift_sigma: float, point_sigma: float, drop_frac: float, ghosts: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.x_buf.copy_(gt_coords.expand(self.B, self.N, self.P, 2))
        self.mask_buf.copy_(gt_mask.expand(self.B, self.N, self.P))
        valid = ~self.mask_buf
        # noise in-place
        self.shifts.normal_().mul_(shift_sigma)
        self.local.normal_().mul_(point_sigma)
        noise_view = self.local + self.shifts
        x_valid = self.x_buf[valid]
        x_valid.add_(noise_view[valid]).clamp_(-1.0, 1.0)
        self.x_buf[valid] = x_valid
        # drop
        present = valid.any(dim=2)
        self.rnd_drop.uniform_()
        is_drop = present & (self.rnd_drop < drop_frac)
        keep_identity = present & (~is_drop)
        if is_drop.any():
            self.local.normal_().mul_(max(shift_sigma, point_sigma) * 0.8)
            base = gt_coords.expand(self.B, self.N, self.P, 2)
            self.x_buf[is_drop] = (base[is_drop] + self.local[is_drop]).clamp_(-1.0, 1.0)
        # ghost
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
            all_idx = torch.arange(x_in.size(0), device=x_in.device)
            free_mask = torch.ones_like(all_idx, dtype=torch.bool)
            if int(keep_ids.numel()) > 0:
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
                    v = (~m_sel).float()
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
    for orig in (1, 0, 2):  # divider→0, ped→1, boundary→2
        cap = int(budgets.get(orig, 0))
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        order += [lab] * max(0, cap)
    out = np.full((N,), -1, dtype=np.int64)
    m = min(N, len(order))
    if m > 0:
        out[:m] = np.asarray(order[:m], dtype=np.int64)
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
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.neck(self.backbone(x))
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        return self.proj(torch.stack(pooled, dim=0).mean(0))


def run_micro_step(
    net: nn.Module,
    rv_expand: torch.Tensor,
    scaler: torch.cuda.amp.GradScaler,
    gt_coords: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_present: torch.Tensor,
    labels_base: torch.Tensor,
    args: argparse.Namespace,
    accum_steps: int,
    buffer: StaticDataBuffer,
) -> Tuple[float, float, float, Optional[Dict]]:
    device = rv_expand.device
    N = gt_coords.size(0)
    P = gt_coords.size(1)
    B = int(args.batch_size)

    x, keep_identity = buffer.generate(
        gt_coords, gt_mask, args.shift_sigma, args.point_sigma, args.drop_frac, args.ghosts
    )

    batch_tgt = {'c': [], 'm': [], 'p': [], 'l': [], 'prior': []}
    for b in range(B):
        keep_b = keep_identity[b]
        tgt_c, tgt_m, tgt_p, match_idx = build_fixed_targets(
            x[b], gt_coords, gt_mask, gt_present, keep_b, use_greedy=(args.matcher == 'greedy')
        )
        labels_slot = torch.full((N,), -1, dtype=torch.long, device=device)
        valid = (match_idx >= 0) & (match_idx < N)
        if valid.any():
            labels_slot[valid] = labels_base[match_idx[valid]]
        input_prior = labels_slot.clone(); input_prior[~keep_b] = -1
        batch_tgt['c'].append(tgt_c)
        batch_tgt['m'].append(tgt_m)
        batch_tgt['p'].append(tgt_p)
        batch_tgt['l'].append(labels_slot)
        batch_tgt['prior'].append(input_prior)

    tgt_coords = torch.stack(batch_tgt['c'])
    tgt_mask_o = torch.stack(batch_tgt['m'])
    tgt_present_o = torch.stack(batch_tgt['p'])
    tgt_labels = torch.stack(batch_tgt['l'])
    input_prior = torch.stack(batch_tgt['prior'])
    del batch_tgt

    # diffusion
    buffer.diff_noise.normal_()
    buffer.xK.copy_(x).mul_(1.0 - args.alpha).add_(buffer.diff_noise, alpha=args.alpha).clamp_(-1, 1)
    sigmas = karras_schedule(args.steps, args.sigma_min, args.sigma_max, args.rho).to(device)

    with torch.cuda.amp.autocast():
        pred_coords, pred_logits, preds, _ = edm_unrolled_train(
            net, buffer.xK, rv_expand, sigmas,
            second_order=args.second_order,
            cond_prior=x, input_labels=input_prior,
            collect_states=False,
            collect_preds=(args.step_loss_weight > 0)
        )
        step_loss = torch.tensor(0.0, device=device)
        if args.step_loss_weight > 0 and preds:
            losses = []
            for item in preds:
                pc, pl = item[0], item[1]
                sem = item[2] if len(item) > 2 else None
                out = criterion(pc, pl, tgt_coords, tgt_mask_o, tgt_present_o,
                                l1_weight=args.l1_weight, cls_weight=args.cls_weight,
                                use_focal=args.use_focal, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                                pred_sem_logits=sem, tgt_sem_labels=tgt_labels, sem_weight=args.sem_weight)
                losses.append(out['loss_cls'] + out['loss_reg'] + out.get('loss_sem', 0.0))
            step_loss = torch.stack(losses).mean()

        final_sem = preds[-1][2] if (preds and len(preds[-1]) > 2) else None
        out_final = criterion(pred_coords, pred_logits, tgt_coords, tgt_mask_o, tgt_present_o,
                              l1_weight=args.l1_weight, cls_weight=args.cls_weight,
                              use_focal=args.use_focal, focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                              pred_sem_logits=final_sem, tgt_sem_labels=tgt_labels, sem_weight=args.sem_weight)
        final_loss = out_final['loss_cls'] + out_final['loss_reg'] + out_final.get('loss_sem', 0.0)
        total_loss = (args.step_loss_weight * step_loss + args.final_loss_weight * final_loss) / accum_steps

    # backward（不保留图）：去噪器图立即释放，显存常量
    scaler.scale(total_loss).backward()

    l_val = total_loss.item() * accum_steps
    s_val = step_loss.item()
    f_val = final_loss.item()

    viz_data = None
    if args.need_viz:
        viz_data = {
            'x': x[-1].detach().cpu().numpy(),
            'keep': keep_identity[-1].detach().cpu().numpy(),
            'pred_c': pred_coords[-1].detach().cpu().numpy(),
            'pred_l': pred_logits[-1].detach().cpu().numpy(),
            'tgt_l': tgt_labels[-1].detach().cpu().numpy(),
        }
        if final_sem is not None:
            viz_data['sem'] = final_sem[-1].detach().cpu().numpy()

    # 立即释放局部引用
    del pred_coords, pred_logits, preds, total_loss, step_loss, final_loss, tgt_coords, input_prior
    return l_val, s_val, f_val, viz_data


def main() -> None:
    ap = argparse.ArgumentParser()
    # IO
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_one_scene_polydiff')
    # Train
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--accum-steps', type=int, default=1)
    # Encoder 控制
    ap.add_argument('--freeze-encoder', action='store_true', default=True)
    ap.add_argument('--no-freeze-encoder', dest='freeze_encoder', action='store_false')
    # Model
    ap.add_argument('--polydiff-cfg', default='official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py')
    ap.add_argument('--pretrained-maptr-ckpt', default='global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth')
    # EDM
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.4)
    ap.add_argument('--alpha', type=float, default=0.03)
    ap.add_argument('--second-order', action='store_true')
    # 数据增强/匹配
    ap.add_argument('--matcher', choices=['greedy', 'hungarian'], default='greedy')
    ap.add_argument('--shift-sigma', type=float, default=0.10)
    ap.add_argument('--point-sigma', type=float, default=0.02)
    ap.add_argument('--drop-frac', type=float, default=0.15)
    ap.add_argument('--ghosts', type=int, default=2)
    # Loss
    ap.add_argument('--l1-weight', type=float, default=20.0)
    ap.add_argument('--cls-weight', type=float, default=5.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    ap.add_argument('--step-loss-weight', type=float, default=1.0)
    ap.add_argument('--final-loss-weight', type=float, default=1.0)
    ap.add_argument('--use-focal', action='store_true')
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    # Sched/opt
    ap.add_argument('--prior-lr-mult', type=float, default=3.0)
    ap.add_argument('--sched', choices=['cosine', 'none'], default='cosine')
    ap.add_argument('--lr-min', type=float, default=1e-5)
    # Misc
    ap.add_argument('--resume-ckpt', default=None)
    ap.add_argument('--resume-epoch', type=int, default=-1)
    ap.add_argument('--log-every', type=int, default=1)
    ap.add_argument('--viz-thr', type=float, default=0.5)

    args = ap.parse_args()
    set_seed(0)
    device = torch.device('cuda')

    # 数据加载
    import json
    with open(args.stats_json, 'r') as f: stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
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

    # 模型与缓冲
    enc = PolyDiffuseImageEncoder256(args.polydiff_cfg, args.pretrained_maptr_ckpt, device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    buffer = StaticDataBuffer(args.batch_size, N, P, device)

    # 优化器
    params = []
    if args.freeze_encoder:
        enc.eval()
        for p in enc.parameters(): p.requires_grad_(False)
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

    print(f"[start] stable+matching, alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    for ep in range(start_ep, args.epochs + 1):
        net.train()
        enc.train() if not args.freeze_encoder else enc.eval()
        opt.zero_grad(set_to_none=True)

        # 编码器一次/epoch（Split-Backward）
        enc_ctx = torch.no_grad() if args.freeze_encoder else torch.enable_grad()
        with torch.cuda.amp.autocast(), enc_ctx:
            rv_single = enc(ras_t)
        if not args.freeze_encoder:
            rv_use = rv_single.detach(); rv_use.requires_grad_(True)
        else:
            rv_use = rv_single
        rv_expand = rv_use.expand(args.batch_size, -1)

        l_sum = s_sum = f_sum = 0.0
        last_viz: Optional[Dict] = None
        args.need_viz = (ep % 50 == 0 or ep == args.epochs)

        for micro in range(args.accum_steps):
            if micro < args.accum_steps - 1: args.need_viz = False
            elif (ep % 50 == 0 or ep == args.epochs): args.need_viz = True
            ls, ss, fs, viz_d = run_micro_step(
                net, rv_expand, scaler, gt_coords_gpu, gt_mask_gpu, gt_present_gpu, labels_base,
                args, args.accum_steps, buffer
            )
            l_sum += ls; s_sum += ss; f_sum += fs
            if viz_d: last_viz = viz_d

        if not args.freeze_encoder and rv_use.grad is not None:
            scaler.scale(rv_single).backward(rv_use.grad)

        scaler.step(opt)
        scaler.update()
        if sched: sched.step()

        del rv_expand
        if not args.freeze_encoder: del rv_use

        if ep % args.log_every == 0 or ep == 1:
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_res = torch.cuda.memory_reserved() / 1024**2
            print(f"[ep {ep:04d}] steps={args.steps} lr={opt.param_groups[0]['lr']:.6g} "
                  f"loss={l_sum/args.accum_steps:.4f} (step={s_sum/args.accum_steps:.4f} final={f_sum/args.accum_steps:.4f}) "
                  f"mem={mem_alloc:.0f}/{mem_res:.0f} MB")

        # 周期性保存 checkpoint（含 encoder + denoiser），供后续推理/对比使用
        # 与 train_one_scene_polydiffuse_encoder.py 保持一致：保存 SlotMLP backbone 权重到 "net" 键。
        if (ep % 50 == 0) or (ep == args.epochs):
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth')
            torch.save(
                {
                    'encoder': enc.state_dict(),
                    'net': net.backbone.state_dict(),
                    'P': P,
                    'N': N,
                    'budgets': budgets,
                },
                ckpt_path,
            )

        # 轻量清理（避免碎片化长时间累积）
        if ep % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[ok] done. out={out_dir}")


if __name__ == '__main__':
    main()
