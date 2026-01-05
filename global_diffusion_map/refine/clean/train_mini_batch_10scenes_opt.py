#!/usr/bin/env python
from __future__ import annotations

"""
10-Scene Mini-Batch Refine Training (Optimized).

Improvements vs. stable DDP trainer:
- Fixed input size via --img-size (default 512) to speed up the encoder dramatically.
- End-to-end DDP (no split-backward); simpler and faster gradient sync.
- Phase-A short-circuit: when there are no drops/ghosts, skip heavy matching and use GT directly.

Usage (example):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    global_diffusion_map/refine/clean/train_mini_batch_10scenes_opt.py \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --scenes S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \
    --encoder resnet --resnet resnet50 --img-size 512 \
    --batch-size 1 --workers 0 --epochs 2400 --phase-a-epochs 2000
"""

import os
import os.path as osp
import json
import argparse
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Performance settings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
import sys
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots, pack_vectors_to_slots
from global_diffusion_map.refine.loss_refine import criterion, hungarian_match_perm, gpu_greedy_match
from global_diffusion_map.refine.single_scene_overfit import load_pickle, RasterEncoder
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train


def set_seed(seed: int = 0) -> None:
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def is_ddp() -> bool:
    return int(os.environ.get('WORLD_SIZE', '1')) > 1


def ddp_setup() -> int:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class TenScenesDataset(Dataset):
    """Scene dataset with lazy loading and fixed-size rasterization.

    For each index, loads static GT, condition raster, and optionally aggregated
    proposals from disk. This avoids preloading all scenes into RAM and keeps
    startup time reasonable for large splits (~694 scenes).
    """

    def __init__(
        self,
        static_root: str,
        rendered_root: str,
        scenes: Sequence[str],
        stats_json: str,
        img_size: int = 512,
        agg_pred_root: str | None = None,
    ) -> None:
        super().__init__()
        self.static_root = static_root
        self.rendered_root = rendered_root
        self.agg_pred_root = agg_pred_root
        self.scenes = list(scenes)
        self.img_size = int(img_size)

        with open(stats_json, 'r') as f:
            stats = json.load(f)
        self.P = int(stats.get('M', 20))
        self.N = int(stats.get('num_queries', 64))
        self.budgets = {
            int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()
        }

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        from PIL import Image
        import torchvision.transforms.functional as TF

        sid = self.scenes[idx]
        # Static GT
        gt_pkl = osp.join(self.static_root, f'{sid}.pkl')
        gt = load_pickle(gt_pkl)
        bounds = gt.get('bounds')
        if bounds is None:
            raise RuntimeError(f'static GT lacks bounds: {sid}')
        gt_pack, gt_mask, gt_present = pack_gt_to_slots(
            gt, bounds, self.budgets, num_points=self.P, num_queries=self.N
        )

        # Condition raster (10_render_gt.png) resized to fixed img_size
        raw = Image.open(osp.join(self.rendered_root, sid, '10_render_gt.png')).convert('RGB')
        if self.img_size > 0:
            raw = raw.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_t = TF.normalize(
            TF.to_tensor(raw),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        item: Dict[str, object] = {
            'scene': sid,
            'ras': img_t,
            'gt_pack': torch.from_numpy(gt_pack).float(),
            'gt_mask': torch.from_numpy(gt_mask).bool(),
            'gt_present': torch.from_numpy(gt_present).long(),
        }

        # Aggregated proposals (optional, used in Phase‑C)
        if self.agg_pred_root:
            agg_pkl = osp.join(self.agg_pred_root, f'{sid}.pkl')
            if not osp.exists(agg_pkl):
                raise RuntimeError(f'aggregated proposal missing for scene {sid}: {agg_pkl}')
            agg = load_pickle(agg_pkl)
            prop_pack, prop_mask, prop_labels = pack_vectors_to_slots(
                agg, bounds, self.budgets, num_points=self.P, num_queries=self.N
            )
            item['prop_pack'] = torch.from_numpy(prop_pack).float()
            item['prop_mask'] = torch.from_numpy(prop_mask).bool()
            item['prop_labels'] = torch.from_numpy(prop_labels).long()

        return item


class StaticDataBuffer:
    def __init__(self, B: int, N: int, P: int, device: torch.device | str) -> None:
        self.B, self.N, self.P = int(B), int(N), int(P)
        self.device = device
        self.x_buf = torch.zeros(B, N, P, 2, device=device)
        self.mask_buf = torch.zeros(B, N, P, dtype=torch.bool, device=device)
        self.shifts = torch.zeros(B, N, 1, 2, device=device)
        self.local = torch.zeros(B, N, P, 2, device=device)
        self.rnd_drop = torch.zeros(B, N, device=device)
        self.ghost_base = torch.zeros(B, N, P, 2, device=device)
        self.diff_noise = torch.zeros(B, N, P, 2, device=device)
        self.xK = torch.zeros(B, N, P, 2, device=device)

    @torch.no_grad()
    def generate(self, gt_coords_batch, gt_mask_batch, shift_sigma, point_sigma, drop_frac, ghosts):
        self.x_buf.copy_(gt_coords_batch)
        self.mask_buf.copy_(gt_mask_batch)
        valid = ~self.mask_buf
        # jitter
        self.shifts.normal_().mul_(shift_sigma)
        self.local.normal_().mul_(point_sigma)
        x_valid = self.x_buf[valid]
        x_valid.add_((self.local + self.shifts)[valid]).clamp_(-1.0, 1.0)
        self.x_buf[valid] = x_valid
        # drops/ghosts
        present = valid.any(dim=2)
        keep_identity = present.clone()
        if drop_frac > 0 or int(ghosts) > 0:
            self.rnd_drop.uniform_()
            is_drop = present & (self.rnd_drop < drop_frac)
            keep_identity = present & (~is_drop)
            if is_drop.any():
                self.local.normal_().mul_(max(shift_sigma, point_sigma) * 0.8)
                self.x_buf[is_drop] = (gt_coords_batch[is_drop] + self.local[is_drop]).clamp_(-1.0, 1.0)
            k = int(min(max(0, int(ghosts)), self.N))
            if k > 0:
                empty = ~present
                self.rnd_drop.uniform_(); self.rnd_drop[present] = -1.0
                _, idx = torch.topk(self.rnd_drop, k=k, dim=1)
                is_ghost = torch.zeros_like(present)
                is_ghost.scatter_(1, idx, True); is_ghost &= empty
                self.ghost_base.uniform_(-1.0, 1.0).mul_(0.6)
                for t in range(1, self.P):
                    self.ghost_base[:, :, t].mul_(0.3).add_(self.ghost_base[:, :, t - 1], alpha=0.7)
                self.x_buf[is_ghost] = self.ghost_base[is_ghost].clamp_(-1.0, 1.0)
        return self.x_buf, keep_identity


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


class StandardResNetEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, version: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        # Handle both newer and older torchvision APIs
        try:
            if version == 'resnet18':
                m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
                d = 512
            elif version == 'resnet34':
                m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
                d = 512
            elif version == 'resnet50':
                m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
                d = 2048
            else:
                raise ValueError(f"Unknown resnet: {version}")
        except Exception:
            if version == 'resnet18':
                m = models.resnet18(pretrained=pretrained); d = 512
            elif version == 'resnet34':
                m = models.resnet34(pretrained=pretrained); d = 512
            elif version == 'resnet50':
                m = models.resnet50(pretrained=pretrained); d = 2048
            else:
                raise
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.proj = nn.Linear(d, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)


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
            if use_greedy:
                x_free = x_in
                present_rem = torch.ones((gt_sel_idx.numel(),), dtype=torch.bool, device=x_in.device)
                pairs = gpu_greedy_match(x_free, gt_sel_coords, gt_sel_mask, present_rem,
                                         w_center=1.0, w_dir=0.2, w_pw=0.5, use_chamfer=True,
                                         max_center_dist=None)
                for (pi, gj_loc) in pairs:
                    pi_i = int(pi)
                    if pi_i in assigned_slots: continue
                    gj_global = int(gt_sel_idx[int(gj_loc)].item())
                    if not bool(remaining_gt[gj_global]): continue
                    g_sel = gt_coords[gj_global]; m_sel = gt_mask[gj_global]
                    v = (~m_sel).float()
                    diff_f = (x_in[pi_i] - g_sel).abs().sum(dim=-1)
                    denom = v.sum().clamp_min(1.0)
                    l1_f = (diff_f * v).sum() / denom
                    g_rev = torch.flip(g_sel, dims=[0])
                    m_rev = torch.flip(m_sel, dims=[0])
                    v_rev = (~m_rev).float()
                    diff_r = (x_in[pi_i] - g_rev).abs().sum(dim=-1)
                    l1_r = (diff_r * v_rev).sum() / denom
                    use_rev = bool(l1_r < l1_f)
                    if use_rev:
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
                    if pi_i in assigned_slots: continue
                    gj_global = int(gt_sel_idx[int(gj_loc)].item())
                    if not bool(remaining_gt[gj_global]): continue
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


def run_micro_step_fast(
    net: nn.Module,
    rv: torch.Tensor,
    gt_coords: torch.Tensor, gt_mask: torch.Tensor, gt_present: torch.Tensor,
    labels_base: torch.Tensor,
    args: argparse.Namespace,
    buffer: StaticDataBuffer,
    is_phase_a: bool,
    use_greedy: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = rv.device
    B = gt_coords.size(0)

    # Phase control
    if is_phase_a:
        s_sigma, p_sigma, drop_f, n_ghosts = float(args.shift_sigma), float(args.point_sigma), 0.0, 0
    else:
        s_sigma, p_sigma, drop_f, n_ghosts = float(args.shift_sigma), float(args.point_sigma), float(args.drop_frac), int(args.ghosts)

    x, keep_identity = buffer.generate(gt_coords, gt_mask, s_sigma, p_sigma, drop_f, n_ghosts)

    # Phase A: shortcut targets (identity)
    if is_phase_a:
        tgt_coords = gt_coords
        tgt_mask_o = gt_mask
        tgt_present_o = gt_present
        tgt_labels = labels_base.unsqueeze(0).expand(B, -1)
        input_prior = tgt_labels
    else:
        batch_tgt = {'c': [], 'm': [], 'p': [], 'l': [], 'prior': []}
        for b in range(B):
            tgt_c, tgt_m, tgt_p, match_idx = build_fixed_targets(
                x[b], gt_coords[b], gt_mask[b], gt_present[b], keep_identity[b], use_greedy
            )
            labels_slot = torch.full((buffer.N,), -1, dtype=torch.long, device=device)
            valid = (match_idx >= 0) & (match_idx < buffer.N)
            if valid.any():
                labels_slot[valid] = labels_base[match_idx[valid]]
            input_prior_b = labels_slot.clone(); input_prior_b[~keep_identity[b]] = -1
            batch_tgt['c'].append(tgt_c); batch_tgt['m'].append(tgt_m)
            batch_tgt['p'].append(tgt_p); batch_tgt['l'].append(labels_slot)
            batch_tgt['prior'].append(input_prior_b)

        tgt_coords = torch.stack(batch_tgt['c'])
        tgt_mask_o = torch.stack(batch_tgt['m'])
        tgt_present_o = torch.stack(batch_tgt['p'])
        tgt_labels = torch.stack(batch_tgt['l'])
        input_prior = torch.stack(batch_tgt['prior'])

    # Diffusion
    buffer.diff_noise.normal_()
    buffer.xK.copy_(x).mul_(1.0 - float(args.alpha)).add_(buffer.diff_noise, alpha=float(args.alpha)).clamp_(-1, 1)
    sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)

    # Forward denoiser
    pred_coords, pred_logits, preds, _ = edm_unrolled_train(
        net, buffer.xK, rv, sigmas,
        second_order=bool(args.second_order),
        cond_prior=x, input_labels=input_prior,
        collect_states=False, collect_preds=(float(args.step_loss_weight) > 0.0)
    )

    # Losses
    step_loss = torch.tensor(0.0, device=device)
    if preds and float(args.step_loss_weight) > 0.0:
        losses = []
        for item in preds:
            pc, pl = item[0], item[1]
            sem = item[2] if len(item) > 2 else None
            out = criterion(pc, pl, tgt_coords, tgt_mask_o, tgt_present_o,
                            l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                            use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                            pred_sem_logits=sem, tgt_sem_labels=tgt_labels, sem_weight=float(args.sem_weight))
            losses.append(out['loss_cls'] + out['loss_reg'] + out.get('loss_sem', 0.0))
        step_loss = torch.stack(losses).mean()

    final_sem = preds[-1][2] if (preds and len(preds[-1]) > 2) else None
    out_final = criterion(pred_coords, pred_logits, tgt_coords, tgt_mask_o, tgt_present_o,
                          l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                          use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                          pred_sem_logits=final_sem, tgt_sem_labels=tgt_labels, sem_weight=float(args.sem_weight))
    final_loss = out_final['loss_cls'] + out_final['loss_reg'] + out_final.get('loss_sem', 0.0)

    total_loss = (float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss)
    return total_loss, step_loss, final_loss


def run_micro_step_phaseC(
    net: nn.Module,
    rv: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_present: torch.Tensor,
    prop_coords: torch.Tensor,
    prop_mask: torch.Tensor,
    prop_labels: torch.Tensor,
    labels_base: torch.Tensor,
    args: argparse.Namespace,
    buffer: StaticDataBuffer,
    use_greedy: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix three sources: GT+noise, synthetic drop/ghost, real proposals."""
    device = rv.device
    B = gt_coords.size(0)

    # Base synthetic from GT with drop/ghost
    x, keep_identity = buffer.generate(
        gt_coords,
        gt_mask,
        float(args.shift_sigma),
        float(args.point_sigma),
        float(args.drop_frac),
        int(args.ghosts),
    )

    # Ratios
    probs = torch.tensor(
        [
            float(args.ratio_gt),
            float(args.ratio_dropghost),
            float(args.ratio_prop),
        ],
        device=device,
        dtype=torch.float32,
    )
    if float(probs.sum()) <= 0:
        probs = torch.tensor([1.0, 0.0, 0.0], device=device)
    probs = probs / probs.sum()
    modes = torch.multinomial(probs, num_samples=B, replacement=True)  # 0:gt+noise,1:drop/ghost,2:proposal

    batch_tgt = {'c': [], 'm': [], 'p': [], 'l': [], 'prior': []}

    for b in range(B):
        mode = int(modes[b].item())
        if mode == 0:
            # GT + light noise
            x[b].copy_(gt_coords[b])
            sigma = float(getattr(args, 'gt_noise_sigma', args.point_sigma))
            if sigma > 0:
                x[b].add_(torch.randn_like(x[b]) * sigma).clamp_(-1.0, 1.0)
            keep_b = (~gt_mask[b]).any(dim=1)
            tgt_c = gt_coords[b]
            tgt_m = gt_mask[b]
            tgt_p = gt_present[b]
            labels_slot = labels_base.clone()
            input_prior_b = labels_slot.clone()
            input_prior_b[~keep_b] = -1
        elif mode == 2:
            # Real aggregated proposal
            x[b].copy_(prop_coords[b])
            keep_b = (~prop_mask[b]).any(dim=1)
            tgt_c, tgt_m, tgt_p, match_idx = build_fixed_targets(
                x[b], gt_coords[b], gt_mask[b], gt_present[b], keep_b, use_greedy
            )
            labels_slot = torch.full((buffer.N,), -1, dtype=torch.long, device=device)
            valid = (match_idx >= 0) & (match_idx < buffer.N)
            if valid.any():
                labels_slot[valid] = labels_base[match_idx[valid]]
            input_prior_b = labels_slot.clone()
            input_prior_b[~keep_b] = -1
        else:
            # Synthetic drop/ghost (already in x)
            keep_b = keep_identity[b]
            tgt_c, tgt_m, tgt_p, match_idx = build_fixed_targets(
                x[b], gt_coords[b], gt_mask[b], gt_present[b], keep_b, use_greedy
            )
            labels_slot = torch.full((buffer.N,), -1, dtype=torch.long, device=device)
            valid = (match_idx >= 0) & (match_idx < buffer.N)
            if valid.any():
                labels_slot[valid] = labels_base[match_idx[valid]]
            input_prior_b = labels_slot.clone()
            input_prior_b[~keep_b] = -1

        batch_tgt['c'].append(tgt_c)
        batch_tgt['m'].append(tgt_m)
        batch_tgt['p'].append(tgt_p)
        batch_tgt['l'].append(labels_slot)
        batch_tgt['prior'].append(input_prior_b)

    tgt_coords = torch.stack(batch_tgt['c'])
    tgt_mask_o = torch.stack(batch_tgt['m'])
    tgt_present_o = torch.stack(batch_tgt['p'])
    tgt_labels = torch.stack(batch_tgt['l'])
    input_prior = torch.stack(batch_tgt['prior'])

    # Diffusion
    buffer.diff_noise.normal_()
    buffer.xK.copy_(x).mul_(1.0 - float(args.alpha)).add_(buffer.diff_noise, alpha=float(args.alpha)).clamp_(-1, 1)
    sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)

    pred_coords, pred_logits, preds, _ = edm_unrolled_train(
        net,
        buffer.xK,
        rv,
        sigmas,
        second_order=bool(args.second_order),
        cond_prior=x,
        input_labels=input_prior,
        collect_states=False,
        collect_preds=(float(args.step_loss_weight) > 0.0),
    )

    step_loss = torch.tensor(0.0, device=device)
    if preds and float(args.step_loss_weight) > 0.0:
        losses = []
        for item in preds:
            pc, pl = item[0], item[1]
            sem = item[2] if len(item) > 2 else None
            out = criterion(
                pc,
                pl,
                tgt_coords,
                tgt_mask_o,
                tgt_present_o,
                l1_weight=float(args.l1_weight),
                cls_weight=float(args.cls_weight),
                use_focal=bool(args.use_focal),
                focal_alpha=float(args.focal_alpha),
                focal_gamma=float(args.focal_gamma),
                pred_sem_logits=sem,
                tgt_sem_labels=tgt_labels,
                sem_weight=float(args.sem_weight),
            )
            losses.append(out['loss_cls'] + out['loss_reg'] + out.get('loss_sem', 0.0))
        step_loss = torch.stack(losses).mean()

    final_sem = preds[-1][2] if (preds and len(preds[-1]) > 2) else None
    out_final = criterion(
        pred_coords,
        pred_logits,
        tgt_coords,
        tgt_mask_o,
        tgt_present_o,
        l1_weight=float(args.l1_weight),
        cls_weight=float(args.cls_weight),
        use_focal=bool(args.use_focal),
        focal_alpha=float(args.focal_alpha),
        focal_gamma=float(args.focal_gamma),
        pred_sem_logits=final_sem,
        tgt_sem_labels=tgt_labels,
        sem_weight=float(args.sem_weight),
    )
    final_loss = out_final['loss_cls'] + out_final['loss_reg'] + out_final.get('loss_sem', 0.0)
    total_loss = float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss
    return total_loss, step_loss, final_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--scenes', nargs='+', default=None,
                    help='Optional explicit scene list; overridden by --scenes-file when provided')
    ap.add_argument('--scenes-file', type=str, default='',
                    help='Optional text file with one scene id per line (takes precedence over --scenes)')
    ap.add_argument('--stats-json', default='global_diffusion_map/refine/work_dirs/av2_stats.json')
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_10scenes_opt')
    ap.add_argument('--agg-pred-root', type=str, default='', help='Aggregated proposal root for training')
    # Optimizations
    ap.add_argument('--img-size', type=int, default=512)
    ap.add_argument('--encoder', choices=['resnet', 'maptr', 'raster'], default='resnet')
    ap.add_argument('--resnet', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18')
    ap.add_argument('--imagenet-pretrained', action='store_true', default=True)
    # Train
    ap.add_argument('--epochs', type=int, default=2400)
    ap.add_argument('--phase-a-epochs', type=int, default=2000)
    ap.add_argument('--training-mode', choices=['phaseAB', 'phaseC'], default='phaseC',
                    help='phaseAB: original A/B curriculum; phaseC: mix gt+noise/drop+ghost/proposal (recommended for full data)')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--save-every', type=int, default=100)
    ap.add_argument('--resume-ckpt', type=str, default='', help='Path to checkpoint with {net, encoder}')
    # EDM / loss
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--alpha', type=float, default=0.03)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--matcher', choices=['greedy', 'hungarian'], default='greedy')
    # Corruptions
    ap.add_argument('--shift-sigma', type=float, default=0.10)
    ap.add_argument('--point-sigma', type=float, default=0.02)
    ap.add_argument('--drop-frac', type=float, default=0.15)
    ap.add_argument('--ghosts', type=int, default=2)
    ap.add_argument('--gt-noise-sigma', type=float, default=0.02, help='Std for gt+noise branch in Phase-C')
    ap.add_argument('--ratio-gt', type=float, default=0.20, help='Phase-C mix ratio: gt+noise (anchor)')
    ap.add_argument('--ratio-dropghost', type=float, default=0.10, help='Phase-C mix ratio: synthetic drop/ghost')
    ap.add_argument('--ratio-prop', type=float, default=0.70, help='Phase-C mix ratio: real proposals (main)')
    # Weights
    ap.add_argument('--l1-weight', type=float, default=20.0)
    ap.add_argument('--cls-weight', type=float, default=10.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    ap.add_argument('--step-loss-weight', type=float, default=1.0)
    ap.add_argument('--final-loss-weight', type=float, default=1.0)
    ap.add_argument('--use-focal', action='store_true')
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    # Multipliers
    ap.add_argument('--prior-lr-mult', type=float, default=2.0)
    ap.add_argument('--enc-lr-mult', type=float, default=1.0)
    # Legacy maptr args (no-op for resnet/raster)
    ap.add_argument('--polydiff-cfg', default='')
    ap.add_argument('--pretrained-maptr-ckpt', default='')
    ap.add_argument('--freeze-encoder', action='store_true')

    args = ap.parse_args()
    set_seed(0)

    ddp = is_ddp()
    local_rank = 0
    device = torch.device('cuda')
    if ddp:
        local_rank = ddp_setup()
        device = torch.device(f'cuda:{local_rank}')

    # Dataset / caps
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    RefineCaps(num_points=P, num_queries=N)

    # Resolve scene list: scenes-file > scenes > all pkls under static-root
    import glob
    if args.scenes_file:
        with open(args.scenes_file, 'r') as f:
            scenes = [ln.strip() for ln in f if ln.strip()]
    elif args.scenes:
        scenes = list(args.scenes)
    else:
        patt = glob.glob(osp.join(args.static_root, '*.pkl'))
        scenes = [osp.splitext(osp.basename(p))[0] for p in patt]
    scenes = sorted(set(scenes))

    agg_root = args.agg_pred_root if args.agg_pred_root else None
    ds = TenScenesDataset(args.static_root, args.rendered_root, scenes, args.stats_json,
                          img_size=args.img_size, agg_pred_root=agg_root)
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if ddp else None
    loader = DataLoader(ds, batch_size=int(args.batch_size), sampler=sampler, shuffle=(sampler is None),
                        num_workers=int(args.workers), pin_memory=True, drop_last=True)

    # Models
    if args.encoder == 'resnet':
        enc = StandardResNetEncoder(out_dim=256, version=args.resnet, pretrained=bool(args.imagenet_pretrained)).to(device)
    elif args.encoder == 'raster':
        enc = RasterEncoder(out_dim=256).to(device)
    else:
        from global_diffusion_map.refine.single_scene_overfit import PolyDiffuseImageEncoder256
        enc = PolyDiffuseImageEncoder256(args.polydiff_cfg, args.pretrained_maptr_ckpt, device=str(device))

    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    if ddp:
        base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)

    # Param groups (build optimizer before optional state restore)
    prior_ids = {id(p) for p in net.backbone.prior_mlp.parameters()}
    prior_ids.add(id(net.backbone.prior_gate))
    if getattr(net.backbone, 'class_emb', None):
        for p in net.backbone.class_emb.parameters():
            prior_ids.add(id(p))
        prior_ids.add(id(net.backbone.class_gate))
    main_params = [p for p in net.parameters() if id(p) not in prior_ids]
    prior_params_list = [p for p in net.parameters() if id(p) in prior_ids]
    param_groups = [
        {'params': enc.parameters(), 'lr': float(args.lr) * float(args.enc_lr_mult), 'weight_decay': 1e-4},
        {'params': main_params, 'lr': float(args.lr), 'weight_decay': 1e-4},
        {'params': prior_params_list, 'lr': float(args.lr) * float(args.prior_lr_mult), 'weight_decay': 1e-4},
    ]

    if ddp:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
        enc = torch.nn.parallel.DistributedDataParallel(enc, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(param_groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(args.epochs), eta_min=float(args.lr_min))

    # Resume from checkpoint (load encoder + net + optimizer/scheduler/epoch)
    start_epoch = 1
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        if 'encoder' in ckpt:
            try:
                (enc.module if isinstance(enc, torch.nn.parallel.DistributedDataParallel) else enc).load_state_dict(
                    ckpt['encoder'], strict=True
                )
                print(f"[resume] encoder loaded from {args.resume_ckpt}")
            except Exception as e:
                print(f"[resume] encoder load (strict) failed: {e}; trying non-strict")
                (enc.module if isinstance(enc, torch.nn.parallel.DistributedDataParallel) else enc).load_state_dict(
                    ckpt['encoder'], strict=False
                )
        if 'net' in ckpt:
            try:
                (net.module if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net).load_state_dict(
                    ckpt['net'], strict=True
                )
                print(f"[resume] net(full) loaded from {args.resume_ckpt}")
            except Exception:
                # Support backbone-only checkpoints for legacy runs
                try:
                    (net.module.backbone if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net.backbone).load_state_dict(
                        ckpt['net'], strict=False
                    )
                    print(f"[resume] net.backbone loaded (non-strict) from {args.resume_ckpt}")
                except Exception as e:
                    print(f"[resume] net load failed: {e}")
        if 'optimizer' in ckpt:
            try:
                opt.load_state_dict(ckpt['optimizer'])
                print(f"[resume] optimizer state loaded")
            except Exception as e:
                print(f"[resume] optimizer load failed: {e}")
        if 'scheduler' in ckpt:
            try:
                sched.load_state_dict(ckpt['scheduler'])
                print(f"[resume] scheduler state loaded")
            except Exception as e:
                print(f"[resume] scheduler load failed: {e}")
        if 'epoch' in ckpt:
            try:
                start_epoch = int(ckpt['epoch']) + 1
                print(f"[resume] starting from epoch {start_epoch}")
            except Exception:
                start_epoch = 1

    enc.train()  # Always train encoder for raster inputs
    scaler = torch.cuda.amp.GradScaler()

    labels_base = torch.from_numpy(_labels_from_budgets(budgets, N)).long().to(device)
    buffer = StaticDataBuffer(int(args.batch_size), N, P, device)

    os.makedirs(args.out_root, exist_ok=True)

    # Training loop
    for ep in range(start_epoch, int(args.epochs) + 1):
        if ddp and sampler is not None:
            sampler.set_epoch(ep)
        net.train(); enc.train()
        l_sum = 0.0; s_sum = 0.0; f_sum = 0.0; n_batches = 0
        is_phase_a = ep <= int(args.phase_a_epochs)

        for batch in loader:
            ras = batch['ras'].to(device, non_blocking=True)
            gt_coords = batch['gt_pack'].to(device, non_blocking=True)
            gt_mask = batch['gt_mask'].to(device, non_blocking=True)
            gt_present = batch['gt_present'].to(device, non_blocking=True)
            prop_coords = batch['prop_pack'].to(device, non_blocking=True) if 'prop_pack' in batch else None
            prop_mask = batch['prop_mask'].to(device, non_blocking=True) if 'prop_mask' in batch else None
            prop_labels = batch['prop_labels'].to(device, non_blocking=True) if 'prop_labels' in batch else None

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                rv = enc(ras)
                if args.training_mode == 'phaseC':
                    if prop_coords is None or prop_mask is None:
                        raise RuntimeError('Phase-C requires aggregated proposals; provide --agg-pred-root')
                    loss, s_loss, f_loss = run_micro_step_phaseC(
                        net,
                        rv,
                        gt_coords,
                        gt_mask,
                        gt_present,
                        prop_coords,
                        prop_mask,
                        prop_labels if prop_labels is not None else labels_base.unsqueeze(0).expand(gt_coords.size(0), -1),
                        labels_base,
                        args,
                        buffer,
                        (args.matcher == 'greedy'),
                    )
                else:
                    loss, s_loss, f_loss = run_micro_step_fast(
                        net, rv, gt_coords, gt_mask, gt_present, labels_base, args, buffer, is_phase_a, (args.matcher == 'greedy')
                    )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            l_sum += float(loss.item()); s_sum += float(s_loss.item()); f_sum += float(f_loss.item())
            n_batches += 1

        sched.step()

        if (not ddp) or dist.get_rank() == 0:
            if n_batches > 0 and (ep % int(args.log_every) == 0 or ep == 1):
                if args.training_mode == 'phaseAB':
                    phase_tag = 'A' if is_phase_a else 'B'
                else:
                    phase_tag = 'C'
                print(f"[ep {ep:04d}] P={phase_tag} loss={l_sum/n_batches:.4f} (step={s_sum/n_batches:.4f} final={f_sum/n_batches:.4f})")
            if ep % int(args.save_every) == 0:
                ckpt_path = osp.join(args.out_root, f'ckpt_ep_{ep:04d}.pth')
                n_st = net.module.state_dict() if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net.state_dict()
                e_st = enc.module.state_dict() if isinstance(enc, torch.nn.parallel.DistributedDataParallel) else enc.state_dict()
                state = {
                    'net': n_st,
                    'encoder': e_st,
                    'optimizer': opt.state_dict(),
                    'scheduler': sched.state_dict(),
                    'epoch': ep,
                }
                torch.save(state, ckpt_path)
                # Convenience: encoder-only file
                torch.save(e_st, osp.join(args.out_root, f'encoder_ep_{ep:04d}.pth'))

    if ddp:
        ddp_cleanup()


if __name__ == '__main__':
    main()
