#!/usr/bin/env python
from __future__ import annotations

"""
10-Scene Mini-Batch Refine Training (Stable Matching, DDP, Phase-A curriculum).

Key features
- Based on the stable-matching trainer (greedy/Hungarian, step+final losses).
- DDP-ready (torchrun) with DistributedSampler and rank-0 checkpointing.
- Phase-A curriculum: easy mode (no drops/ghosts) for many epochs (default 2000),
  then switch to hard mode (full jitter + drops + ghosts).
- Saves checkpoint every 100 epochs by default.

Example (4 GPUs 0/1/2/3)
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    global_diffusion_map/refine/clean/train_mini_batch_10scenes_stable_ddp.py \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --scenes S1 S2 S3 S4 S5 S6 S7 S8 S9 S10 \
    --epochs 2400 --phase-a-epochs 2000 --batch-size 1 --save-every 100 \
    --polydiff-cfg official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py \
    --pretrained-maptr-ckpt global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth
"""

import os
import os.path as osp
import json
import argparse
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Reduce fragmentation; keep benchmark off to avoid algo-cache churn
# Keep allocator config minimal for widest compatibility
# Newer keys like garbage_collection_threshold may not be supported on this build.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
import sys
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.loss_refine import criterion, hungarian_match_perm, gpu_greedy_match
from global_diffusion_map.refine.single_scene_overfit import load_pickle, RasterEncoder
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train


def set_seed(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_ddp() -> bool:
    return int(os.environ.get('WORLD_SIZE', '1')) > 1


def ddp_setup() -> int:
    # Expect LOCAL_RANK provided by torchrun
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class TenScenesDataset(Dataset):
    """Preloads 10 scenes (small) into RAM for fast iteration."""
    def __init__(self, static_root: str, rendered_root: str, scenes: Sequence[str], stats_json: str) -> None:
        super().__init__()
        self.static_root = static_root
        self.rendered_root = rendered_root
        self.scenes = list(scenes)
        with open(stats_json, 'r') as f:
            stats = json.load(f)
        self.P = int(stats.get('M', 20))
        self.N = int(stats.get('num_queries', 64))
        self.budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}

        from PIL import Image
        import torchvision.transforms.functional as TF

        self.bank: List[Dict[str, object]] = []
        for sid in self.scenes:
            gt = load_pickle(osp.join(self.static_root, f'{sid}.pkl'))
            bounds = gt.get('bounds')
            if bounds is None:
                raise RuntimeError(f'static GT lacks bounds: {sid}')
            gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, self.budgets, num_points=self.P, num_queries=self.N)
            raw = Image.open(osp.join(self.rendered_root, sid, '10_render_gt.png')).convert('RGB')
            w, h = raw.size
            nw, nh = max(32, (w // 32) * 32), max(32, (h // 32) * 32)
            if nw != w or nh != h:
                raw = raw.resize((nw, nh), Image.BILINEAR)
            img_t = TF.normalize(TF.to_tensor(raw), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.bank.append({
                'scene': sid,
                'ras': img_t,                                 # [3,H,W] float
                'gt_pack': torch.from_numpy(gt_pack).float(),  # [N,P,2]
                'gt_mask': torch.from_numpy(gt_mask).bool(),   # [N,P]
                'gt_present': torch.from_numpy(gt_present).long(),  # [N]
            })

    def __len__(self) -> int:
        return len(self.bank)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.bank[idx]


class StaticDataBuffer:
    """Static buffers sized to per-GPU batch for jitter/drop/ghost + EDM xK/noise."""
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
    def generate(self,
                 gt_coords_batch: torch.Tensor,  # [B,N,P,2]
                 gt_mask_batch: torch.Tensor,    # [B,N,P]
                 shift_sigma: float,
                 point_sigma: float,
                 drop_frac: float,
                 ghosts: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.x_buf.copy_(gt_coords_batch)
        self.mask_buf.copy_(gt_mask_batch)
        valid = ~self.mask_buf
        # jitter (shift + point)
        self.shifts.normal_().mul_(shift_sigma)
        self.local.normal_().mul_(point_sigma)
        noise_view = self.local + self.shifts
        x_valid = self.x_buf[valid]
        x_valid.add_(noise_view[valid]).clamp_(-1.0, 1.0)
        self.x_buf[valid] = x_valid
        # drops
        present = valid.any(dim=2)
        self.rnd_drop.uniform_()
        is_drop = present & (self.rnd_drop < drop_frac)
        keep_identity = present & (~is_drop)
        if is_drop.any():
            self.local.normal_().mul_(max(shift_sigma, point_sigma) * 0.8)
            self.x_buf[is_drop] = (gt_coords_batch[is_drop] + self.local[is_drop]).clamp_(-1.0, 1.0)
        # ghosts
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
            if use_greedy:
                x_free = x_in
                present_rem = torch.ones((gt_sel_idx.numel(),), dtype=torch.bool, device=x_in.device)
                pairs = gpu_greedy_match(x_free, gt_sel_coords, gt_sel_mask, present_rem,
                                         w_center=1.0, w_dir=0.2, w_pw=0.5, use_chamfer=True,
                                         max_center_dist=None)
                assigned_slots = set(keep_ids.tolist())
                for (pi_free, gj_loc) in pairs:
                    pi_i = int(pi_free)
                    if pi_i in assigned_slots:
                        continue
                    gj_global = int(gt_sel_idx[int(gj_loc)].item())
                    if not bool(remaining_gt[gj_global]):
                        continue
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
                assigned_slots = set(keep_ids.tolist())
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
    for orig in (1, 0, 2):  # MapTR ids: divider=1,ped=0,boundary=2 → labels 0,1,2
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


class StandardResNetEncoder(nn.Module):
    """Simple ResNet encoder for BEV raster (ImageNet weights optional)."""
    def __init__(self, out_dim: int = 256, version: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        def build_resnet(name: str):
            # Handle both new (weights=...) and old (pretrained=...) torchvision APIs
            try:
                if name == 'resnet18':
                    return models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None), 512
                if name == 'resnet34':
                    return models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None), 512
                if name == 'resnet50':
                    return models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None), 2048
            except Exception:
                if name == 'resnet18':
                    return models.resnet18(pretrained=pretrained), 512
                if name == 'resnet34':
                    return models.resnet34(pretrained=pretrained), 512
                if name == 'resnet50':
                    return models.resnet50(pretrained=pretrained), 2048
            raise ValueError(f'unknown resnet: {name}')

        if version not in ('resnet18', 'resnet34', 'resnet50'):
            raise ValueError(f"Unknown resnet version: {version}")
        backbone, feat_dim = build_resnet(version)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(feat_dim, out_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)        # [B,C,1,1]
        feat = feat.view(feat.size(0), -1)
        return self.proj(feat)


def run_micro_step(
    net: nn.Module,
    rv: torch.Tensor,              # [B,256]
    scaler: torch.cuda.amp.GradScaler,
    gt_coords: torch.Tensor,       # [B,N,P,2]
    gt_mask: torch.Tensor,         # [B,N,P]
    gt_present: torch.Tensor,      # [B,N]
    labels_base: torch.Tensor,     # [N]
    args: argparse.Namespace,
    buffer: StaticDataBuffer,
    is_phase_a: bool,
    use_greedy: bool,
) -> Tuple[float, float, float]:
    device = rv.device
    B, N, P = gt_coords.size(0), gt_coords.size(1), gt_coords.size(2)

    # Phase control
    if is_phase_a:
        s_sigma = float(args.shift_sigma)
        p_sigma = float(args.point_sigma)
        drop_f = 0.0
        n_ghosts = 0
    else:
        s_sigma = float(args.shift_sigma)
        p_sigma = float(args.point_sigma)
        drop_f = float(args.drop_frac)
        n_ghosts = int(args.ghosts)

    x, keep_identity = buffer.generate(gt_coords, gt_mask, s_sigma, p_sigma, drop_f, n_ghosts)

    # Build targets per sample
    batch_tgt = {'c': [], 'm': [], 'p': [], 'l': [], 'prior': []}
    for b in range(B):
        keep_b = keep_identity[b]
        tgt_c, tgt_m, tgt_p, match_idx = build_fixed_targets(
            x[b], gt_coords[b], gt_mask[b], gt_present[b], keep_b, use_greedy=use_greedy
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

    # Diffusion
    buffer.diff_noise.normal_()
    buffer.xK.copy_(x).mul_(1.0 - float(args.alpha)).add_(buffer.diff_noise, alpha=float(args.alpha)).clamp_(-1, 1)
    sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)

    with torch.cuda.amp.autocast():
        pred_coords, pred_logits, preds, _ = edm_unrolled_train(
            net, buffer.xK, rv, sigmas,
            second_order=bool(args.second_order),
            cond_prior=x, input_labels=input_prior,
            collect_states=False, collect_preds=(float(args.step_loss_weight) > 0.0)
        )
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

    # Backward on denoiser graph (encoder split-backward handled in caller)
    scaler.scale(total_loss).backward()

    l_val = float(total_loss.item())
    s_val = float(step_loss.item())
    f_val = float(final_loss.item())
    del pred_coords, pred_logits, preds, total_loss, step_loss, final_loss, tgt_coords, input_prior
    return l_val, s_val, f_val


def main() -> None:
    ap = argparse.ArgumentParser(description='10-scene mini-batch refine training (stable match, DDP)')
    # Data
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/refine/work_dirs/av2_stats.json')
    # Model & encoder
    ap.add_argument('--polydiff-cfg', default='official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py')
    ap.add_argument('--pretrained-maptr-ckpt', default='global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth')
    # Freeze encoder by default; allow overriding with --no-freeze-encoder
    ap.add_argument('--freeze-encoder', action='store_true', default=True)
    ap.add_argument('--no-freeze-encoder', dest='freeze_encoder', action='store_false')
    ap.add_argument('--encoder', choices=['resnet', 'maptr', 'raster'], default='resnet')
    ap.add_argument('--resnet', choices=['resnet18', 'resnet34', 'resnet50'], default='resnet18')
    ap.add_argument('--imagenet-pretrained', dest='imagenet_pretrained', action='store_true')
    ap.add_argument('--no-imagenet', dest='imagenet_pretrained', action='store_false')
    ap.set_defaults(imagenet_pretrained=True)
    # Train
    ap.add_argument('--epochs', type=int, default=2400)
    ap.add_argument('--phase-a-epochs', type=int, default=2000, help='Easy phase epochs (no drops/ghosts)')
    ap.add_argument('--batch-size', type=int, default=1, help='per-GPU scenes per batch')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--prior-lr-mult', type=float, default=2.0)
    ap.add_argument('--enc-lr-mult', type=float, default=1.0, help='encoder LR multiplier relative to --lr')
    ap.add_argument('--sched', choices=['cosine', 'none'], default='cosine')
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--log-every', type=int, default=10)
    ap.add_argument('--save-every', type=int, default=100)
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_10scenes_stable_ddp')
    # EDM + corruption
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--alpha', type=float, default=0.03)
    ap.add_argument('--second-order', action='store_true')
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

    args = ap.parse_args()
    set_seed(0)

    # DDP init
    ddp = is_ddp()
    local_rank = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ddp:
        local_rank = ddp_setup()
        device = torch.device(f'cuda:{local_rank}')

    # Stats/caps
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    RefineCaps(num_points=P, num_queries=N)

    # Dataset/Loader
    ds = TenScenesDataset(args.static_root, args.rendered_root, args.scenes, args.stats_json)
    # Tiny, fully preloaded dataset → avoid worker spawn overhead
    if len(ds) < 100:
        args.workers = 0
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if ddp else None
    loader = DataLoader(ds, batch_size=int(args.batch_size), sampler=sampler, shuffle=(sampler is None),
                        num_workers=int(args.workers), pin_memory=True, drop_last=True)

    # Models
    if args.encoder == 'maptr':
        enc = PolyDiffuseImageEncoder256(args.polydiff_cfg, args.pretrained_maptr_ckpt, device=str(device))
    elif args.encoder == 'raster':
        enc = RasterEncoder(out_dim=256).to(device)
    else:
        enc = StandardResNetEncoder(out_dim=256, version=args.resnet, pretrained=bool(args.imagenet_pretrained)).to(device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)

    # Param groups (prior boosted LR)
    prior_params: List[torch.nn.Parameter] = []
    prior_params += list(net.backbone.prior_mlp.parameters())
    prior_params += [net.backbone.prior_gate]
    if getattr(net.backbone, 'class_emb', None) is not None:
        prior_params += list(net.backbone.class_emb.parameters())
        prior_params += [net.backbone.class_gate]
    prior_ids = {id(p) for p in prior_params}
    main_params = [p for p in net.parameters() if id(p) not in prior_ids]
    enc_params: List[torch.nn.Parameter] = []
    if not args.freeze_encoder:
        enc_params = list(enc.parameters())

    param_groups = []
    if enc_params:
        param_groups.append({'params': enc_params, 'lr': float(args.lr) * float(args.enc_lr_mult), 'weight_decay': 1e-4})
    param_groups.append({'params': main_params, 'lr': float(args.lr), 'weight_decay': 1e-4})
    param_groups.append({'params': prior_params, 'lr': float(args.lr) * float(args.prior_lr_mult), 'weight_decay': 1e-4})

    # Wrap for DDP
    if ddp:
        # Small per-GPU batch → SyncBN if present
        base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
        net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
        if not args.freeze_encoder:
            enc = torch.nn.parallel.DistributedDataParallel(enc, device_ids=[local_rank], output_device=local_rank)

    # Optim / sched / scaler
    opt = torch.optim.AdamW(param_groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(args.epochs), eta_min=float(args.lr_min)) if args.sched == 'cosine' else None
    scaler = torch.cuda.amp.GradScaler()

    labels_base = torch.from_numpy(_labels_from_budgets(budgets, N)).long().to(device)
    buffer = StaticDataBuffer(int(args.batch_size), N, P, device)

    # Freeze encoder params if requested
    if args.freeze_encoder:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
    else:
        enc.train()

    # IO
    os.makedirs(args.out_root, exist_ok=True)

    def save_ckpt(ep: int) -> None:
        if ddp and dist.get_rank() != 0:
            return
        ckpt_path = osp.join(args.out_root, f'ckpt_ep_{ep:04d}.pth')
        state = {}
        # Unwrap DDP modules if needed
        net_state = net.module.backbone.state_dict() if isinstance(net, torch.nn.parallel.DistributedDataParallel) else net.backbone.state_dict()
        state['net'] = net_state
        # Always save encoder (frozen or not) for reproducibility
        enc_state = enc.module.state_dict() if isinstance(enc, torch.nn.parallel.DistributedDataParallel) else enc.state_dict()
        state['encoder'] = enc_state
        torch.save(state, ckpt_path)
        # Convenience: write an encoder-only checkpoint alongside
        enc_only_path = osp.join(args.out_root, f'encoder_ep_{ep:04d}.pth')
        torch.save(enc_state, enc_only_path)

    # Training loop
    for ep in range(1, int(args.epochs) + 1):
        if ddp and sampler is not None:
            sampler.set_epoch(ep)

        if not args.freeze_encoder:
            enc.train()
        net.train()

        if ddp and dist.get_rank() == 0:
            pass

        l_sum = 0.0
        s_sum = 0.0
        f_sum = 0.0
        n_batches = 0

        is_phase_a = ep <= int(args.phase_a_epochs)
        use_greedy = (args.matcher == 'greedy')

        for batch in loader:
            ras = batch['ras'].to(device, non_blocking=True)
            gt_coords = batch['gt_pack'].to(device, non_blocking=True)
            gt_mask = batch['gt_mask'].to(device, non_blocking=True)
            gt_present = batch['gt_present'].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            enc_ctx = torch.no_grad() if args.freeze_encoder else torch.enable_grad()
            with torch.cuda.amp.autocast(), enc_ctx:
                rv_single = enc(ras)  # [B,256]

            # Split-Backward: DETACH encoder output for denoiser graph
            if not args.freeze_encoder:
                rv_use = rv_single.detach()
                rv_use.requires_grad_(True)
            else:
                rv_use = rv_single

            # IMPORTANT: pass DDP wrapper directly so .backward triggers allreduce
            ls, ss, fs = run_micro_step(
                net,
                rv_use,
                scaler,
                gt_coords,
                gt_mask,
                gt_present,
                labels_base,
                args,
                buffer,
                is_phase_a,
                use_greedy,
            )

            if not args.freeze_encoder and rv_use.grad is not None:
                # Bridge gradient from detached rv_use back to encoder output rv_single
                scaler.scale(rv_single).backward(rv_use.grad)

            scaler.step(opt)
            scaler.update()

            l_sum += ls; s_sum += ss; f_sum += fs
            n_batches += 1

        if sched is not None:
            sched.step()

        # Reduce metrics across ranks for logging
        if ddp:
            t = torch.tensor([l_sum, s_sum, f_sum, n_batches], dtype=torch.float32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            l_sum, s_sum, f_sum, n_batches = [float(x) for x in t.tolist()]

        if (not ddp) or dist.get_rank() == 0:
            if n_batches > 0 and (ep % int(args.log_every) == 0 or ep == 1):
                print(f"[ep {ep:04d}] phase={'A' if is_phase_a else 'B'} freeze_enc={args.freeze_encoder} loss={l_sum/max(1,n_batches):.4f} step={s_sum/max(1,n_batches):.4f} final={f_sum/max(1,n_batches):.4f}")
            if ep % int(args.save_every) == 0:
                save_ckpt(ep)

    # Final save
    if (not ddp) or dist.get_rank() == 0:
        save_ckpt(int(args.epochs))

    if ddp:
        ddp_cleanup()


if __name__ == '__main__':
    main()
