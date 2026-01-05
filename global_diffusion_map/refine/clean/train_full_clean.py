#!/usr/bin/env python
from __future__ import annotations

"""
Clean multi-scene refine training (EDM, DDP) — GT+noise hybrid.

Aligned with train_one_scene_clean.py, but supports:
- Multiple scenes (train/val lists)
- Distributed Data Parallel (DDP) training
- Cosine LR by default; prior branch LR multiplier

Batch items are independent scenes. Each sample is built by jitter/drop/ghost
around the per-scene GT pack; targets are fixed via identity+GPU greedy match.
"""

import argparse
import json
import os
import os.path as osp
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.single_scene_overfit import RasterEncoder, load_pickle
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train
from global_diffusion_map.refine.loss_refine import criterion, gpu_greedy_match

# Reuse jitter/matching utilities from one-scene clean script to stay aligned
from global_diffusion_map.refine.clean.train_one_scene_clean import (
    build_fixed_targets,
    jitter_drop_ghost,
)


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def gpu_augment_batch(
    gt_coords: torch.Tensor,  # [B,N,P,2]
    gt_mask: torch.Tensor,    # [B,N,P]
    shift_sigma: float = 0.10,
    point_sigma: float = 0.02,
    drop_frac: float = 0.15,
    ghosts: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """On-GPU augmentation: structured jitter + drop + ghost.

    Returns:
      x_b:          augmented proposals [B,N,P,2]
      keep_identity: mask [B,N] — True for identity-kept GT slots
    """
    B, N, P, _ = gt_coords.shape
    device = gt_coords.device
    # Start from clean GT as proposal base
    x_b = gt_coords.clone()
    # Structured jitter (global shift + local perturb)
    shifts = torch.randn(B, N, 1, 2, device=device, dtype=gt_coords.dtype) * float(shift_sigma)
    locals = torch.randn(B, N, P, 2, device=device, dtype=gt_coords.dtype) * float(point_sigma)
    noise = shifts + locals
    valid = (~gt_mask).unsqueeze(-1)  # [B,N,P,1]
    x_b = torch.where(valid, x_b + noise, x_b).clamp(-1.0, 1.0)
    # Present mask
    present = (~gt_mask).any(dim=2)  # [B,N]
    # Drop a fraction of present GT slots
    rand = torch.rand(B, N, device=device)
    is_drop = (rand < float(drop_frac)) & present
    keep_identity = present & (~is_drop)
    if is_drop.any():
        drop_noise = torch.randn_like(x_b) * max(float(shift_sigma), float(point_sigma)) * 2.0
        x_b = torch.where(is_drop.unsqueeze(-1).unsqueeze(-1), (gt_coords + drop_noise).clamp(-1.0, 1.0), x_b)
    # Add ghosts into empty slots with probability ghosts/N
    if int(ghosts) > 0:
        is_empty = ~present
        ghost_prob = float(ghosts) / max(1, int(N))
        add_ghost = (torch.rand(B, N, device=device) < ghost_prob) & is_empty
        if add_ghost.any():
            # random smooth polyline in [-1,1]^2
            gl = (torch.rand(B, N, P, 2, device=device, dtype=gt_coords.dtype) * 2.0 - 1.0) * 0.6
            # simple smoothing along sequence
            for k in range(1, P):
                gl[:, :, k, :] = 0.7 * gl[:, :, k, :] + 0.3 * gl[:, :, k - 1, :]
            x_b = torch.where(add_ghost.unsqueeze(-1).unsqueeze(-1), gl, x_b)
    return x_b, keep_identity


@torch.no_grad()
def build_fixed_targets_fast(
    x_in: torch.Tensor,         # [N,P,2]
    gt_coords: torch.Tensor,    # [N,P,2]
    gt_mask: torch.Tensor,      # [N,P]
    gt_present: torch.Tensor,   # [N]
    keep_identity_mask: torch.Tensor,  # [N]
    max_center_dist: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized fixed targets (identity + greedy match + flip) to avoid Python sync overhead.

    Notes:
      - Uses gpu_greedy_match for pairing, then applies distance gate and flip selection in vectorized form.
      - Avoids .item() calls and per-pair loops.
    """
    device = x_in.device
    N, P, _ = x_in.shape
    tgt_coords = torch.zeros_like(gt_coords)
    tgt_mask_o = torch.ones_like(gt_mask, dtype=torch.bool)
    tgt_present_o = torch.zeros_like(gt_present)
    match_idx = torch.full_like(gt_present, fill_value=-1, dtype=torch.long)

    keep_mask = (keep_identity_mask & (gt_present > 0))  # [N]
    # Identity mapping
    if bool(keep_mask.any()):
        km = keep_mask.view(N, 1, 1)
        tgt_coords = torch.where(km, gt_coords, tgt_coords)
        tgt_mask_o = torch.where(keep_mask.view(N, 1), gt_mask, tgt_mask_o)
        tgt_present_o = torch.where(keep_mask, torch.ones_like(tgt_present_o), tgt_present_o)
        ids = torch.arange(N, device=device, dtype=torch.long)
        match_idx = torch.where(keep_mask, ids, match_idx)

    remaining_gt = (gt_present > 0) & (~keep_mask)
    if bool(remaining_gt.any()):
        # Pairs from greedy matcher (list of tuples on device)
        pairs = gpu_greedy_match(x_in, gt_coords, gt_mask, gt_present)
        if len(pairs) > 0:
            pt = torch.as_tensor(pairs, device=device, dtype=torch.long)
            pi_all, gj_all = pt[:, 0], pt[:, 1]
            not_kept = ~keep_mask[pi_all]
            rem_ok = remaining_gt[gj_all]
            valid = not_kept & rem_ok
            if bool(valid.any()):
                pi_vec = pi_all[valid]
                gj_vec = gj_all[valid]
                # Distance gate by masked centers
                # masked mean over valid points
                v_pi = torch.ones_like(gt_mask[pi_vec], dtype=x_in.dtype) - gt_mask[pi_vec].to(x_in.dtype)
                v_gj = torch.ones_like(gt_mask[gj_vec], dtype=x_in.dtype) - gt_mask[gj_vec].to(x_in.dtype)
                # avoid zero div
                cnt_pi = v_pi.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
                cnt_gj = v_gj.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
                c_pi = (x_in[pi_vec] * v_pi.unsqueeze(-1)).sum(dim=1) / cnt_pi  # [M,2]
                c_gj = (gt_coords[gj_vec] * v_gj.unsqueeze(-1)).sum(dim=1) / cnt_gj
                d = torch.norm(c_pi - c_gj, dim=1)
                dm = d <= float(max_center_dist)
                if bool(dm.any()):
                    pi_vec = pi_vec[dm]
                    gj_vec = gj_vec[dm]
                    x_c = x_in[pi_vec]
                    g_c = gt_coords[gj_vec]
                    g_cf = torch.flip(g_c, dims=[1])
                    l1_fwd = (x_c - g_c).abs().mean(dim=(1, 2))
                    l1_rev = (x_c - g_cf).abs().mean(dim=(1, 2))
                    do_flip = (l1_rev < l1_fwd).view(-1, 1, 1)
                    gt_sel = torch.where(do_flip, g_cf, g_c)
                    gm = gt_mask[gj_vec]
                    gm_f = torch.flip(gm, dims=[1])
                    mask_sel = torch.where(do_flip.squeeze(-1), gm_f, gm)
                    tgt_coords[pi_vec] = gt_sel
                    tgt_mask_o[pi_vec] = mask_sel
                    tgt_present_o[pi_vec] = 1
                    match_idx[pi_vec] = gj_vec

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


class MultiSceneCleanDataset(Dataset):
    def __init__(
        self,
        static_root: str,
        rendered_root: str,
        scenes: Sequence[str],
        caps: RefineCaps,
        budgets: Dict[int, int],
        shift_sigma: float = 0.10,
        point_sigma: float = 0.02,
        drop_frac: float = 0.15,
        ghosts: int = 2,
        repeat_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.static_root = static_root
        self.rendered_root = rendered_root
        self.caps = caps
        self.scenes = list(scenes)
        self.budgets = {int(k): int(v) for k, v in budgets.items()}
        self.shift_sigma = float(shift_sigma)
        self.point_sigma = float(point_sigma)
        self.drop_frac = float(drop_frac)
        self.ghosts = int(ghosts)
        self.repeat_per_epoch = max(1, int(repeat_per_epoch))

        # Preload GT packs and rasters to memory for speed
        from PIL import Image
        self.items: List[Dict[str, np.ndarray]] = []
        for scene in self.scenes:
            gt = load_pickle(osp.join(static_root, f'{scene}.pkl'))
            bounds = gt.get('bounds')
            if bounds is None:
                continue
            gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, self.budgets,
                                                            num_points=self.caps.num_points,
                                                            num_queries=self.caps.num_queries)
            # Load raster and resize to a fixed canvas (1024x1024) for batching
            img = Image.open(osp.join(rendered_root, scene, '10_render_gt.png')).convert('RGB')
            # Use faster BILINEAR for small test set to reduce CPU overhead
            img = img.resize((1024, 1024), Image.Resampling.BILINEAR)
            ras = np.asarray(img, dtype=np.float32) / 255.0
            ras = ras.transpose(2, 0, 1)  # [3,H,W]
            self.items.append({'scene': scene, 'bounds': np.asarray(bounds, dtype=np.float32),
                               'gt_pack': gt_pack, 'gt_mask': gt_mask, 'gt_present': gt_present, 'ras': ras})
        if len(self.items) == 0:
            raise RuntimeError('no valid scenes with bounds were found')

    def __len__(self) -> int:
        # Virtually repeat the small scene set to satisfy large batch / DDP and avoid empty batches with drop_last=True
        return len(self.items) * self.repeat_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # CPU-offload path: build proposals + fixed targets + prior/target labels here.
        j = idx % len(self.items)
        it = self.items[j]
        gt_pack = it['gt_pack']
        gt_mask = it['gt_mask']
        gt_present = it['gt_present']
        # 1) CPU jitter/drop/ghost via numpy
        x_np, keep_identity_np, _is_drop_np, _is_ghost_np = jitter_drop_ghost(
            gt_pack, gt_mask,
            shift_sigma=self.shift_sigma, point_sigma=self.point_sigma,
            drop_frac=self.drop_frac, ghosts=self.ghosts)
        # 2) Convert to tensors (CPU)
        x_t = torch.from_numpy(x_np).float()
        gt_c_t = torch.from_numpy(gt_pack).float()
        gt_m_t = torch.from_numpy(gt_mask).bool()
        gt_p_t = torch.from_numpy(gt_present).long()
        keep_t = torch.from_numpy(keep_identity_np).bool()
        # 3) Fixed targets on CPU (vectorized helper works on CPU as well)
        tc, tm, tp, mi = build_fixed_targets_fast(
            x_t, gt_c_t, gt_m_t, gt_p_t, keep_t,
            max_center_dist=0.3,
        )
        # 4) Vectorized labels on CPU
        N = x_np.shape[0]
        labels_all = _labels_from_budgets(self.budgets, N)
        labels_all_t = torch.from_numpy(labels_all).long()
        labels_all_b = labels_all_t.unsqueeze(0)  # [1,N]
        safe_mi = mi.clamp(min=0)
        tgt_sem = torch.gather(labels_all_b, 1, safe_mi.unsqueeze(0)).squeeze(0)
        tgt_sem = torch.where(mi == -1, torch.full_like(tgt_sem, -1), tgt_sem)
        inp_prior = torch.where(keep_t, tgt_sem, torch.full_like(tgt_sem, -1))

        return {
            'raster': torch.from_numpy(it['ras']).float(),   # [3,H,W]
            'proposal': x_t,                                  # [N,P,2]
            'tgt_coords': tc,                                 # [N,P,2]
            'tgt_mask': tm,                                   # [N,P]
            'tgt_present': tp,                                # [N]
            'input_prior_labels': inp_prior.long(),           # [N]
            'target_sem_labels': tgt_sem.long(),              # [N]
        }


def is_main_process() -> bool:
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def main() -> None:
    ap = argparse.ArgumentParser(description='Clean full training (DDP) — multi-scene EDM refine')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--scene-list', nargs='*', default=None, help='List of scene IDs for training')
    ap.add_argument('--scene-file', default=None, help='Path to a text file with scene IDs (one per line)')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--iters-per-epoch', type=int, default=0, help='0 = iterate full dataset per epoch; >0 = cap iterations')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--prefetch', type=int, default=2, help='prefetch_factor for DataLoader workers (>0 only when workers>0)')
    ap.add_argument('--accum-steps', type=int, default=1, help='Gradient accumulation steps (>=1)')
    ap.add_argument('--repeat-per-epoch', type=int, default=1000, help='Virtual repeats of the small scene set per epoch')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--prior-lr-mult', type=float, default=2.0)
    ap.add_argument('--sched', choices=['cosine', 'none'], default='cosine')
    ap.add_argument('--lr-min', type=float, default=1e-5)
    # EDM schedule
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.03)
    # augmentation
    ap.add_argument('--shift-sigma', type=float, default=0.10)
    ap.add_argument('--point-sigma', type=float, default=0.02)
    ap.add_argument('--drop-frac', type=float, default=0.15)
    ap.add_argument('--ghosts', type=int, default=2)
    # loss
    ap.add_argument('--l1-weight', type=float, default=20.0)
    ap.add_argument('--cls-weight', type=float, default=5.0)
    ap.add_argument('--use-focal', action='store_true')
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    ap.add_argument('--step-loss-weight', type=float, default=1.0)
    ap.add_argument('--final-loss-weight', type=float, default=1.0)
    ap.add_argument('--anchor-max-center-dist', type=float, default=0.3)
    # io
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_full_clean')
    ap.add_argument('--save-every', type=int, default=5)
    # ddp
    ap.add_argument('--dist-backend', default='nccl')
    ap.add_argument('--local_rank', type=int, default=-1)
    args = ap.parse_args()

    # DDP init
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank if args.local_rank >= 0 else 0))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend=args.dist_backend)
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(0 + (dist.get_rank() if distributed else 0))

    # Stats / caps / budgets
    stats = json.load(open(args.stats_json, 'r'))
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_points=P, num_queries=N)

    # Scene list
    if args.scene_file:
        scenes = [s.strip() for s in open(args.scene_file, 'r').read().splitlines() if s.strip()]
    elif args.scene_list:
        scenes = list(args.scene_list)
    else:
        # fallback: first 10 scenes from static_root
        scenes = [fn[:-4] for fn in sorted(os.listdir(args.static_root)) if fn.endswith('.pkl')][:10]
    if len(scenes) == 0:
        raise RuntimeError('no scenes provided')

    # Dataset / Loader
    ds = MultiSceneCleanDataset(
        args.static_root, args.rendered_root, scenes, caps, budgets,
        shift_sigma=float(args.shift_sigma), point_sigma=float(args.point_sigma),
        drop_frac=float(args.drop_frac), ghosts=int(args.ghosts),
        repeat_per_epoch=int(args.repeat_per_epoch))
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True) if distributed else None
    # prefetch_factor is effective only when workers>0; allow CLI override
    pf = int(getattr(args, 'prefetch', 2))
    dl = DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(int(args.workers) > 0),
        prefetch_factor=(pf if int(args.workers) > 0 else None),
    )

    # Model
    enc = RasterEncoder(out_dim=256).to(device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)

    # Optimizer with prior LR multiplier
    prior_params: List[torch.nn.Parameter] = []
    prior_params += list(base.prior_mlp.parameters())
    prior_params += [base.prior_gate]
    if getattr(base, 'class_emb', None) is not None:
        prior_params += list(base.class_emb.parameters())
        prior_params += [base.class_gate]
    prior_ids = {id(p) for p in prior_params}
    rest_backbone = [p for p in base.parameters() if id(p) not in prior_ids]
    param_groups = [
        {'params': list(enc.parameters()) + rest_backbone, 'lr': float(args.lr), 'weight_decay': 1e-4},
        {'params': prior_params, 'lr': float(args.lr) * float(args.prior_lr_mult), 'weight_decay': 1e-4},
    ]
    opt = torch.optim.AdamW(param_groups)
    # Scheduler
    if args.sched == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=float(args.lr_min))
    else:
        sched = None

    # Wrap with DDP (wrap net and enc for forward; optimizer references base/enc params directly)
    if distributed:
        net = DDP(net, device_ids=[local_rank], find_unused_parameters=False)
        enc = DDP(enc, device_ids=[local_rank], find_unused_parameters=False)

    out_dir = osp.join(args.out_root, 'run')
    if is_main_process():
        os.makedirs(out_dir, exist_ok=True)

    # Training loop
    for ep in range(1, int(args.epochs) + 1):
        if sampler is not None:
            sampler.set_epoch(ep)
        # Avoid explicit device sync here; DDP synchronizes during backward/step
        enc.train(); base.train()
        iters_cap = int(args.iters_per_epoch)
        accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
        accum_counter = 0
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(dl):
            if iters_cap > 0 and it >= iters_cap:
                break
            # move to device
            r_b = batch['raster'].to(device, non_blocking=True)                # [B,3,H,W]
            tgt_c_b = batch['tgt_coords'].to(device, non_blocking=True)        # [B,N,P,2]
            tgt_m_b = batch['tgt_mask'].to(device, non_blocking=True)          # [B,N,P]
            tgt_p_b = batch['tgt_present'].to(device, non_blocking=True)       # [B,N]
            # Proposals and fixed targets are precomputed on CPU by dataset workers
            x_b = batch['proposal'].to(device, non_blocking=True)              # [B,N,P,2]

            # Fixed targets per sample
            B = x_b.shape[0]
            # Targets and labels (already built in dataset)
            tgt_coords = batch['tgt_coords'].to(device, non_blocking=True)
            tgt_mask_o = batch['tgt_mask'].to(device, non_blocking=True)
            tgt_present_o = batch['tgt_present'].to(device, non_blocking=True)
            input_prior_labels = batch['input_prior_labels'].to(device, non_blocking=True)  # [B,N]
            target_sem_labels = batch['target_sem_labels'].to(device, non_blocking=True)  # [B,N]

            # SDEdit start from x_b with small alpha
            noise = torch.randn_like(x_b)
            xK = torch.clamp((1.0 - float(args.alpha)) * x_b + float(args.alpha) * noise, -1.0, 1.0)

            # Encode raster and run EDM
            with torch.cuda.amp.autocast(enabled=False):
                rv = enc(r_b)
            sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
            pred_coords, pred_logits, preds, _states = edm_unrolled_train(
                net if not isinstance(net, DDP) else net.module,
                xK, rv, sigmas, second_order=bool(args.second_order),
                cond_prior=x_b, input_labels=input_prior_labels)

            # Loss
            step_losses = []
            for item in preds:
                pc, pl = item[0], item[1]
                sem_step = item[2] if (len(item) >= 3) else None
                out = criterion(pc, pl, tgt_coords, tgt_mask_o, tgt_present_o,
                                l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                                use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                pred_sem_logits=sem_step, tgt_sem_labels=target_sem_labels, sem_weight=float(args.sem_weight))
                step_losses.append(out['loss_cls'] + out['loss_reg'] + out.get('loss_sem', pc.new_zeros([])))
            step_loss = torch.stack(step_losses).mean() if step_losses else pred_coords.new_zeros([])
            out_final = criterion(pred_coords, pred_logits, tgt_coords, tgt_mask_o, tgt_present_o,
                                  l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                                  use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                  pred_sem_logits=(preds[-1][2] if (len(preds) > 0 and len(preds[-1]) >= 3) else None),
                                  tgt_sem_labels=target_sem_labels, sem_weight=float(args.sem_weight))
            final_loss = out_final['loss_cls'] + out_final['loss_reg'] + out_final.get('loss_sem', pred_coords.new_zeros([]))
            loss = float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss

            # Gradient accumulation
            (loss / float(accum_steps)).backward()
            accum_counter += 1
            if (accum_counter % accum_steps) == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            if is_main_process() and (it % 50 == 0):
                cur_lr = opt.param_groups[0]['lr']
                print(f"[ep {ep:03d} it {it:04d}] lr={cur_lr:.6g} loss={float(loss.item()):.6f} (step={float(step_loss.item()):.6f} final={float(final_loss.item()):.6f})")

        # Flush remaining grads if accumulation incomplete at epoch end
        if (accum_counter % accum_steps) != 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        # Scheduler step per epoch
        if sched is not None:
            sched.step()

        # Save ckpt
        if is_main_process() and ((ep % int(args.save_every) == 0) or (ep == int(args.epochs))):
            os.makedirs(out_dir, exist_ok=True)
            torch.save({'encoder': (enc.module.state_dict() if isinstance(enc, DDP) else enc.state_dict()),
                        'net': (base.state_dict()), 'P': P, 'N': N, 'budgets': budgets},
                       osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth'))

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if is_main_process():
        print(f"[ok] full clean training done. out={out_dir}")


if __name__ == '__main__':
    main()
