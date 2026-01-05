#!/usr/bin/env python
from __future__ import annotations

"""
Ultra-fast multi-scene refine training with on-GPU augmentation + target build.

Design
- Dataset: RAM cache only (no per-item compute), returns clean GT pack + raster
- Training: GPU masked ops for jitter/drop/ghost and vectorized target build
- Matching: per-sample gpu_greedy_match inside a lightweight B-loop

This minimizes CPU overhead, avoids CPU↔GPU sync points in the hot path,
and keeps the step dominated by matrix ops.
"""

import argparse
import os
import os as _os
import json
import glob
import re
import sys
import os
import os.path as osp
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import gc

import sys
# Enforce a stable device ordering to avoid accidental multi-GPU context init.
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.single_scene_overfit import RasterEncoder, load_pickle
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train
from global_diffusion_map.refine.loss_refine import criterion, gpu_greedy_match, hungarian_match_perm


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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


class MemoryDataset(Dataset):
    def __init__(self, static_root: str, rendered_root: str, scenes: List[str], caps: RefineCaps, budgets: Dict[int, int], virtual_mult: int = 1) -> None:
        super().__init__()
        self.items: List[Dict] = []
        self._n: int = 0
        self._virtual_mult: int = max(1, int(virtual_mult))
        for s in scenes:
            gt = load_pickle(osp.join(static_root, f'{s}.pkl'))
            bounds = gt.get('bounds')
            if bounds is None:
                continue
            gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, num_points=caps.num_points, num_queries=caps.num_queries)
            img_path = osp.join(rendered_root, s, '10_render_gt.png')
            img = Image.open(img_path).convert('RGB').resize((1024, 1024), Image.Resampling.BILINEAR)
            ras = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            self.items.append({'gt_pack': gt_pack, 'gt_mask': gt_mask, 'gt_present': gt_present, 'ras': ras})
        self._n = len(self.items)
        if self._n == 0:
            raise RuntimeError('no valid scenes to load (bounds missing?)')

    def __len__(self) -> int:
        # Virtual length to amortize epoch overhead and keep GPU busy longer per epoch
        return self._n * self._virtual_mult

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx % self._n]
        return {
            'raster': torch.from_numpy(it['ras']).float(),
            'gt_coords': torch.from_numpy(it['gt_pack']).float(),
            'gt_mask': torch.from_numpy(it['gt_mask']).bool(),
            'gt_present': torch.from_numpy(it['gt_present']).long(),
        }


@torch.no_grad()
def gpu_augment_and_build_targets(
    gt_coords: torch.Tensor,   # [B,N,P,2]
    gt_mask: torch.Tensor,     # [B,N,P]
    gt_present: torch.Tensor,  # [B,N]
    labels_all_b: torch.Tensor, # [B,N]
    shift_sigma: float,
    point_sigma: float,
    drop_frac: float,
    ghosts: int,
    max_center_dist: float,
) -> Dict[str, torch.Tensor]:
    device = gt_coords.device
    B, N, P, _ = gt_coords.shape
    # Augmentation (masked ops)
    x_b = gt_coords.clone()
    shifts = torch.randn(B, N, 1, 2, device=device, dtype=gt_coords.dtype) * float(shift_sigma)
    locals = torch.randn(B, N, P, 2, device=device, dtype=gt_coords.dtype) * float(point_sigma)
    valid = (~gt_mask).unsqueeze(-1)
    x_b = torch.where(valid, x_b + shifts + locals, x_b).clamp(-1.0, 1.0)

    present = (gt_present > 0)
    rand = torch.rand(B, N, device=device)
    is_drop = (rand < float(drop_frac)) & present
    keep_identity = present & (~is_drop)

    # fill dropped with stronger noise around GT
    drop_noise = torch.randn_like(x_b) * (2.0 * max(float(shift_sigma), float(point_sigma)))
    x_b = torch.where(is_drop.view(B, N, 1, 1), (gt_coords + drop_noise).clamp(-1.0, 1.0), x_b)

    # ghosts in empty slots
    if int(ghosts) > 0:
        is_empty = ~present
        add_ghost = (torch.rand(B, N, device=device) < (float(ghosts) / max(1, N))) & is_empty
        gl = (torch.rand(B, N, P, 2, device=device, dtype=gt_coords.dtype) * 2.0 - 1.0) * 0.6
        x_b = torch.where(add_ghost.view(B, N, 1, 1), gl, x_b)

    # Targets init
    tgt_coords = torch.zeros_like(gt_coords)
    tgt_mask_o = torch.ones_like(gt_mask)
    tgt_present_o = torch.zeros_like(gt_present)
    match_idx = torch.full_like(gt_present, -1)

    # Identity assignment (permutation-invariant orientation): choose fwd/rev by lower masked L1
    self_idx = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    match_idx = torch.where(keep_identity, self_idx, match_idx)
    for b in range(B):
        ids = torch.nonzero(keep_identity[b] & (gt_present[b] > 0), as_tuple=False).view(-1)
        if int(ids.numel()) == 0:
            continue
        xs = x_b[b].index_select(0, ids)          # [K,P,2]
        gs = gt_coords[b].index_select(0, ids)    # [K,P,2]
        ms = gt_mask[b].index_select(0, ids)      # [K,P]
        v = (~ms).float()
        diff_f = (xs - gs).abs().sum(dim=-1)      # [K,P]
        denom = v.sum(dim=1).clamp_min(1.0)
        l1f = (diff_f * v).sum(dim=1) / denom
        gs_rev = torch.flip(gs, dims=[1])
        ms_rev = torch.flip(ms, dims=[1])
        diff_r = (xs - gs_rev).abs().sum(dim=-1)
        l1r = (diff_r * v).sum(dim=1) / denom
        use_rev = (l1r < l1f).view(-1)
        g_final = gs.clone(); m_final = ms.clone()
        if bool(use_rev.any()):
            g_final[use_rev] = gs_rev[use_rev]
            m_final[use_rev] = ms_rev[use_rev]
        tgt_coords[b].index_copy_(0, ids, g_final)
        tgt_mask_o[b].index_copy_(0, ids, m_final)
        tgt_present_o[b].index_copy_(0, ids, torch.ones_like(ids, dtype=tgt_present_o.dtype))

    # Per-sample permutation-invariant Hungarian match for remaining GT
    for b in range(B):
        rem_gt = present[b] & (~keep_identity[b])
        if not bool(rem_gt.any()):
            continue
        sel_idx = torch.nonzero(rem_gt, as_tuple=False).view(-1)
        gt_sel = gt_coords[b].index_select(0, sel_idx)
        gm_sel = gt_mask[b].index_select(0, sel_idx)
        pairs, perm_choice = hungarian_match_perm(x_b[b], None, gt_sel, gm_sel, None, cls_weight=0.0, reg_weight=50.0, use_l1_beta=0.0)
        if not pairs:
            continue
        for (pi, gj_local) in pairs:
            pi_i = int(pi)
            if bool(keep_identity[b, pi_i]):
                continue
            gj_global = int(sel_idx[int(gj_local)].item())
            if not bool(rem_gt[gj_global]):
                continue
            k = int(perm_choice[int(gj_local)]) if (perm_choice is not None and len(perm_choice) > int(gj_local)) else 0
            if k == 1:
                tgt_coords[b, pi_i] = torch.flip(gt_coords[b, gj_global], dims=[0])
                tgt_mask_o[b, pi_i] = torch.flip(gt_mask[b, gj_global], dims=[0])
            else:
                tgt_coords[b, pi_i] = gt_coords[b, gj_global]
                tgt_mask_o[b, pi_i] = gt_mask[b, gj_global]
            tgt_present_o[b, pi_i] = 1
            match_idx[b, pi_i] = gj_global

    safe_idx = match_idx.clamp(min=0)
    target_sem = torch.gather(labels_all_b, 1, safe_idx)
    target_sem = torch.where(match_idx == -1, torch.full_like(target_sem, -1), target_sem)
    input_prior = torch.where(keep_identity, target_sem, torch.full_like(target_sem, -1))

    return {
        'x_b': x_b,
        'tgt_coords': tgt_coords,
        'tgt_mask': tgt_mask_o,
        'tgt_present': tgt_present_o,
        'input_prior': input_prior,
        'target_sem': target_sem,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='GPU-fast refine training (mini multi-scene)')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--scene-file', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--accum-steps', type=int, default=1, help='gradient accumulation steps (>=1)')
    # Augment
    ap.add_argument('--shift-sigma', type=float, default=0.10)
    ap.add_argument('--point-sigma', type=float, default=0.02)
    ap.add_argument('--drop-frac', type=float, default=0.15)
    ap.add_argument('--ghosts', type=int, default=2)
    ap.add_argument('--anchor-max-center-dist', type=float, default=0.1)
    # Optim
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--prior-lr-mult', type=float, default=2.0)
    ap.add_argument('--l1-weight', type=float, default=20.0)
    ap.add_argument('--cls-weight', type=float, default=5.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    # EDM schedule
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--alpha', type=float, default=0.03)
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_gpu_fast')
    ap.add_argument('--save-every', type=int, default=50, help='Save checkpoint every N epochs')
    ap.add_argument('--auto-restart-every', type=int, default=0, help='Auto-exit with nonzero code every N epochs (0=disabled)')
    ap.add_argument('--virtual-mult', type=int, default=100, help='Virtual dataset length multiplier per epoch (>=1)')
    ap.add_argument('--resume', default='', help='Explicit checkpoint path to resume from (overrides auto-scan)')
    ap.add_argument('--vec-loss', action='store_true', default=True,
                    help='enable permutation-invariant vector loss (min over forward/reverse)')
    args = ap.parse_args()

    set_seed(0)
    # Enable MapTR-like orientation-agnostic vector loss by default
    if bool(getattr(args, 'vec_loss', True)):
        os.environ.setdefault('REFINE_VEC_LOSS', '1')
    else:
        os.environ['REFINE_VEC_LOSS'] = '0'
    # Bind to the first visible CUDA device (use CUDA_VISIBLE_DEVICES externally to select GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True  # speed-up on Ampere+
    except Exception:
        pass

    # Stats
    stats = json.load(open(args.stats_json, 'r'))
    P = int(stats.get('M', 20)); N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_points=P, num_queries=N)

    # Scenes
    scenes = [s.strip() for s in open(args.scene_file, 'r').read().splitlines() if s.strip()]
    ds = MemoryDataset(args.static_root, args.rendered_root, scenes, caps, budgets, virtual_mult=int(args.virtual_mult))

    # Model
    enc = RasterEncoder(out_dim=256).to(device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)

    # Optimizer (prior branches with LR mult)
    prior_params: List[torch.nn.Parameter] = []
    prior_params += list(base.prior_mlp.parameters())
    prior_params += [base.prior_gate]
    if getattr(base, 'class_emb', None) is not None:
        prior_params += list(base.class_emb.parameters())
        prior_params += [base.class_gate]
    prior_ids = {id(p) for p in prior_params}
    rest_backbone = [p for p in base.parameters() if id(p) not in prior_ids]
    opt = optim.AdamW([
        {'params': list(enc.parameters()) + rest_backbone, 'lr': float(args.lr), 'weight_decay': 1e-4},
        {'params': prior_params, 'lr': float(args.lr) * float(args.prior_lr_mult), 'weight_decay': 1e-4},
    ])

    labels_all_t = torch.from_numpy(_labels_from_budgets(budgets, N)).long().to(device)

    os.makedirs(args.out_root, exist_ok=True)
    # Auto-resume latest checkpoint under out_root
    start_epoch = 1
    try:
        latest_ckpt = ''
        # Explicit --resume takes priority
        if args.resume and osp.isfile(args.resume):
            latest_ckpt = args.resume
        else:
            # Try last_checkpoint.txt pointer
            last_ptr = osp.join(args.out_root, 'last_checkpoint.txt')
            if osp.isfile(last_ptr):
                with open(last_ptr, 'r') as f:
                    ptr = f.read().strip()
                    if ptr and osp.isfile(ptr):
                        latest_ckpt = ptr
            # Fallback to glob scan
            if not latest_ckpt:
                ckpt_files = sorted(glob.glob(osp.join(args.out_root, 'ckpt_ep_*.pth')))
                if ckpt_files:
                    latest_ckpt = ckpt_files[-1]
        if latest_ckpt:
            print(f"[auto-resume] loading {latest_ckpt} ...")
            ckpt = torch.load(latest_ckpt, map_location=device)
            if 'encoder' in ckpt:
                enc.load_state_dict(ckpt['encoder'], strict=False)
            if 'net' in ckpt:
                base.load_state_dict(ckpt['net'], strict=False)
            if 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch']) + 1
            else:
                m = re.search(r'ckpt_ep_(\d+)\.pth', latest_ckpt)
                if m:
                    start_epoch = int(m.group(1)) + 1
            print(f"[auto-resume] resume from epoch {start_epoch}")
    except Exception as e:
        print(f"[auto-resume] skipped: {e}")

    print(f"[start] gpu-fast training: batch={args.batch} scenes={len(scenes)} P={P} N={N} vis={_os.environ.get('CUDA_VISIBLE_DEVICES','<all>')} start_ep={start_epoch}")
    # Precompute fixed Karras schedule once to avoid per-step allocations
    sigmas_fixed = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
    # Wrap one training step in a function to force scope cleanup each iteration
    def train_step(batch: Dict[str, torch.Tensor]) -> float:
        r = batch['raster'].to(device, non_blocking=True)
        gt_c = batch['gt_coords'].to(device, non_blocking=True)
        gt_m = batch['gt_mask'].to(device, non_blocking=True)
        gt_p = batch['gt_present'].to(device, non_blocking=True)

        labels_b = labels_all_t.unsqueeze(0).expand(gt_c.shape[0], -1)
        with torch.no_grad():
            data = gpu_augment_and_build_targets(
                gt_c, gt_m, gt_p, labels_b,
                float(args.shift_sigma), float(args.point_sigma), float(args.drop_frac), int(args.ghosts), float(args.anchor_max_center_dist),
            )

        noise = torch.randn_like(data['x_b'])
        xK = torch.clamp((1.0 - float(args.alpha)) * data['x_b'] + float(args.alpha) * noise, -1.0, 1.0)

        with torch.cuda.amp.autocast(enabled=True):
            rv = enc(r)
            pred_coords, pred_logits, preds, _ = edm_unrolled_train(
                net, xK, rv, sigmas_fixed, second_order=False,
                cond_prior=data['x_b'], input_labels=data['input_prior'])
            out_final = criterion(pred_coords, pred_logits, data['tgt_coords'], data['tgt_mask'], data['tgt_present'],
                                  l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                                  pred_sem_logits=(preds[-1][2] if (len(preds) > 0 and len(preds[-1]) >= 3) else None),
                                  tgt_sem_labels=data['target_sem'], sem_weight=float(args.sem_weight))
            loss = out_final['loss_cls'] + out_final['loss_reg'] + out_final.get('loss_sem', pred_coords.new_zeros([]))

        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step()
        val = float(loss.item())
        # Explicitly drop local refs (belt and braces)
        del r, gt_c, gt_m, gt_p, labels_b, data, noise, xK, rv, pred_coords, pred_logits, preds, out_final, loss
        return val

    for ep in range(start_epoch, int(args.epochs) + 1):
        enc.train(); base.train()
        # Recreate DataLoader each epoch to avoid any lingering references/pinned buffers
        dl = DataLoader(
            ds,
            batch_size=int(args.batch),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        for it, batch in enumerate(dl):
            loss_val = train_step(batch)
            if it % 50 == 0:
                print(f"[ep {ep:03d} it {it:04d}] loss={loss_val:.6f}")
                try:
                    gc.collect(); torch.cuda.empty_cache()
                except Exception:
                    pass
        # epoch-level cache cleanup
        try:
            del dl
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
        if ep % int(args.save_every) == 0:
            ckpt_path = osp.join(args.out_root, f'ckpt_ep_{ep:04d}.pth')
            torch.save({'encoder': enc.state_dict(), 'net': base.state_dict(), 'epoch': ep}, ckpt_path)
            # Update pointer for faster resume
            try:
                with open(osp.join(args.out_root, 'last_checkpoint.txt'), 'w') as f:
                    f.write(ckpt_path)
            except Exception:
                pass
            print(f"[checkpoint] saved: {ckpt_path}")
            if int(args.auto_restart_every) > 0 and (ep % int(args.auto_restart_every) == 0):
                print(f"[auto-restart] epoch {ep} reached; exiting to clear memory and resume...")
                sys.exit(1)

    print(f"[ok] gpu-fast training done. out={args.out_root}")


if __name__ == '__main__':
    main()
