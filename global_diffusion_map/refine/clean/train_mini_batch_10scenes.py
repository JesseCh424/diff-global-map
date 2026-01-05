#!/usr/bin/env python
from __future__ import annotations

"""
Mini-batch refine training over 10 scenes with optional DDP (4 GPUs).

Goals
- Move from single-scene overfit to small multi-scene training to validate generalization.
- Support proposal-like corruption (drop/ghost/heavy jitter) with probability p to improve robustness.
- Keep the backbone/loss/EDM schedule aligned with the clean one-scene trainer.

Usage (example, 4 GPUs 4/5/6/7)
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    global_diffusion_map/refine/clean/train_mini_batch_10scenes.py \
    --static-root maptracker/work_dirs/static_gt_vector/av2_oldsplit/val \
    --rendered-root maptracker/work_dirs/rendered_gt/av2_oldsplit/val \
    --scenes  <scene_id_1> ... <scene_id_10> \
    --epochs 200 --prop-prob 0.3 --steps 8 --alpha 0.03

Notes
- Batch size per-GPU defaults to 1 scene. Set via --batch.
- SyncBatchNorm is enabled when DDP is detected.
"""

import argparse
import json
import os
import os.path as osp
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots
from global_diffusion_map.refine.loss_refine import criterion, hungarian_match_perm
from global_diffusion_map.refine.single_scene_overfit import RasterEncoder, load_pickle
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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


def _rand_ghost(P: int, scale: float = 0.6) -> np.ndarray:
    pts = (np.random.rand(P, 2).astype(np.float32) * 2.0 - 1.0) * scale
    for k in range(1, P):
        pts[k] = 0.7 * pts[k] + 0.3 * pts[k - 1]
    return np.clip(pts, -1.0, 1.0)


def jitter_drop_ghost(
    gt_pack: np.ndarray,   # [N,P,2]
    gt_mask: np.ndarray,   # [N,P] True=pad
    shift_sigma: float,
    point_sigma: float,
    drop_frac: float,
    ghosts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N, P, _ = gt_pack.shape
    x = gt_pack.copy()
    valid = ~gt_mask
    shifts = np.random.normal(scale=shift_sigma, size=(N, 1, 2)).astype(np.float32)
    local = np.random.normal(scale=point_sigma, size=x.shape).astype(np.float32)
    noise_total = shifts + local
    x[valid] = np.clip(x[valid] + noise_total[valid], -1.0, 1.0)
    present_gt = (~gt_mask).any(axis=1)
    ids = np.where(present_gt)[0].tolist()
    np.random.shuffle(ids)
    k_drop = max(0, int(round(len(ids) * float(drop_frac))))
    drop_ids = set(ids[:k_drop])
    keep_identity = present_gt.copy()
    is_drop = np.zeros((N,), dtype=bool)
    for i in drop_ids:
        keep_identity[i] = False
        is_drop[i] = True
        rnd = np.random.normal(scale=max(shift_sigma, point_sigma) * 0.8, size=(P, 2)).astype(np.float32)
        x[i] = np.clip(gt_pack[i] + rnd, -1.0, 1.0)
    empty_ids = np.where(~present_gt)[0].tolist()
    np.random.shuffle(empty_ids)
    is_ghost = np.zeros((N,), dtype=bool)
    for j in empty_ids[:max(0, int(ghosts))]:
        x[j] = _rand_ghost(P)
        is_ghost[j] = True
    return x, keep_identity, is_drop, is_ghost


def build_fixed_targets(
    x_in: torch.Tensor,         # [N,P,2]
    gt_coords: torch.Tensor,    # [N,P,2]
    gt_mask: torch.Tensor,      # [N,P]
    gt_present: torch.Tensor,   # [N]
    keep_identity_mask: torch.Tensor,  # [N]
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

    remaining_gt = (gt_present > 0).clone()
    remaining_gt[keep_ids] = False
    if bool(remaining_gt.any()):
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


class MultiSceneDataset(Dataset):
    def __init__(self, static_root: str, rendered_root: str, scenes: Sequence[str], stats_json: str,
                 simplify_labels: bool = True) -> None:
        super().__init__()
        self.static_root = static_root
        self.rendered_root = rendered_root
        self.scenes = list(scenes)
        with open(stats_json, 'r') as f:
            stats = json.load(f)
        self.P = int(stats.get('M', 20))
        self.N = int(stats.get('num_queries', 64))
        self.budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
        # preload (10 scenes → fine)
        self.bank: List[Dict[str, object]] = []
        for sid in self.scenes:
            gt = load_pickle(osp.join(self.static_root, f'{sid}.pkl'))
            bounds = gt.get('bounds')
            if bounds is None:
                raise RuntimeError(f'static GT lacks bounds: {sid}')
            gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, self.budgets, num_points=self.P, num_queries=self.N)
            # raster
            from PIL import Image
            ras = np.asarray(Image.open(osp.join(self.rendered_root, sid, '10_render_gt.png')).convert('RGB'), dtype=np.float32) / 255.0
            ras = ras.transpose(2, 0, 1).astype(np.float32)  # CHW
            self.bank.append({'scene': sid, 'bounds': bounds, 'gt_pack': gt_pack, 'gt_mask': gt_mask, 'ras': ras})

    def __len__(self) -> int:
        return len(self.bank)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.bank[idx]


def is_ddp() -> bool:
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1


def ddp_setup() -> int:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser(description='Mini-batch refine training over 10 scenes (DDP-ready)')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--scenes', nargs='+', required=True, help='List of scene IDs (e.g., 10 items)')
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    # Training
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=1, help='per-GPU batch size (scenes)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--prior-lr-mult', type=float, default=2.0)
    ap.add_argument('--sched', choices=['cosine', 'none'], default='cosine')
    ap.add_argument('--lr-min', type=float, default=1e-5)
    # EDM
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.03)
    # Mix policy
    ap.add_argument('--prop-prob', type=float, default=0.3, help='Probability to train on corrupted proposal (drop/ghost/heavy jitter)')
    ap.add_argument('--shift-sigma', type=float, default=0.10)
    ap.add_argument('--point-sigma', type=float, default=0.02)
    ap.add_argument('--heavy-shift', type=float, default=0.30)
    ap.add_argument('--heavy-point', type=float, default=0.10)
    ap.add_argument('--drop-frac', type=float, default=0.15)
    ap.add_argument('--ghosts', type=int, default=2)
    # Loss weights
    ap.add_argument('--l1-weight', type=float, default=20.0)
    ap.add_argument('--cls-weight', type=float, default=5.0)
    ap.add_argument('--use-focal', action='store_true')
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    ap.add_argument('--step-loss-weight', type=float, default=1.0)
    ap.add_argument('--final-loss-weight', type=float, default=1.0)
    ap.add_argument('--anchor-max-center-dist', type=float, default=0.05)
    # IO
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_mini_batch_10scenes')
    ap.add_argument('--save-every', type=int, default=50)
    args = ap.parse_args()

    set_seed(0)

    # DDP init
    local_rank = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ddp = is_ddp()
    if ddp:
        local_rank = ddp_setup()
        device = f'cuda:{local_rank}'

    # Stats/caps
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_points=P, num_queries=N)

    # Dataset & loader
    ds = MultiSceneDataset(args.static_root, args.rendered_root, args.scenes, args.stats_json)
    sampler = DistributedSampler(ds, shuffle=True) if ddp else None
    loader = DataLoader(ds, batch_size=int(args.batch), sampler=sampler, shuffle=(sampler is None),
                        num_workers=int(args.workers), pin_memory=True, drop_last=False)

    # Model
    enc = RasterEncoder(out_dim=256).to(device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    # Separate prior parameters for higher LR
    prior_params: List[torch.nn.Parameter] = []
    prior_params += list(net.backbone.prior_mlp.parameters())
    prior_params += [net.backbone.prior_gate]
    if getattr(net.backbone, 'class_emb', None) is not None:
        prior_params += list(net.backbone.class_emb.parameters())
        prior_params += [net.backbone.class_gate]
    prior_ids = {id(p) for p in prior_params}
    rest_backbone = [p for p in net.backbone.parameters() if id(p) not in prior_ids]
    enc_params = list(enc.parameters())
    param_groups = [
        {'params': enc_params + rest_backbone, 'lr': float(args.lr), 'weight_decay': 1e-4},
        {'params': prior_params, 'lr': float(args.lr) * float(getattr(args, 'prior_lr_mult', 2.0)), 'weight_decay': 1e-4},
    ]
    if ddp:
        # SyncBatchNorm for small per-GPU batch
        base = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base)
        net = EDMPrecondRefine(base, sigma_data=1.0).to(device)  # rebuild wrapper with synced backbone
    opt = torch.optim.AdamW(param_groups)
    if args.sched == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=float(args.lr_min))
    else:
        sched = None

    out_dir = osp.join(args.out_root, 'run')
    if (not ddp) or (ddp and dist.get_rank() == 0):
        os.makedirs(out_dir, exist_ok=True)

    for ep in range(1, int(args.epochs) + 1):
        if ddp:
            sampler.set_epoch(ep)  # type: ignore[union-attr]
        enc.train(); base.train()
        # epoch accumulators for logging (rank0)
        ep_loss_sum = 0.0
        ep_step_sum = 0.0
        ep_final_sum = 0.0
        ep_iters = 0
        # local iterator length (for display only)
        local_total = len(loader)
        for batch in loader:
            # Batch assembly
            ras_list: List[torch.Tensor] = []
            x_list: List[torch.Tensor] = []
            tgt_coords_list: List[torch.Tensor] = []
            tgt_mask_list: List[torch.Tensor] = []
            tgt_present_list: List[torch.Tensor] = []
            input_prior_list: List[torch.Tensor] = []

            for k in range(len(batch['scene'])) if isinstance(batch, dict) and isinstance(batch.get('scene'), list) else range(int(args.batch)):
                # Support both default collate (dict of lists) and simple list batches
                if isinstance(batch, dict):
                    gt_pack = batch['gt_pack'][k].numpy() if hasattr(batch['gt_pack'][k], 'numpy') else batch['gt_pack'][k]
                    gt_mask = batch['gt_mask'][k].numpy() if hasattr(batch['gt_mask'][k], 'numpy') else batch['gt_mask'][k]
                    ras = batch['ras'][k].numpy() if hasattr(batch['ras'][k], 'numpy') else batch['ras'][k]
                else:
                    gt_pack = batch[k]['gt_pack']
                    gt_mask = batch[k]['gt_mask']
                    ras = batch[k]['ras']

                # Mix policy: proposal-like corruption with prob p, else light jitter
                if np.random.rand() < float(args.prop_prob):
                    x_np, keep_identity_np, *_ = jitter_drop_ghost(
                        gt_pack, gt_mask,
                        shift_sigma=float(getattr(args, 'heavy_shift', 0.30)),
                        point_sigma=float(getattr(args, 'heavy_point', 0.10)),
                        drop_frac=float(args.drop_frac), ghosts=int(args.ghosts))
                else:
                    x_np, keep_identity_np, *_ = jitter_drop_ghost(
                        gt_pack, gt_mask,
                        shift_sigma=float(getattr(args, 'shift_sigma', 0.10)),
                        point_sigma=float(getattr(args, 'point_sigma', 0.02)),
                        drop_frac=0.0, ghosts=0)

                # Build targets per sample
                gt_coords = torch.from_numpy(gt_pack).float().to(device)
                gt_mask_t = torch.from_numpy(gt_mask).bool().to(device)
                gt_present_t = torch.from_numpy((~gt_mask).any(axis=1).astype(np.int64)).long().to(device)
                keep_identity = torch.from_numpy(keep_identity_np).to(device)
                t_coords, t_mask, t_present, match_idx = build_fixed_targets(
                    torch.from_numpy(x_np).float().to(device), gt_coords, gt_mask_t, gt_present_t, keep_identity_mask=keep_identity)

                # labels per slot per sample
                labels_all = _labels_from_budgets(budgets, N)
                labels_by_slot = np.full((N,), -1, dtype=np.int64)
                mi = match_idx.detach().cpu().numpy()
                for i in range(N):
                    j = int(mi[i])
                    if j >= 0 and j < N:
                        labels_by_slot[i] = labels_all[j]
                input_prior_np = labels_by_slot.copy()
                input_prior_np[~keep_identity_np] = -1

                ras_list.append(torch.from_numpy(ras).float().to(device))
                x_list.append(torch.from_numpy(x_np).float().to(device))
                tgt_coords_list.append(t_coords)
                tgt_mask_list.append(t_mask)
                tgt_present_list.append(t_present)
                input_prior_list.append(torch.from_numpy(input_prior_np).long().to(device))

            # Stack batch
            ras_t = torch.stack(ras_list, dim=0)  # [B,3,H,W]
            x = torch.stack(x_list, dim=0)        # [B,N,P,2]
            tgt_coords = torch.stack(tgt_coords_list, dim=0)
            tgt_mask_o = torch.stack(tgt_mask_list, dim=0)
            tgt_present_o = torch.stack(tgt_present_list, dim=0)
            input_prior_labels = torch.stack(input_prior_list, dim=0)  # [B,N]

            # SDEdit start
            xK = torch.clamp((1.0 - float(args.alpha)) * x + float(args.alpha) * torch.randn_like(x), -1.0, 1.0)
            with torch.cuda.amp.autocast(enabled=False):
                rv = enc(ras_t)
            sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
            pred_coords, pred_logits, preds, _states = edm_unrolled_train(
                net, xK, rv, sigmas, second_order=bool(args.second_order),
                cond_prior=x, input_labels=input_prior_labels)

            # Loss
            step_losses = []
            for item in preds:
                pc, pl = item[0], item[1]
                sem_step = item[2] if (len(item) >= 3) else None
                out = criterion(pc, pl, tgt_coords, tgt_mask_o, tgt_present_o,
                                l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                                use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                pred_sem_logits=sem_step, tgt_sem_labels=input_prior_labels, sem_weight=float(args.sem_weight))
                step_losses.append(out['loss_cls'] + out['loss_reg'] + out.get('loss_sem', pc.new_zeros([])))
            step_loss = torch.stack(step_losses).mean() if step_losses else pred_coords.new_zeros([])
            out_final = criterion(pred_coords, pred_logits, tgt_coords, tgt_mask_o, tgt_present_o,
                                  l1_weight=float(args.l1_weight), cls_weight=float(args.cls_weight),
                                  use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                  pred_sem_logits=(preds[-1][2] if (len(preds) > 0 and len(preds[-1]) >= 3) else None),
                                  tgt_sem_labels=input_prior_labels, sem_weight=float(args.sem_weight))
            final_loss = out_final['loss_cls'] + out_final['loss_reg'] + out_final.get('loss_sem', pred_coords.new_zeros([]))
            loss = float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss

            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()

            # update epoch accumulators
            ep_iters += 1
            try:
                ep_loss_sum += float(loss.item())
                ep_step_sum += float(step_loss.item()) if step_losses else 0.0
                ep_final_sum += float(final_loss.item())
            except Exception:
                pass

            # per-iteration log on rank0
            if (not ddp) or (ddp and dist.get_rank() == 0):
                cur_lr = opt.param_groups[0]['lr']
                print(f"[ep {ep:04d} it {ep_iters:03d}/{local_total}] lr={cur_lr:.6g} loss={float(loss.item()):.6f} step={float(step_loss.item()):.6f} final={float(final_loss.item()):.6f}")

        # epoch end
        if sched is not None:
            sched.step()
        if (not ddp) or (ddp and dist.get_rank() == 0):
            # epoch summary
            if ep_iters > 0:
                print(f"[ep {ep:04d} avg] loss={ep_loss_sum/ep_iters:.6f} step={ep_step_sum/ep_iters:.6f} final={ep_final_sum/ep_iters:.6f}")
            if (ep % int(args.save_every) == 0) or (ep == int(args.epochs)):
                torch.save({'encoder': (enc.state_dict()), 'net': (base.state_dict()), 'P': P, 'N': N, 'budgets': budgets},
                           osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth'))
            if (ep % 10 == 0) or (ep == 1):
                cur_lr = opt.param_groups[0]['lr']
                print(f"[ep {ep:04d}] lr={cur_lr:.6g} done")

    # Print completion before tearing down PG to avoid get_rank() errors
    if ddp:
        try:
            is_rank0 = (dist.get_rank() == 0)
        except Exception:
            is_rank0 = True
        if is_rank0:
            print(f"[ok] mini-batch training finished. Out: {out_dir}")
        ddp_cleanup()
    else:
        print(f"[ok] mini-batch training finished. Out: {out_dir}")


if __name__ == '__main__':
    main()