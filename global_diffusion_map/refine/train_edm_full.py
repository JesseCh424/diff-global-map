#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import os.path as osp
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as tud

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, pack_gt_to_slots, pack_vectors_to_slots
from global_diffusion_map.refine.loss_refine import criterion, HungarianMatcher, gpu_greedy_match
from global_diffusion_map.refine.augment import augment_planA_from_gt_torch
from global_diffusion_map.refine.single_scene_overfit import RasterEncoder, letterbox
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _load_pickle(path: str) -> Dict:
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_raster_png(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # [3,H,W]


def _scene_list_from_root(root: str) -> List[str]:
    ids: List[str] = []
    if not osp.isdir(root):
        return ids
    for fn in os.listdir(root):
        if fn.endswith('.pkl'):
            ids.append(osp.splitext(fn)[0])
    ids.sort()
    return ids


class MultiSceneSimDataset:
    """Dynamic simulation dataset across many scenes.

    Uses existing dynamic modes (delete/create/refine) with jitter/drop/ghost。
    不新增额外高斯或仿射变换（按当前实现）。
    """

    def __init__(
        self,
        static_root: str,
        rendered_root: str,
        agg_pred_root: str,
        caps: RefineCaps,
        class_budgets: Dict[int, int],
        jitter_sigma_m: float = 0.6,
        drop_frac_range: Sequence[float] = (0.3, 0.5),
        ghosts_range: Sequence[int] = (3, 6),
    ) -> None:
        self.static_root = static_root
        self.rendered_root = rendered_root
        self.agg_pred_root = agg_pred_root
        self.caps = caps
        self.jitter_sigma_m = float(jitter_sigma_m)
        self.drop_lo, self.drop_hi = float(drop_frac_range[0]), float(drop_frac_range[1])
        self.gh_lo, self.gh_hi = int(ghosts_range[0]), int(ghosts_range[1])
        self.scene_ids = _scene_list_from_root(static_root)
        self.budgets = {int(k): int(v) for k, v in class_budgets.items()}
        # lightweight caches
        self._gt_cache: Dict[str, Dict] = {}
        self._ras_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _rand_ghost(self, P: int) -> np.ndarray:
        pts = np.random.uniform(-1.0, 1.0, size=(P, 2)).astype(np.float32)
        for k in range(1, P):
            pts[k] = 0.7 * pts[k] + 0.3 * pts[k - 1]
        return pts

    def sample(self) -> Dict[str, torch.Tensor]:
        assert len(self.scene_ids) > 0, 'no scenes under static_root'
        scene = random.choice(self.scene_ids)
        gt_pkl = osp.join(self.static_root, f'{scene}.pkl')
        gt = self._gt_cache.get(scene)
        if gt is None:
            gt = _load_pickle(gt_pkl)
            self._gt_cache[scene] = gt
        bounds = gt.get('bounds')
        if bounds is None:
            raise RuntimeError(f'bounds-missing: {gt_pkl}')

        # pack GT once per sample
        tgt_coords, tgt_mask, tgt_present = pack_gt_to_slots(gt, bounds, self.budgets,
                                                             num_points=self.caps.num_points,
                                                             num_queries=self.caps.num_queries)
        # choose mode (skip CPU aug when REFINE_GPU_AUG=1; let GPU override build proposals)
        import numpy as _np
        prop = tgt_coords.copy()
        if os.environ.get('REFINE_GPU_AUG', '0') != '1':
            mode = random.choice(['delete_test', 'create_test', 'refine_test'])
            present = ~tgt_mask.all(axis=1)
            N, P = self.caps.num_queries, self.caps.num_points
            if mode == 'delete_test':
                prop[present] = _np.clip(prop[present] + _np.random.normal(scale=self.jitter_sigma_m, size=prop[present].shape).astype(_np.float32), -1.0, 1.0)
                k = random.randint(self.gh_lo, self.gh_hi)
                empty = _np.where(~present)[0].tolist()
                random.shuffle(empty)
                for i in empty[:k]:
                    prop[i] = self._rand_ghost(P)
            elif mode == 'create_test':
                ids = _np.where(present)[0].tolist()
                random.shuffle(ids)
                drop_num = max(1, int(round(len(ids) * random.uniform(self.drop_lo, self.drop_hi))))
                for i in ids[:drop_num]:
                    prop[i] = _np.random.normal(size=(P, 2)).astype(_np.float32)
            else:  # refine_test
                prop[present] = _np.clip(prop[present] + _np.random.normal(scale=self.jitter_sigma_m, size=prop[present].shape).astype(_np.float32), -1.0, 1.0)

        # raster
        ras = self._ras_cache.get(scene)
        if ras is None:
            ras_raw = _load_raster_png(osp.join(self.rendered_root, scene, '10_render_gt.png'))  # CHW RGB [0,1]
            canvas_bgr, _, _, _ = letterbox(ras_raw, 1024, (1024, 1024))  # HWC BGR uint8
            # convert to CHW RGB [0,1]
            canvas_rgb = canvas_bgr[:, :, ::-1].astype(np.float32) / 255.0
            ras = np.transpose(canvas_rgb, (2, 0, 1))
            self._ras_cache[scene] = ras

        # tensors
        x = torch.from_numpy(prop).float()
        r = torch.from_numpy(ras).float()
        tgt_c = torch.from_numpy(tgt_coords).float()
        tgt_m = torch.from_numpy(tgt_mask).bool()
        tgt_p = torch.from_numpy(tgt_present).long()
        return {'proposal': x, 'raster': r, 'tgt_coords': tgt_c, 'tgt_mask': tgt_m, 'tgt_present': tgt_p}


class MultiSceneRealDataset:
    """真实上游 Proposal（聚合向量）数据集。

    不新增额外噪声；训练阶段由 xK 混合 (alpha) 统一处理。
    """

    def __init__(
        self,
        static_root: str,
        rendered_root: str,
        agg_pred_root: str,
        caps: RefineCaps,
        class_budgets: Dict[int, int],
    ) -> None:
        self.static_root = static_root
        self.rendered_root = rendered_root
        self.agg_pred_root = agg_pred_root
        self.caps = caps
        self.budgets = {int(k): int(v) for k, v in class_budgets.items()}
        # choose scenes that exist in both static and aggregated preds
        s_static = set(_scene_list_from_root(static_root))
        s_agg = set(_scene_list_from_root(agg_pred_root))
        self.scene_ids = sorted(list(s_static & s_agg))
        self._gt_cache: Dict[str, Dict] = {}
        self._ras_cache: Dict[str, np.ndarray] = {}
        self._agg_cache: Dict[str, Dict] = {}

    def __len__(self) -> int:
        return len(self.scene_ids)

    def sample(self) -> Dict[str, torch.Tensor]:
        assert len(self.scene_ids) > 0, 'no overlapping scenes between static and aggregated preds'
        scene = random.choice(self.scene_ids)
        # static GT
        gt = self._gt_cache.get(scene)
        if gt is None:
            gt = _load_pickle(osp.join(self.static_root, f'{scene}.pkl'))
            self._gt_cache[scene] = gt
        bounds = gt.get('bounds')
        if bounds is None:
            raise RuntimeError(f'bounds-missing: static has no bounds for scene={scene}')
        tgt_coords, tgt_mask, tgt_present = pack_gt_to_slots(gt, bounds, self.budgets,
                                                             num_points=self.caps.num_points,
                                                             num_queries=self.caps.num_queries)
        # proposals from aggregated preds
        agg = self._agg_cache.get(scene)
        if agg is None:
            agg = _load_pickle(osp.join(self.agg_pred_root, f'{scene}.pkl'))
            self._agg_cache[scene] = agg
        prop_coords, prop_mask, _labs = pack_vectors_to_slots(agg, bounds, self.budgets,
                                                              num_points=self.caps.num_points,
                                                              num_queries=self.caps.num_queries)
        # raster
        ras = self._ras_cache.get(scene)
        if ras is None:
            ras_raw = _load_raster_png(osp.join(self.rendered_root, scene, '10_render_gt.png'))
            canvas_bgr, _, _, _ = letterbox(ras_raw, 1024, (1024, 1024))
            canvas_rgb = canvas_bgr[:, :, ::-1].astype(np.float32) / 255.0
            ras = np.transpose(canvas_rgb, (2, 0, 1))
            self._ras_cache[scene] = ras

        # tensors
        x = torch.from_numpy(prop_coords).float()
        r = torch.from_numpy(ras).float()
        tgt_c = torch.from_numpy(tgt_coords).float()
        tgt_m = torch.from_numpy(tgt_mask).bool()
        tgt_p = torch.from_numpy(tgt_present).long()
        return {'proposal': x, 'raster': r, 'tgt_coords': tgt_c, 'tgt_mask': tgt_m, 'tgt_present': tgt_p}


def main() -> None:
    ap = argparse.ArgumentParser(description='EDM full-data refine training (multi-scene, 70/30 sim/real mix)')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--iters-per-epoch', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--mix-real-prob', type=float, default=0.3, help='probability a sample comes from real-proposal dataset')
    # DDP
    ap.add_argument('--ddp', action='store_true', help='enable DistributedDataParallel (torchrun)')
    ap.add_argument('--local_rank', type=int, default=0)
    ap.add_argument('--find-unused', action='store_true', help='DDP: set find_unused_parameters=True (default False)')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4, help='DataLoader workers per process (IterableDataset)')
    ap.add_argument('--prefetch', type=int, default=2, help='DataLoader prefetch_factor (per worker)')
    # EDM schedule
    ap.add_argument('--steps', type=int, default=18)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=1.5)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.03, help='SDEdit blend (xK=(1-a)*x + a*N)')
    # Loss
    ap.add_argument('--cls-weight', type=float, default=1.0)
    ap.add_argument('--use-focal', action='store_true')
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    # Enable per-step denoise loss by default (align with one-scene flow)
    ap.add_argument('--step-loss-weight', type=float, default=1.0)
    ap.add_argument('--final-loss-weight', type=float, default=1.0)
    ap.add_argument('--smooth-weight', type=float, default=0.0, help='curvature/smoothness regularization weight')
    ap.add_argument('--reg-len-exp', type=float, default=0.0, help='exponent for length-proportional weighting of regression loss (0=off, 1=proportional)')
    ap.add_argument('--smooth-inv-len-exp', type=float, default=0.0, help='exponent for inverse-length weighting of smooth loss (0=off, 1=proportional)')
    ap.add_argument('--dir-weight', type=float, default=0.2, help='directional consistency loss weight')
    # LR scheduler
    ap.add_argument('--sched', choices=['cosine', 'plateau', 'none'], default='cosine', help='LR scheduler type')
    ap.add_argument('--lr-min', type=float, default=2e-5, help='min LR for cosine/plateau clamp')
    ap.add_argument('--plateau-factor', type=float, default=0.5, help='ReduceLROnPlateau factor')
    ap.add_argument('--plateau-patience', type=int, default=10, help='ReduceLROnPlateau patience (epochs)')
    # IO
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_edm_full')
    ap.add_argument('--save-viz', action='store_true', help='save training-time visualizations (off by default)')
    ap.add_argument('--amp', action='store_true', help='enable mixed precision (AMP) for speed')
    ap.add_argument('--log-every', type=int, default=20, help='print progress every N iters within an epoch (rank0 only)')
    ap.add_argument('--accum-steps', type=int, default=1, help='gradient accumulation steps per optimizer update')
    ap.add_argument('--ckpt-every', type=int, default=1, help='save checkpoint every N epochs (rank0 only)')
    ap.add_argument('--resume-ckpt', type=str, default=None, help='optional checkpoint to resume weights (encoder/net)')
    ap.add_argument('--ckpt-iters', type=int, default=0, help='save checkpoint every N iters within an epoch (0=off, rank0 only)')
    # Anchor matching options
    ap.add_argument('--anchor-match', action='store_true', help='Use input proposals (pre-denoise) to build fixed GT assignment (anchor matching)')
    ap.add_argument('--anchor-max-center-dist', type=float, default=0.2, help='Max normalized center distance for anchor matching; pairs beyond are dropped (treated as background)')
    args = ap.parse_args()

    # Preflight
    if not osp.isdir(args.static_root):
        raise FileNotFoundError(args.static_root)
    if not osp.isdir(args.rendered_root):
        raise FileNotFoundError(args.rendered_root)
    if not osp.isdir(args.agg_pred_root):
        raise FileNotFoundError(args.agg_pred_root)

    # DDP init
    import torch.distributed as dist
    use_ddp = bool(args.ddp) or (int(os.environ.get('WORLD_SIZE', '1')) > 1)
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank)) if use_ddp else 0
    world_size = int(os.environ.get('WORLD_SIZE', '1')) if use_ddp else 1
    if use_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    set_seed(0 + (local_rank if use_ddp else 0))
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    # Set defaults to align full training with one-scene GPU-optimized path
    os.environ.setdefault('REFINE_VEC_LOSS', '1')
    os.environ.setdefault('REFINE_GPU_MATCH', '1')
    os.environ.setdefault('REFINE_GPU_CHAMFER', '1')
    os.environ.setdefault('REFINE_GPU_AUG', '1')
    os.environ.setdefault('GPU_AUG_JITTER', '0.0')
    os.environ.setdefault('GPU_AUG_DROP_LO', '0.2')
    os.environ.setdefault('GPU_AUG_DROP_HI', '0.4')
    os.environ.setdefault('GPU_AUG_GH_LO', '2')
    os.environ.setdefault('GPU_AUG_GH_HI', '4')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_queries=N, num_points=P)

    ds_sim = MultiSceneSimDataset(
        static_root=args.static_root,
        rendered_root=args.rendered_root,
        agg_pred_root=args.agg_pred_root,
        caps=caps, class_budgets=budgets,
        jitter_sigma_m=0.0, drop_frac_range=(0.2, 0.4), ghosts_range=(2, 4))
    ds_real = MultiSceneRealDataset(
        static_root=args.static_root,
        rendered_root=args.rendered_root,
        agg_pred_root=args.agg_pred_root,
        caps=caps, class_budgets=budgets)

    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    enc = RasterEncoder(out_dim=256).to(device)
    # Disable semantic head for speed & DDP grad completeness (unused sem loss)
    # Enable semantic head to align with one-scene training (3 classes)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    # DDP preferred over DP for better scaling
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        enc = DDP(enc, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=bool(args.find_unused))
        net = DDP(net, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=bool(args.find_unused))
    opt = optim.AdamW(list(enc.parameters()) + list(net.parameters()), lr=args.lr, weight_decay=args.wd)
    # establish default start epoch
    start_epoch = 1
    resume_data = None
    # Optional resume (weights + optimizer/scheduler if present)
    if args.resume_ckpt is not None and str(args.resume_ckpt).strip() != '':
        ckpt_path = str(args.resume_ckpt)
        try:
            data = torch.load(ckpt_path, map_location=device)
            resume_data = data
            if 'encoder' in data:
                if hasattr(enc, 'module'):
                    enc.module.load_state_dict(data['encoder'], strict=False)
                else:
                    enc.load_state_dict(data['encoder'], strict=False)
            if 'net' in data:
                if hasattr(base, 'module'):
                    base.module.load_state_dict(data['net'], strict=False)
                else:
                    base.load_state_dict(data['net'], strict=False)
            print(f"[resume] loaded weights from {ckpt_path}")
        except Exception as e:
            print(f"[warn] resume failed: {e}")
    # LR scheduler (cosine/plateau/none) — default cosine
    sched = None
    if getattr(args, 'sched', None) is None:
        # add CLI options if not present (backward compatible when imported differently)
        pass
    
    # Note: CLI args for scheduler are defined below (added to parser)
    use_amp = bool(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = osp.join(args.out_root, 'run')
    if bool(args.save_viz) and ((not use_ddp) or (dist.get_rank() == 0)):
        viz_dir = osp.join(out_dir, 'viz')
        os.makedirs(viz_dir, exist_ok=True)
    matcher = HungarianMatcher(w_center=1.0, w_dir=0.2, w_pw=0.5)

    # Build mixed iterable dataset (sim/real) and DataLoader for throughput
    class _MixIterable(tud.IterableDataset):
        def __init__(self, ds_sim, ds_real, p_real: float) -> None:
            super().__init__()
            self.ds_sim = ds_sim
            self.ds_real = ds_real
            self.p_real = float(p_real)

        def __iter__(self):
            import random as _rnd
            while True:
                if _rnd.random() < self.p_real:
                    yield self.ds_real.sample()
                else:
                    yield self.ds_sim.sample()

    mix_ds = _MixIterable(ds_sim, ds_real, p_real=float(args.mix_real_prob))
    per_proc_batch = max(1, int(args.batch // (world_size if use_ddp else 1)))
    loader = tud.DataLoader(
        mix_ds,
        batch_size=per_proc_batch,
        num_workers=max(0, int(args.workers)),
        pin_memory=True,
        persistent_workers=(int(args.workers) > 0),
        prefetch_factor=max(1, int(args.prefetch)) if int(args.workers) > 0 else None,
    )
    data_iter = iter(loader)

    torch.backends.cudnn.benchmark = True
    enc.train(); base.train()
    # Build LR scheduler after optimizer and before training loop
    # Placed here to access args; mirrors one-scene defaults
    try:
        import torch.optim.lr_scheduler as _lrs
        if getattr(args, 'sched', 'cosine') == 'cosine':
            sched = _lrs.CosineAnnealingLR(opt, T_max=int(args.epochs), eta_min=float(getattr(args, 'lr_min', 2e-5)))
        elif getattr(args, 'sched', 'cosine') == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            sched = ReduceLROnPlateau(opt, mode='min', factor=float(getattr(args, 'plateau_factor', 0.5)),
                                      patience=int(getattr(args, 'plateau_patience', 10)),
                                      min_lr=float(getattr(args, 'lr_min', 2e-5)))
        else:
            sched = None
    except Exception:
        sched = None

    # If resuming, also restore optimizer/scheduler state and start epoch when available
    if resume_data is not None:
        try:
            if 'optimizer' in resume_data:
                opt.load_state_dict(resume_data['optimizer'])
            # load scheduler state if both present
            has_sched = ('sched' in resume_data)
            if has_sched and (sched is not None):
                try:
                    sched.load_state_dict(resume_data['sched'])
                except Exception:
                    pass
            # infer start epoch
            start_epoch = int(resume_data.get('epoch', 0)) + 1
            if start_epoch <= 1:
                # fallback: parse from filename ckpt_ep_XXXX.pth
                import re as _re
                m = _re.search(r"ckpt_ep_(\d+)\.pth", str(args.resume_ckpt))
                if m:
                    start_epoch = int(m.group(1)) + 1
            print(f"[resume] start from epoch {start_epoch}")
            # If no scheduler state, fast-forward scheduler to maintain LR continuity
            if (not has_sched) and (sched is not None) and (start_epoch > 1):
                try:
                    for _ in range(start_epoch - 1):
                        sched.step()
                    print(f"[resume] fast-forwarded scheduler to epoch {start_epoch-1}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[warn] resume (opt/sched/epoch) failed: {e}")
    else:
        # No scheduler state in ckpt (old format). If starting from later epoch, fast-forward scheduler.
        try:
            if (sched is not None) and (start_epoch > 1):
                for _ in range(start_epoch - 1):
                    sched.step()
                print(f"[resume] fast-forwarded scheduler to epoch {start_epoch-1}")
        except Exception:
            pass

    for ep in range(start_epoch, args.epochs + 1):
        # Precompute sigma schedule once per epoch (PolyDiffuse-style)
        sigmas_epoch = karras_schedule(max(1, int(args.steps)), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
        for it in range(args.iters_per_epoch):
            opt.zero_grad(set_to_none=True)
            accum = max(1, int(args.accum_steps))
            loss_print = None
            # Optional: build anchor matching (fixed assignment per batch) before denoising
            anchor_targets = None
            if bool(getattr(args, 'anchor_match', False)):
                # x_b/r_b/tgt_* will be set in the first accumulation iteration below; we need a one-shot fetch here
                try:
                    tmp_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    tmp_batch = next(data_iter)
                # move to device
                x_b_anchor = tmp_batch['proposal'].to(device, non_blocking=True)
                r_b_anchor = tmp_batch['raster'].to(device, non_blocking=True)
                tgt_c_b_anchor = tmp_batch['tgt_coords'].to(device, non_blocking=True)
                tgt_m_b_anchor = tmp_batch['tgt_mask'].to(device, non_blocking=True)
                tgt_p_b_anchor = tmp_batch['tgt_present'].to(device, non_blocking=True)
                # stash iterator back by rebuilding it with the fetched sample first
                def _prepend(first, it):
                    yield first
                    for v in it:
                        yield v
                data_iter = _prepend(tmp_batch, data_iter)
                # Build fixed targets by matching input proposals (x_b) to GT with distance gating
                B_fix = x_b_anchor.shape[0]
                tgt_coords_list_fix: List[torch.Tensor] = []
                tgt_mask_list_fix: List[torch.Tensor] = []
                tgt_present_list_fix: List[torch.Tensor] = []
                tgt_labels_b_fix = None
                use_gpu_match = (os.environ.get('REFINE_GPU_MATCH', '0') == '1')
                max_d = float(getattr(args, 'anchor_max_center_dist', 0.2))
                for b_fix in range(B_fix):
                    pc_b = x_b_anchor[b_fix]        # [N,P,2] proposals (normalized)
                    gt_b = tgt_c_b_anchor[b_fix]
                    gm_b = tgt_m_b_anchor[b_fix]
                    gp_b = tgt_p_b_anchor[b_fix]
                    if use_gpu_match:
                        pairs = gpu_greedy_match(pc_b, gt_b, gm_b, gp_b,
                                                 w_center=1.0, w_dir=0.2, w_pw=0.5,
                                                 use_chamfer=(os.environ.get('REFINE_GPU_CHAMFER', '1') == '1'))
                    else:
                        # fallback: CPU Hungarian on numpy
                        matcher = HungarianMatcher(w_center=1.0, w_dir=0.2, w_pw=0.5)
                        pc_np = pc_b.detach().cpu().numpy()
                        gt_np = gt_b.detach().cpu().numpy()
                        gm_np = gm_b.detach().cpu().numpy()
                        gp_np = gp_b.detach().cpu().numpy()
                        valid_gt_idx = np.where(gp_np > 0)[0].tolist()
                        tgt_list = [gt_np[j] for j in valid_gt_idx] if len(valid_gt_idx) > 0 else []
                        if len(tgt_list) > 0:
                            pairs_raw = matcher(pc_np, np.stack(tgt_list, axis=0))
                            pairs = [(pi, gj_loc) for (pi, gj_loc) in pairs_raw]
                        else:
                            pairs = []
                    # Distance gating on normalized centers
                    pc_np = pc_b.detach().cpu().numpy()
                    gt_np = gt_b.detach().cpu().numpy()
                    gm_np = gm_b.detach().cpu().numpy()
                    gp_np = gp_b.detach().cpu().numpy()
                    valid_gt_idx = np.where(gp_np > 0)[0].tolist()
                    def _center(arr: np.ndarray) -> np.ndarray:
                        return np.nanmean(arr, axis=0).astype(np.float32) if arr.size else np.zeros((2,), np.float32)
                    new_tgt = np.zeros_like(gt_np)
                    new_msk = np.ones_like(gm_np)
                    new_pre = np.zeros_like(gp_np)
                    order: List[int] = []
                    for orig in (1, 0, 2):
                        cap = int(budgets.get(orig, 0))
                        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                        order += [lab] * max(0, cap)
                    new_lab = np.full_like(gp_np, fill_value=-1, dtype=np.int64)
                    for pi, gj_loc in pairs:
                        if gj_loc < 0 or gj_loc >= len(valid_gt_idx):
                            continue
                        gj = valid_gt_idx[gj_loc]
                        c_pi = _center(pc_np[pi])
                        c_gj = _center(gt_np[gj])
                        d = float(np.linalg.norm(c_pi - c_gj))
                        if d > max_d:
                            continue  # too far: leave as background
                        new_tgt[pi] = gt_np[gj]
                        new_msk[pi] = gm_np[gj]
                        new_pre[pi] = 1
                        if len(order) > gj:
                            new_lab[pi] = int(order[gj])
                    tgt_coords_list_fix.append(torch.from_numpy(new_tgt))
                    tgt_mask_list_fix.append(torch.from_numpy(new_msk))
                    tgt_present_list_fix.append(torch.from_numpy(new_pre))
                    new_lab_t = torch.from_numpy(new_lab)[None, ...]
                    tgt_labels_b_fix = new_lab_t if (tgt_labels_b_fix is None) else torch.cat([tgt_labels_b_fix, new_lab_t], dim=0)
                tgt_coords_m_fix = torch.stack(tgt_coords_list_fix, dim=0).to(device)
                tgt_mask_m_fix = torch.stack(tgt_mask_list_fix, dim=0).to(device)
                tgt_present_m_fix = torch.stack(tgt_present_list_fix, dim=0).to(device)
                if tgt_labels_b_fix is None:
                    tgt_labels_b_fix = torch.full((B_fix, N), -1, dtype=torch.long)
                tgt_labels_m_fix = tgt_labels_b_fix.to(device)
                anchor_targets = (tgt_coords_m_fix, tgt_mask_m_fix, tgt_present_m_fix, tgt_labels_m_fix)

            for a in range(accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                # move to device
                x_b = batch['proposal'].to(device, non_blocking=True)
                r_b = batch['raster'].to(device, non_blocking=True)
                tgt_c_b = batch['tgt_coords'].to(device, non_blocking=True)
                tgt_m_b = batch['tgt_mask'].to(device, non_blocking=True)
                tgt_p_b = batch['tgt_present'].to(device, non_blocking=True)

                # Optional: override CPU-side sim augmentation with GPU augmentation
                if os.environ.get('REFINE_GPU_AUG', '0') == '1':
                    # 统一使用增强函数（禁 jit；drop∈[0.2,0.4]；ghost∈[2,4]）
                    jitter_sigma = float(os.environ.get('GPU_AUG_JITTER', '0.0'))
                    drop_lo = float(os.environ.get('GPU_AUG_DROP_LO', '0.2'))
                    drop_hi = float(os.environ.get('GPU_AUG_DROP_HI', '0.4'))
                    gh_lo = int(float(os.environ.get('GPU_AUG_GH_LO', '2')))
                    gh_hi = int(float(os.environ.get('GPU_AUG_GH_HI', '4')))
                    prop = tgt_c_b.clone()
                    prop, _present_prop = augment_planA_from_gt_torch(
                        prop, tgt_m_b,
                        jitter_sigma=jitter_sigma,
                        drop_lo=drop_lo, drop_hi=drop_hi,
                        ghosts_lo=gh_lo, ghosts_hi=gh_hi,
                    )
                    x_b = prop

                # xK on GPU
                xK = torch.clamp((1.0 - float(args.alpha)) * x_b + float(args.alpha) * torch.randn_like(x_b), -1.0, 1.0)
                # DDP no_sync on accumulation steps (all but last)
                enc_ctx = enc.no_sync if (use_ddp and a < accum - 1 and hasattr(enc, 'no_sync')) else nullcontext
                net_ctx = net.no_sync if (use_ddp and a < accum - 1 and hasattr(net, 'no_sync')) else nullcontext
                with enc_ctx():
                    with net_ctx():
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            rv = enc(r_b)
                            sigmas = sigmas_epoch
                            pred_coords, pred_logits, _preds, _states = edm_unrolled_train(net, xK, rv, sigmas, second_order=bool(args.second_order))

                    # per-step losses（加入语义监督；标签按 budgets 顺序对齐 pack_gt_to_slots 的槽位布局）
                    # Build budget-ordered labels once per batch: MapTR labels 0=divider,1=ped,2=boundary
                    order: List[int] = []
                    for orig in (1, 0, 2):
                        cap = int(budgets.get(orig, 0))
                        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                        order += [lab] * max(0, cap)
                    B_local = x_b.shape[0]
                    step_tgt_labels = torch.full((B_local, N), -1, dtype=torch.long, device=device)
                    upto = min(N, len(order))
                    if upto > 0:
                        step_tgt_labels[:, :upto] = torch.as_tensor(order[:upto], dtype=torch.long, device=device)[None, :]

                    step_losses = []
                    for item in _preds:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            pc, pl = item[0], item[1]
                            sem_step = item[2] if (len(item) >= 3) else None
                        else:
                            continue
                        ls = criterion(pc, pl, tgt_c_b, tgt_m_b, tgt_p_b,
                                       l1_weight=1.0, cls_weight=float(args.cls_weight),
                                       use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                       pred_sem_logits=sem_step, tgt_sem_labels=step_tgt_labels, sem_weight=1.0,
                                       smooth_weight=float(args.smooth_weight),
                                       reg_len_exp=float(args.reg_len_exp), smooth_inv_len_exp=float(args.smooth_inv_len_exp),
                                       dir_weight=float(args.dir_weight))
                        step_losses.append(
                            ls['loss_cls']
                            + ls['loss_reg']
                            + ls.get('loss_sem', pc.new_zeros([]))
                            + ls.get('loss_smooth', pc.new_zeros([]))
                        )
                    step_loss = torch.stack(step_losses).mean() if step_losses else torch.tensor(0.0, device=device)

                    # final matched loss + semantic labels
                    B = x_b.shape[0]
                    tgt_coords_list: List[torch.Tensor] = []
                    tgt_mask_list: List[torch.Tensor] = []
                    tgt_present_list: List[torch.Tensor] = []
                    tgt_labels_b = None
                    if anchor_targets is not None:
                        tgt_coords_m, tgt_mask_m, tgt_present_m, tgt_labels_m = anchor_targets
                    else:
                        use_gpu_match = (os.environ.get('REFINE_GPU_MATCH', '0') == '1')
                        for b in range(B):
                            if use_gpu_match:
                                # GPU greedy matching: keep tensors on device
                                pc_b = pred_coords[b]        # [N,P,2]
                                gt_b = tgt_c_b[b]            # [N,P,2]
                                gm_b = tgt_m_b[b]            # [N,P]
                                gp_b = tgt_p_b[b]            # [N]
                                pairs = gpu_greedy_match(pc_b, gt_b, gm_b, gp_b,
                                                     w_center=1.0, w_dir=0.2, w_pw=0.5,
                                                     use_chamfer=(os.environ.get('REFINE_GPU_CHAMFER', '1') == '1'))
                                # Map to numpy arrays for reuse of packing logic
                                gt_np = gt_b.detach().cpu().numpy()
                                gm_np = gm_b.detach().cpu().numpy()
                            gp_np = gp_b.detach().cpu().numpy()
                            valid_gt_idx = np.where(gp_np > 0)[0].tolist()
                            new_tgt = np.zeros_like(gt_np)
                            new_msk = np.ones_like(gm_np)
                            new_pre = np.zeros_like(gp_np)
                            order: List[int] = []
                            for orig in (1, 0, 2):
                                cap = int(budgets.get(orig, 0))
                                lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                                order += [lab] * max(0, cap)
                            new_lab = np.full_like(gp_np, fill_value=-1, dtype=np.int64)
                            for pi, gj_loc in pairs:
                                gj = valid_gt_idx[gj_loc]
                                new_tgt[pi] = gt_np[gj]
                                new_msk[pi] = gm_np[gj]
                                new_pre[pi] = 1
                                if len(order) > gj:
                                    new_lab[pi] = int(order[gj])
                        else:
                            pc_np = pred_coords[b].detach().cpu().numpy()
                            gt_np = tgt_c_b[b].detach().cpu().numpy()
                            gm_np = tgt_m_b[b].detach().cpu().numpy()
                            gp_np = tgt_p_b[b].detach().cpu().numpy()
                            valid_gt_idx = np.where(gp_np > 0)[0].tolist()
                            if len(valid_gt_idx) > 0:
                                tgt_list = [gt_np[j] for j in valid_gt_idx]
                                pairs = matcher(pc_np, np.stack(tgt_list, axis=0))
                                new_tgt = np.zeros_like(gt_np)
                                new_msk = np.ones_like(gm_np)
                                new_pre = np.zeros_like(gp_np)
                                # Build GT class order from budgets (MapTR labeling)
                                order: List[int] = []
                                for orig in (1, 0, 2):
                                    cap = int(budgets.get(orig, 0))
                                    lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                                    order += [lab] * max(0, cap)
                                new_lab = np.full_like(gp_np, fill_value=-1, dtype=np.int64)
                                for pi, gj_loc in pairs:
                                    gj = valid_gt_idx[gj_loc]
                                    new_tgt[pi] = gt_np[gj]
                                    new_msk[pi] = gm_np[gj]
                                    new_pre[pi] = 1
                                    if len(order) > gj:
                                        new_lab[pi] = int(order[gj])
                            else:
                                new_tgt, new_msk, new_pre = gt_np, gm_np, gp_np
                                new_lab = np.full_like(gp_np, fill_value=-1, dtype=np.int64)
                        tgt_coords_list.append(torch.from_numpy(new_tgt))
                        tgt_mask_list.append(torch.from_numpy(new_msk))
                        tgt_present_list.append(torch.from_numpy(new_pre))
                        new_lab_t = torch.from_numpy(new_lab)[None, ...]
                        tgt_labels_b = new_lab_t if (tgt_labels_b is None) else torch.cat([tgt_labels_b, new_lab_t], dim=0)
                        tgt_coords_m = torch.stack(tgt_coords_list, dim=0).to(device)
                        tgt_mask_m = torch.stack(tgt_mask_list, dim=0).to(device)
                        tgt_present_m = torch.stack(tgt_present_list, dim=0).to(device)
                        if tgt_labels_b is None:
                            tgt_labels_b = torch.full((B, N), -1, dtype=torch.long)
                        tgt_labels_m = tgt_labels_b.to(device)

                    # Final-step semantic logits (if available)
                    pred_sem_logits = None
                    if isinstance(_preds, (list, tuple)) and len(_preds) > 0:
                        last = _preds[-1]
                        if isinstance(last, (list, tuple)) and len(last) >= 3:
                            pred_sem_logits = last[2]

                    losses_final = criterion(pred_coords, pred_logits, tgt_coords_m, tgt_mask_m, tgt_present_m,
                                             l1_weight=1.0, cls_weight=float(args.cls_weight),
                                             use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                                             pred_sem_logits=pred_sem_logits, tgt_sem_labels=tgt_labels_m, sem_weight=1.0,
                                             smooth_weight=float(args.smooth_weight),
                                             reg_len_exp=float(args.reg_len_exp), smooth_inv_len_exp=float(args.smooth_inv_len_exp),
                                             dir_weight=float(args.dir_weight))
                    final_loss = (
                        losses_final['loss_cls']
                        + losses_final['loss_reg']
                        + losses_final.get('loss_sem', pred_coords.new_zeros([]))
                        + losses_final.get('loss_smooth', pred_coords.new_zeros([]))
                    )
                    micro = (float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss) / float(accum)
                if use_amp:
                    scaler.scale(micro).backward()
                else:
                    micro.backward()
                loss_print = micro.detach() if loss_print is None else loss_print + micro.detach()

            if use_amp:
                scaler.step(opt); scaler.update()
            else:
                opt.step()

            # inner-epoch progress logging (rank0 only)
            if ((not use_ddp) or (dist.get_rank() == 0)) and (int(args.log_every) > 0):
                if (it + 1) % int(args.log_every) == 0 or (it == 0):
                    cur_lr = opt.param_groups[0]['lr']
                    lp = loss_print.item() if loss_print is not None else (float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss).item()
                    print(f"[ep {ep:04d} it {it+1:04d}/{args.iters_per_epoch}] lr={cur_lr:.6g} loss={lp:.6f} step={step_loss.item():.6f} final={final_loss.item():.6f}", flush=True)

            # intra-epoch checkpointing every N iters (rank0 only)
            if (int(getattr(args, 'ckpt_iters', 0)) > 0) and ((not use_ddp) or (dist.get_rank() == 0)):
                if (it + 1) % int(args.ckpt_iters) == 0:
                    enc_sd = enc.module.state_dict() if hasattr(enc, 'module') else enc.state_dict()
                    ckpt = {
                        'encoder': enc_sd,
                        'net': base.state_dict(),
                        'P': P,
                        'N': N,
                        'budgets': budgets,
                        'optimizer': opt.state_dict(),
                        'epoch': ep,
                        'iter': it + 1,
                        'iters_per_epoch': int(args.iters_per_epoch),
                    }
                    if sched is not None:
                        try:
                            ckpt['sched'] = sched.state_dict()
                        except Exception:
                            pass
                    os.makedirs(out_dir, exist_ok=True)
                    torch.save(ckpt, osp.join(out_dir, f'ckpt_ep_{ep:04d}_it_{it+1:04d}.pth'))

        # Compute epoch loss for logging/scheduler on all ranks
        lp_epoch = loss_print.item() if 'loss_print' in locals() and loss_print is not None else (float(args.step_loss_weight) * step_loss + float(args.final_loss_weight) * final_loss).item()
        if ep % 1 == 0 and ((not use_ddp) or (dist.get_rank() == 0)):
            print(f"[ep {ep:04d}] loss={lp_epoch:.6f} step={step_loss.item():.6f} final={final_loss.item():.6f}", flush=True)
        # Step LR scheduler (all ranks to keep LR in sync)
        if sched is not None:
            if getattr(args, 'sched', 'cosine') == 'plateau':
                sched.step(lp_epoch)
            else:
                sched.step()
        if (ep % max(1, int(args.ckpt_every)) == 0) and ((not use_ddp) or (dist.get_rank() == 0)):
            # save checkpoint
            # Save encoder (may be DP-wrapped) and base (inner net weights)
            enc_sd = enc.module.state_dict() if hasattr(enc, 'module') else enc.state_dict()
            ckpt = {
                'encoder': enc_sd,
                'net': base.state_dict(),
                'P': P,
                'N': N,
                'budgets': budgets,
                'optimizer': opt.state_dict(),
                'epoch': ep,
            }
            if sched is not None:
                try:
                    ckpt['sched'] = sched.state_dict()
                except Exception:
                    pass
            os.makedirs(out_dir, exist_ok=True)
            torch.save(ckpt, osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth'))

    print(f"[ok] EDM full-data training done. Checkpoints under {out_dir}.")


if __name__ == '__main__':
    main()
    # Reduce CPU thread contention with DataLoader when CPU is hot
    try:
        torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
    except Exception:
        pass
