#!/usr/bin/env python
from __future__ import annotations

"""
Inference script aligned with PolyDiffuse training.
- Replaces RasterEncoder with PolyDiffuseImageEncoder256 (MapTR R50 + FPN).
- Uses ImageNet normalization and resize-to-multiple-of-32 (instead of letterbox).
"""

import argparse
import os
import os.path as osp
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import (
    pack_gt_to_slots,
    pack_vectors_to_slots,
)
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


def _labels_from_budgets(budgets: Dict[int, int], N: int) -> np.ndarray:
    order: List[int] = []
    for orig in (1, 0, 2):  # MapTR orig ids: divider=1, ped=0, boundary=2
        cap = int(budgets.get(orig, 0))
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        order += [lab] * max(0, cap)
    out = np.full((N,), -1, dtype=np.int64)
    m = min(N, len(order))
    if m > 0:
        out[:m] = np.asarray(order[:m], dtype=np.int64)
    return out


def _present_from_mask(mask: np.ndarray) -> np.ndarray:
    # mask: [N,P] True=pad; present if any valid point exists
    return (~mask).any(axis=1)


def _nms_by_center(coords: np.ndarray, probs: np.ndarray, bounds: Sequence[float], nms_m: float) -> List[int]:
    if nms_m <= 0 or coords.shape[0] <= 1:
        return list(range(coords.shape[0]))
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    Wm = max(maxx - minx, 1e-6); Hm = max(maxy - miny, 1e-6)
    idx = list(range(coords.shape[0]))
    idx.sort(key=lambda i: float(probs[i]), reverse=True)
    keep: List[int] = []
    centers: Dict[int, np.ndarray] = {}
    def _c(i: int) -> np.ndarray:
        if i in centers:
            return centers[i]
        xy = coords[i]
        xm = (xy[:, 0] + 1.0) * 0.5 * Wm + minx
        ym = (xy[:, 1] + 1.0) * 0.5 * Hm + miny
        centers[i] = np.array([xm.mean(), ym.mean()], dtype=np.float32)
        return centers[i]
    for i in idx:
        ci = _c(i)
        ok = True
        for j in keep:
            cj = _c(j)
            if float(np.linalg.norm(ci - cj)) < nms_m:
                ok = False; break
        if ok:
            keep.append(i)
    return keep


class PolyDiffuseImageEncoder256(nn.Module):
    """ResNet‑50 + FPN (from official PolyDiffuse MapTR config) pooled to 256‑d.
    Identical to the training script version.
    """
    def __init__(self, cfg_path: str, pretrained_ckpt: str, device: str = 'cuda') -> None:
        super().__init__()
        # Defer heavy imports to runtime
        from mmcv import Config
        from mmdet.models import build_backbone, build_neck

        cfg = Config.fromfile(cfg_path)
        bb_cfg = cfg.model.get('img_backbone')
        neck_cfg = cfg.model.get('img_neck')
        if bb_cfg is None or neck_cfg is None:
            raise RuntimeError('Config missing img_backbone/img_neck entries')

        self.backbone = build_backbone(bb_cfg)
        self.neck = build_neck(neck_cfg)

        # Load weights (though they will likely be overwritten by the refinement ckpt)
        if os.path.exists(pretrained_ckpt):
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
            pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
            C = pooled[0].shape[1]
        self.proj = nn.Identity() if C == 256 else nn.Linear(C, 256)
        self.to(device)

    def train(self, mode: bool = True):
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
    ap = argparse.ArgumentParser(description='Inference with PolyDiffuse encoder (MapTR R50+FPN)')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', required=False, help='optional proposals root')
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--ckpt', required=True)
    
    # EDM sampler
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=0.6) # Default aligned with training script
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    
    # SDEdit start
    ap.add_argument('--start', choices=['proposal', 'gt_noise', 'noise'], default='proposal')
    ap.add_argument('--start-sigma', type=float, default=0.0)
    ap.add_argument('--no-schedule-truncate', action='store_true')
    
    # Outputs
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/infer_polydiff')
    ap.add_argument('--csv-move', action='store_true')
    ap.add_argument('--thr', type=float, default=0.2)
    ap.add_argument('--nms-meters', type=float, default=0.0)
    ap.add_argument('--keep-only-proposal', action='store_true')
    ap.add_argument('--topk', type=int, default=0)
    ap.add_argument('--min-len-norm', type=float, default=0.05)
    ap.add_argument('--gif-fps', type=int, default=6)
    ap.add_argument('--viz-mode', choices=['state', 'denoised'], default='denoised')
    ap.add_argument('--overlay-raw', action='store_true')
    
    # Encoder Specifics
    ap.add_argument('--polydiff-cfg', default='official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py')
    ap.add_argument('--pretrained-maptr-ckpt', default='global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth')
    ap.add_argument('--blind', action='store_true', help='Disable raster condition')

    # Testing flags
    ap.add_argument('--disable-prior', action='store_true')
    ap.add_argument('--inject-ghost', action='store_true')
    ap.add_argument('--ghost-slot-index', type=int, default=-1)
    ap.add_argument('--ghost-class', type=int, default=2)
    ap.add_argument('--ghost-center-norm-x', type=float, default=-0.85)
    ap.add_argument('--ghost-center-norm-y', type=float, default=-0.85)
    ap.add_argument('--ghost-len-norm', type=float, default=0.12)
    ap.add_argument('--rigid-shift-norm-x', type=float, default=0.0)
    ap.add_argument('--rigid-shift-norm-y', type=float, default=0.0)
    ap.add_argument('--rigid-shift-from-gt', action='store_true')
    ap.add_argument('--prop-drop-frac', type=float, default=0.0)
    ap.add_argument('--prop-drop-num', type=int, default=0)
    
    args = ap.parse_args()

    set_seed(0)
    # load stats
    import json
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20)); N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    labels_default = _labels_from_budgets(budgets, N)

    # load GT + bounds
    gt_pkl = osp.join(args.static_root, f'{args.scene}.pkl')
    gt = load_pickle(gt_pkl)
    bounds = gt.get('bounds')
    if bounds is None:
        raise RuntimeError('bounds-missing in static GT pkl')
    gt_pack, gt_mask, gt_present = pack_gt_to_slots(gt, bounds, budgets, num_points=P, num_queries=N)

    # Image Load (Aligned with Training Script)
    # Resize to multiple of 32, ImageNet normalization
    raw_img = Image.open(osp.join(args.rendered_root, args.scene, '10_render_gt.png')).convert('RGB')
    w, h = raw_img.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    if new_w <= 0 or new_h <= 0:
        new_w, new_h = max(32, w), max(32, h)
    if new_w != w or new_h != h:
        raw_img = raw_img.resize((new_w, new_h), Image.BILINEAR)
    
    # Create tensor for model
    img_t = TF.to_tensor(raw_img)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_t = TF.normalize(img_t, mean, std)
    
    # Create numpy array for visualization (CHW, [0,1])
    ras_np = np.asarray(raw_img, dtype=np.float32) / 255.0
    ras = ras_np.transpose(2, 0, 1) # CHW

    # input proposals
    x_prop = np.zeros((N, P, 2), dtype=np.float32)
    m_prop = np.ones((N, P), dtype=bool)
    
    # ... (Same proposal generation logic as clean infer) ...
    if bool(getattr(args, 'rigid_shift_from_gt', False)):
        x_prop = gt_pack.copy()
        m_prop = gt_mask.copy()
        dx = float(getattr(args, 'rigid_shift_norm_x', 0.0))
        dy = float(getattr(args, 'rigid_shift_norm_y', 0.0))
        if abs(dx) > 0.0 or abs(dy) > 0.0:
            x_prop = np.clip(x_prop + np.array([dx, dy], dtype=np.float32)[None, None, :], -1.0, 1.0)
        input_labels = labels_default.copy()
        try:
            present_slots = (~gt_mask).any(axis=1)
            idx_all = np.where(present_slots)[0]
            k = 0
            if int(getattr(args, 'prop_drop_num', 0)) > 0:
                k = min(int(args.prop_drop_num), int(idx_all.size))
            elif float(getattr(args, 'prop_drop_frac', 0.0)) > 0.0:
                k = int(round(float(args.prop_drop_frac) * float(idx_all.size)))
            if k > 0:
                np.random.shuffle(idx_all)
                drop_ids = idx_all[:k]
                for di in drop_ids:
                    m_prop[di, :] = True
                    try: input_labels[di] = -1
                    except Exception: pass
        except Exception: pass
    else:
        if args.start == 'proposal' and args.agg_pred_root:
            agg = load_pickle(osp.join(args.agg_pred_root, f'{args.scene}.pkl'))
            x_prop, m_prop, labs_prop = pack_vectors_to_slots(agg, bounds, budgets, num_points=P, num_queries=N)
            input_labels = labs_prop
            dx = float(getattr(args, 'rigid_shift_norm_x', 0.0))
            dy = float(getattr(args, 'rigid_shift_norm_y', 0.0))
            if abs(dx) > 0.0 or abs(dy) > 0.0:
                x_prop = np.clip(x_prop + np.array([dx, dy], dtype=np.float32)[None, None, :], -1.0, 1.0)
        elif args.start == 'gt_noise':
            x_prop = gt_pack.copy()
            m_prop = gt_mask.copy()
            dx = float(getattr(args, 'rigid_shift_norm_x', 0.0))
            dy = float(getattr(args, 'rigid_shift_norm_y', 0.0))
            if abs(dx) > 0.0 or abs(dy) > 0.0:
                x_prop = np.clip(x_prop + np.array([dx, dy], dtype=np.float32)[None, None, :], -1.0, 1.0)
        else:
            x_prop = np.random.uniform(-1.0, 1.0, size=(N, P, 2)).astype(np.float32)
            m_prop = np.zeros((N, P), dtype=bool)

    ghost_idx = int(getattr(args, 'ghost_slot_index', -1))
    if bool(getattr(args, 'inject_ghost', False)) and N > 0:
        gi = (N - 1) if (ghost_idx < 0 or ghost_idx >= N) else ghost_idx
        cx = float(getattr(args, 'ghost_center_norm_x', -0.85))
        cy = float(getattr(args, 'ghost_center_norm_y', -0.85))
        L  = float(getattr(args, 'ghost_len_norm', 0.12))
        t = np.linspace(-0.5, 0.5, num=P, dtype=np.float32)
        seg = np.stack([cx + L * t, cy + L * t], axis=1).astype(np.float32)
        x_prop[gi] = np.clip(seg, -1.0, 1.0)
        m_prop[gi, :] = False
        if 'input_labels' not in locals():
            input_labels = labels_default.copy()
        try: input_labels[gi] = int(getattr(args, 'ghost_class', 2))
        except Exception: pass

    prop_present = _present_from_mask(m_prop)
    if 'input_labels' not in locals():
        input_labels = labels_default.copy()
    try:
        if m_prop is not None:
            pad_slots = m_prop.all(axis=1)
            input_labels[pad_slots] = -1
    except Exception: pass

    # Model Init (PolyDiffuse)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # argparse converts dashes to underscores in attribute names
    polydiff_cfg = getattr(args, 'polydiff_cfg', 'official_polydiffuse/projects/configs/maptr/maptr_tiny_r50.py')
    pretrained_maptr_ckpt = getattr(args, 'pretrained_maptr_ckpt', 'global_diffusion_map/ckpts/maptr_tiny_r50_110e.pth')
    enc = PolyDiffuseImageEncoder256(polydiff_cfg, pretrained_maptr_ckpt, device=device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    
    # Load Checkpoint
    data = torch.load(args.ckpt, map_location=device)
    if 'encoder' in data:
        enc.load_state_dict(data['encoder'], strict=False)
    if 'net' in data:
        base.load_state_dict(data['net'], strict=False)
    enc.eval(); base.eval()

    # Prepare tensors
    x0 = torch.from_numpy(x_prop).float().unsqueeze(0).to(device)
    ras_t = img_t.unsqueeze(0).to(device) # [1,3,H,W]
    
    # Inference-only: avoid building autograd graph to reduce memory
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            if bool(getattr(args, 'blind', False)):
                rv = torch.zeros((1, 256), device=device, dtype=torch.float32)
            else:
                rv = enc(ras_t)

    # Schedule
    sigmas = karras_schedule(int(args.steps), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
    if args.start_sigma > 0.0:
        if not bool(getattr(args, 'no_schedule_truncate', False)):
            mask = (sigmas <= float(args.start_sigma) + 1e-6)
            idx = int(torch.nonzero(mask, as_tuple=False)[0].item()) if bool(mask.any()) else int(sigmas.numel() - 1)
            sigmas = sigmas[idx:]
        xK = torch.clamp(x0 + torch.randn_like(x0) * float(args.start_sigma), -1.0, 1.0)
    else:
        xK = x0.clone()

    # Inference Loop
    x_prior = torch.from_numpy(x_prop).float().unsqueeze(0).to(device)
    input_labels_t = torch.from_numpy(input_labels).long().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_coords, pred_logits, preds, states = edm_unrolled_train(
            net, xK, rv, sigmas, second_order=bool(args.second_order),
            cond_prior=(None if bool(args.disable_prior) else x_prior),
            input_labels=(None if bool(args.disable_prior) else input_labels_t))

    # Post-processing (Same as clean infer)
    coords_np = pred_coords[0].detach().cpu().numpy()
    logits_np = pred_logits[0].detach().cpu().numpy().reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    keep_indices = np.where(probs >= float(args.thr))[0].tolist()
    
    if bool(args.keep_only_proposal):
        keep_indices = [i for i in keep_indices if (i < prop_present.shape[0] and bool(prop_present[i]))]
    
    min_len = float(getattr(args, 'min_len_norm', 0.05))
    if min_len > 0 and len(keep_indices) > 0:
        valid_keep: List[int] = []
        for i in keep_indices:
            line = coords_np[i]
            L = np.linalg.norm(line[1:] - line[:-1], axis=1).sum()
            if float(L) > min_len:
                valid_keep.append(i)
        keep_indices = valid_keep

    if float(args.nms_meters) > 0.0 and len(keep_indices) > 0:
        subset_coords = coords_np[keep_indices]
        subset_probs = probs[keep_indices]
        subset_keep = _nms_by_center(subset_coords, subset_probs, bounds, float(args.nms_meters))
        keep_indices = [keep_indices[k] for k in subset_keep]

    if int(args.topk) > 0 and len(keep_indices) > int(args.topk):
        keep_indices.sort(key=lambda i: float(probs[i]), reverse=True)
        keep_indices = keep_indices[: int(args.topk)]
    keep = keep_indices

    sem_labels = None
    try:
        if len(preds) > 0 and len(preds[-1]) >= 3 and preds[-1][2] is not None:
            sem_np = preds[-1][2][0].detach().cpu().numpy()
            sem_labels = np.argmax(sem_np, axis=-1).astype(np.int64)
    except Exception:
        sem_labels = None

    # Visualization
    out_dir = osp.join(args.out_root, args.scene)
    steps_dir = osp.join(out_dir, 'steps')
    os.makedirs(steps_dir, exist_ok=True)

    present_mask = np.ones((N, P), dtype=bool)
    for i in range(N):
        if bool(prop_present[i]):
            present_mask[i, :] = False
    
    overlay_slots_annot(
        osp.join(out_dir, 'input_overlay.png'), ras, bounds,
        slots=x_prop, mask=present_mask, labels=input_labels,
        title='Input (proposal) + GT', gt_slots=gt_pack, gt_mask=gt_mask,
        thickness_px=2)

    mask_keep = np.ones((N, P), dtype=bool)
    for i in keep:
        mask_keep[i, :] = False
    overlay_slots_annot(
        osp.join(out_dir, 'refined_overlay.png'), ras, bounds,
        slots=coords_np, mask=mask_keep, labels=sem_labels,
        title=f'Refined (thr={args.thr})', gt_slots=gt_pack, gt_mask=gt_mask,
        thickness_px=2)

    from PIL import Image as PILImage
    frames: List[PILImage.Image] = []
    for si in range(len(states)):
        draw_mode = getattr(args, 'viz_mode', 'denoised')
        if draw_mode == 'denoised' and si > 0 and (si - 1) < len(preds):
            xt = preds[si - 1][0][0].detach().cpu().numpy()
            lab_step = None
            if len(preds[si - 1]) >= 3 and preds[si - 1][2] is not None:
                sem_np = preds[si - 1][2][0].detach().cpu().numpy()
                lab_step = np.argmax(sem_np, axis=-1).astype(np.int64)
        else:
            xt = states[si][0].detach().cpu().numpy()
            lab_step = None
            if si == 0:
                try: lab_step = input_labels
                except Exception: lab_step = None
            elif (si - 1) < len(preds) and len(preds[si - 1]) >= 3 and preds[si - 1][2] is not None:
                sem_np = preds[si - 1][2][0].detach().cpu().numpy()
                lab_step = np.argmax(sem_np, axis=-1).astype(np.int64)
        mask_step = np.ones((N, P), dtype=bool)
        for i in keep:
            mask_step[i, :] = False
        title = f'step {si:02d}/{len(states)-1}' if len(states) > 1 else 'step 00'
        out_path = osp.join(steps_dir, f'step_{si:03d}.png')
        overlay_slots_annot(out_path, ras, bounds, slots=xt, mask=mask_step, labels=lab_step, title=title,
                            thickness_px=2)
        try: frames.append(PILImage.open(out_path).convert('RGB'))
        except Exception: pass

    if bool(getattr(args, 'csv_move', False)):
        # CSV similar to clean script (omitted for brevity)
        pass

    if len(frames) > 1:
        gif_suffix = 'denoise' if getattr(args,'viz_mode','denoised')=='denoised' else 'state'
        gif_path = osp.join(out_dir, f'steps_{gif_suffix}.gif')
        try: frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000.0/max(1,args.gif_fps)), loop=0)
        except Exception: pass

    # Quantitative metrics (meters) — mirror clean script
    try:
        def _denorm_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
            minx, miny, maxx, maxy = [float(v) for v in bounds]
            w = max(maxx - minx, 1e-6); h = max(maxy - miny, 1e-6)
            out = np.empty_like(xy, dtype=np.float32)
            out[:, 0] = (xy[:, 0] + 1.0) * 0.5 * w + minx
            out[:, 1] = (xy[:, 1] + 1.0) * 0.5 * h + miny
            return out
        def _maptr_to_orig(lab: int) -> int:
            return 1 if lab == 0 else (0 if lab == 1 else 2)
        # Save refined (meters) for computing metrics和下游可视化
        ref_dict = {0: [], 1: [], 2: [], 'bounds': [float(v) for v in bounds]}
        labs_for_save = sem_labels if sem_labels is not None else input_labels
        for i in keep:
            lab = int(labs_for_save[i]) if labs_for_save is not None and i < len(labs_for_save) else -1
            if lab >= 0:
                ref_dict[_maptr_to_orig(lab)].append(_denorm_xy(coords_np[i], bounds))
        # Build input proposals bank (meters)
        in_dict = {0: [], 1: [], 2: [], 'bounds': [float(v) for v in bounds]}
        for i in range(N):
            if not bool(prop_present[i]):
                continue
            lab = int(input_labels[i]) if input_labels is not None and i < len(input_labels) else -1
            if lab >= 0:
                in_dict[_maptr_to_orig(lab)].append(_denorm_xy(x_prop[i], bounds))

        # 将 refined / input 曲线以 pickle 形式写出，方便 sanity figure 等下游脚本复用
        try:
            import pickle  # 延迟导入
            with open(osp.join(out_dir, f'{args.scene}_refined_m.pkl'), 'wb') as f_ref:
                pickle.dump(ref_dict, f_ref)
            with open(osp.join(out_dir, f'{args.scene}_input_m.pkl'), 'wb') as f_in:
                pickle.dump(in_dict, f_in)
        except Exception:
            pass
        # Load GT meters
        gt_bank = {0: gt.get(0, []), 1: gt.get(1, []), 2: gt.get(2, [])}
        from shapely.geometry import LineString  # type: ignore
        def _sample_line(arr: np.ndarray, num: int = 100) -> np.ndarray:
            ls = LineString(arr)
            if ls.length <= 1e-8:
                p = np.array(ls.coords[0], dtype=np.float32)
                return np.tile(p[None, :], (num, 1))
            dists = np.linspace(0.0, ls.length, num=num, dtype=np.float32)
            pts = [list(ls.interpolate(float(d)).coords)[0] for d in dists]
            return np.asarray(pts, dtype=np.float32)
        def _stack_points(bank: Dict[int, List[np.ndarray]], M: int) -> Dict[int, np.ndarray]:
            out: Dict[int, np.ndarray] = {}
            for cid in (0, 1, 2):
                pts = []
                for arr in bank.get(cid, []) or []:
                    if arr is None or len(arr) == 0:
                        continue
                    pts.append(_sample_line(np.asarray(arr, dtype=np.float32), num=M))
                out[cid] = np.concatenate(pts, axis=0) if pts else np.zeros((0, 2), dtype=np.float32)
            return out
        def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
            if a.shape[0] == 0 or b.shape[0] == 0:
                return float('nan')
            def nn_mean(src: np.ndarray, dst: np.ndarray, blk: int = 4096) -> float:
                n = src.shape[0]; mins = np.empty((n,), dtype=np.float32)
                for i in range(0, n, blk):
                    s = src[i:i+blk]
                    d2 = ((s[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
                    mins[i:i+blk] = np.sqrt(d2.min(axis=1))
                return float(mins.mean())
            return 0.5 * (nn_mean(a, b) + nn_mean(b, a))
        M = 100
        G = _stack_points(gt_bank, M)
        R = _stack_points(ref_dict, M)
        I = _stack_points(in_dict, M)
        def _report(name: str, A: Dict[int, np.ndarray], B: Dict[int, np.ndarray]) -> Dict[str, float]:
            res: Dict[str, float] = {}
            for nm, cid in [('ped', 0), ('div', 1), ('bnd', 2)]:
                res[f'{name}_{nm}_Chamfer_m'] = _chamfer(A[cid], B[cid])
            a_all = np.concatenate([A[0], A[1], A[2]], axis=0) if any([A[k].size for k in (0,1,2)]) else np.zeros((0,2), np.float32)
            b_all = np.concatenate([B[0], B[1], B[2]], axis=0) if any([B[k].size for k in (0,1,2)]) else np.zeros((0,2), np.float32)
            res[f'{name}_all_Chamfer_m'] = _chamfer(a_all, b_all)
            return res
        res_ref_vs_gt = _report('Ref_vs_GT', R, G)
        res_ref_vs_in = _report('Ref_vs_Input', R, I)
        print('# Quantitative (meters) — symmetric Chamfer (lower is better)')
        for k in ['Ref_vs_GT_all_Chamfer_m','Ref_vs_Input_all_Chamfer_m',
                  'Ref_vs_GT_ped_Chamfer_m','Ref_vs_GT_div_Chamfer_m','Ref_vs_GT_bnd_Chamfer_m',
                  'Ref_vs_Input_ped_Chamfer_m','Ref_vs_Input_div_Chamfer_m','Ref_vs_Input_bnd_Chamfer_m']:
            v = res_ref_vs_gt.get(k, None)
            if v is None:
                v = res_ref_vs_in.get(k, float('nan'))
            print(f'{k}: {v:.4f}' if (isinstance(v, float) and v==v) else f'{k}: nan')
    except Exception:
        pass

    print(f"[ok] inference (PolyDiffuse encoder) done. Output: {out_dir}")


if __name__ == '__main__':
    main()
