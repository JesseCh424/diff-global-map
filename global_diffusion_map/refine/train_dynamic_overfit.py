#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
import os.path as osp
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.single_scene_dataset import DynamicOverfitDataset, load_raster_png
from global_diffusion_map.refine.dataset_refine import RefineCaps
from global_diffusion_map.refine.loss_refine import criterion, HungarianMatcher, gpu_greedy_match
from global_diffusion_map.refine.single_scene_overfit import RasterEncoder, overlay_on_raster, load_pickle
from global_diffusion_map.refine.model_refine import SlotMLPWithTime


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main() -> None:
    ap = argparse.ArgumentParser(description='Dynamic super overfitting (delete/create/refine modes)')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--epochs', type=int, default=800)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--w-center', type=float, default=1.0)
    ap.add_argument('--w-dir', type=float, default=0.2)
    ap.add_argument('--w-pw', type=float, default=0.5)
    ap.add_argument('--hybrid', action='store_true', help='Use hybrid batch: 1 easy (refine) + 1 dynamic per step')
    ap.add_argument('--alpha', type=float, default=0.03)
    ap.add_argument('--cls-weight', type=float, default=1.0)
    ap.add_argument('--use-focal', action='store_true')
    ap.add_argument('--focal-alpha', type=float, default=0.25)
    ap.add_argument('--focal-gamma', type=float, default=2.0)
    ap.add_argument('--sem-weight', type=float, default=1.0)
    # pure denoising: no geometric/background regularizers
    # diffusion sampler (multi-step) options
    ap.add_argument('--sampler-steps', type=int, default=1, help='unrolled diffusion steps per batch (>=1)')
    ap.add_argument('--sigma-min', type=float, default=0.01)
    ap.add_argument('--sigma-max', type=float, default=1.0)
    ap.add_argument('--sampler-schedule', type=str, default='log', choices=['log', 'linear'])
    ap.add_argument('--sampler-mode', type=str, default='sdedit', choices=['sdedit', 'noise'], help='start from GT+mix (sdedit) or pure noise around GT (noise)')
    ap.add_argument('--freeze-reg-head', action='store_true', help='Freeze regression head during training')
    ap.add_argument('--freeze-enc', action='store_true', help='Freeze raster encoder during training')
    ap.add_argument('--no-matcher', action='store_true', help='Disable Hungarian matcher; use packed GT targets directly')
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/train_dynamic_overfit')
    ap.add_argument('--resume-ckpt', type=str, default=None)
    args = ap.parse_args()

    # Reduce CPU thread contention
    try:
        torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
    except Exception:
        pass

    import json
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_queries=N, num_points=P)

    set_seed(0)
    ds = DynamicOverfitDataset(
        static_root=args.static_root,
        rendered_root=args.rendered_root,
        agg_pred_root=args.agg_pred_root,
        scene=args.scene,
        caps=caps,
        class_budgets=budgets,
        length=max(2000, args.epochs * 2),
        drop_frac_range=(0.2, 0.4),
        ghosts_range=(2, 4),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = RasterEncoder(out_dim=256).to(device)
    net = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N).to(device)
    # Optional freezing
    if args.freeze_enc:
        for p in enc.parameters():
            p.requires_grad_(False)
    if args.freeze_reg_head:
        for p in net.head.reg_head.parameters():
            p.requires_grad_(False)
    if args.resume_ckpt:
        try:
            ckpt = torch.load(args.resume_ckpt, map_location=device)
            if 'encoder' in ckpt:
                enc.load_state_dict(ckpt['encoder'], strict=False)
            if 'net' in ckpt:
                net.load_state_dict(ckpt['net'], strict=False)
            print(f'[resume] loaded {args.resume_ckpt}')
        except Exception as e:
            print(f'[warn] resume failed: {e}')
    opt = optim.AdamW(list(enc.parameters()) + list(net.parameters()), lr=args.lr, weight_decay=args.wd)

    out_dir = osp.join(args.out_root, args.scene)
    viz_dir = osp.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    matcher = HungarianMatcher(w_center=float(args.w_center), w_dir=float(args.w_dir), w_pw=float(args.w_pw))
    enc.train(); net.train()
    for ep in range(1, args.epochs + 1):
        sample_dyn = ds[ep - 1]
        # build easy (refine) sample directly from GT pack
        gt_pack_np = ds.gt_pack.copy()
        present = ~ds.gt_mask.all(axis=1)
        gt_easy = gt_pack_np.copy()
        gt_easy[present] = np.clip(gt_easy[present] + np.random.normal(scale=0.6, size=gt_easy[present].shape).astype(np.float32), -1.0, 1.0)
        x_easy = torch.from_numpy(gt_easy).float().to(device)
        r_easy = torch.from_numpy(ds.raster).float().to(device)
        tgt_c_easy = torch.from_numpy(ds.gt_pack).float().to(device)
        tgt_m_easy = torch.from_numpy(ds.gt_mask).bool().to(device)
        tgt_p_easy = torch.from_numpy(ds.gt_present).long().to(device)

        # dynamic sample from dataset
        x_dyn = sample_dyn['proposal'].to(device)
        r_dyn = sample_dyn['raster'].to(device)
        tgt_c_dyn = sample_dyn['tgt_coords'].to(device)
        tgt_m_dyn = sample_dyn['tgt_mask'].to(device)
        tgt_p_dyn = sample_dyn['tgt_present'].to(device)

        if args.hybrid:
            x_b = torch.stack([x_easy, x_dyn], dim=0)
            r_b = torch.stack([r_easy, r_dyn], dim=0)
            tgt_c_b = torch.stack([tgt_c_easy, tgt_c_dyn], dim=0)
            tgt_m_b = torch.stack([tgt_m_easy, tgt_m_dyn], dim=0)
            tgt_p_b = torch.stack([tgt_p_easy, tgt_p_dyn], dim=0)
        else:
            x_b = x_dyn[None, ...]
            r_b = r_dyn[None, ...]
            tgt_c_b = tgt_c_dyn[None, ...]
            tgt_m_b = tgt_m_dyn[None, ...]
            tgt_p_b = tgt_p_dyn[None, ...]

        # diffusion start state and schedule
        x_np = x_b.detach().cpu().numpy()
        if args.sampler_mode == 'noise':
            noise = np.random.normal(size=x_np.shape).astype(np.float32)
            x_t = np.clip(x_np + float(args.sigma_max) * noise, -1.0, 1.0)
        else:
            x_t = (1.0 - args.alpha) * x_np + args.alpha * np.random.normal(size=x_np.shape).astype(np.float32)
            x_t = np.clip(x_t, -1.0, 1.0)
        x_t = torch.from_numpy(x_t).to(device)

        rv = enc(r_b)
        steps = max(1, int(args.sampler_steps))
        if steps == 1:
            sigmas = [float(max(args.sigma_min, 1e-4))]
        else:
            if args.sampler_schedule == 'log':
                import numpy as _np
                sigmas = _np.exp(_np.linspace(np.log(max(args.sigma_max, 1e-3)), np.log(max(args.sigma_min, 1e-4)), steps)).astype(float).tolist()
            else:
                import numpy as _np
                sigmas = _np.linspace(args.sigma_max, args.sigma_min, steps).astype(float).tolist()

        total_loss = torch.tensor(0.0, device=device)
        pred_coords = None; pred_logits = None; pred_sem = None
        for si, sigma in enumerate(sigmas):
            t_scalar = torch.full((x_t.shape[0],), fill_value=float(sigma), device=device)
            out = net(x_t, rv, t_scalar)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                pred_coords, pred_logits, pred_sem = out
            else:
                pred_coords, pred_logits = out
                pred_sem = None
            losses = criterion(pred_coords, pred_logits, tgt_c_b, tgt_m_b, tgt_p_b,
                               l1_weight=1.0, cls_weight=float(args.cls_weight),
                               use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                               pred_sem_logits=pred_sem, tgt_sem_labels=tgt_labels_b if ('tgt_labels_b' in locals()) else None, sem_weight=float(args.sem_weight))
            step_loss = (losses['loss_cls'] + losses['loss_reg'] + losses.get('loss_sem', torch.tensor(0.0, device=device)))
            total_loss = total_loss + step_loss
            if si < len(sigmas) - 1:
                sigma_next = float(sigmas[si + 1]); eps = 1e-6
                x_t = x_t + (sigma_next - float(sigma)) * (x_t - pred_coords).detach() / (float(sigma) + eps)
                x_t = x_t.clamp(-1.0, 1.0)
        loss = total_loss / float(len(sigmas))

        if args.no_matcher:
            # Use packed GT targets directly (no reassignment)
            tgt_coords = tgt_c_b
            tgt_mask = tgt_m_b
            tgt_present2 = tgt_p_b
        else:
            # Hungarian match per batch element (GPU greedy optional)
            tgt_coords_list = []
            tgt_mask_list = []
            tgt_present_list = []
            use_gpu_match = (os.environ.get('REFINE_GPU_MATCH', '0') == '1')
            for b in range(x_t.shape[0]):
                if use_gpu_match:
                    pc_b = pred_coords[b]
                    gt_b = tgt_c_b[b]
                    gm_b = tgt_m_b[b]
                    gp_b = tgt_p_b[b]
                    pairs = gpu_greedy_match(pc_b, gt_b, gm_b, gp_b,
                                             w_center=1.0, w_dir=0.2, w_pw=0.5,
                                             use_chamfer=(os.environ.get('REFINE_GPU_CHAMFER', '1') == '1'))
                    gt_np = gt_b.detach().cpu().numpy()
                    gm_np = gm_b.detach().cpu().numpy()
                    gp_np = gp_b.detach().cpu().numpy()
                    valid_gt_idx = np.where(gp_np > 0)[0].tolist()
                    new_tgt = np.zeros_like(gt_np)
                    new_msk = np.ones_like(gm_np)
                    new_pre = np.zeros_like(gp_np)
                    new_lab = np.full_like(gp_np, fill_value=-1, dtype=np.int64)
                    order = []
                    for orig in (1, 0, 2):
                        cap = int(budgets.get(orig, 0))
                        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                        order += [lab] * max(0, cap)
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
                        # matched semantic labels (MapTR order): default -1 (ignore)
                        new_lab = np.full_like(gp_np, fill_value=-1, dtype=np.int64)
                        # build GT class order from budgets
                        order = []
                        for orig in (1, 0, 2):
                            cap = int(budgets.get(orig, 0))
                            lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                            order += [lab] * max(0, cap)
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
                # collect labels
                tgt_label_tensor = torch.from_numpy(new_lab)
                if b == 0:
                    tgt_labels_b = tgt_label_tensor[None, ...]
                else:
                    tgt_labels_b = torch.cat([tgt_labels_b, tgt_label_tensor[None, ...]], dim=0)
            tgt_coords = torch.stack(tgt_coords_list, dim=0).to(device)
            tgt_mask = torch.stack(tgt_mask_list, dim=0).to(device)
            tgt_present2 = torch.stack(tgt_present_list, dim=0).to(device)
            if 'tgt_labels_b' not in locals():
                tgt_labels_b = torch.full((x_t.shape[0], N), -1, dtype=torch.long)
            tgt_labels_b = tgt_labels_b.to(device)

        losses = criterion(pred_coords, pred_logits, tgt_coords, tgt_mask, tgt_present2,
                           l1_weight=1.0, cls_weight=float(args.cls_weight),
                           use_focal=bool(args.use_focal), focal_alpha=float(args.focal_alpha), focal_gamma=float(args.focal_gamma),
                           pred_sem_logits=pred_sem, tgt_sem_labels=tgt_labels_b, sem_weight=float(args.sem_weight))
        loss = (losses['loss_cls'] + losses['loss_reg'] + losses.get('loss_sem', torch.tensor(0.0, device=device)))
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 20 == 0 or ep == 1:
            print(f"[ep {ep:04d}] sigma=[{sigmas[0]:.3f}->{sigmas[-1]:.3f}] steps={len(sigmas)} loss={loss.item():.6f}")
        if ep % 100 == 0 or ep == args.epochs:
            with torch.no_grad():
                pc = pred_coords.detach().cpu().numpy()[0]
                pl = pred_logits.detach().cpu().numpy()[0].reshape(-1)
                prob = 1.0 / (1.0 + np.exp(-pl))
                keep = np.where(prob >= 0.5)[0].tolist()
                pred_list = [pc[i] for i in keep]
                gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
                bounds = gt.get('bounds')
                ras = ds.raster
                overlay_on_raster(osp.join(viz_dir, f'ep_{ep:04d}.png'), ras, bounds, ds.gt_pack, pred_list)
                # also save per-slot cls map for ep
                try:
                    from global_diffusion_map.refine.single_scene_overfit import overlay_slots_annot
                    from global_diffusion_map.refine.single_scene_overfit import letterbox as _letterbox
                    from global_diffusion_map.refine.single_scene_overfit import denorm_xy as _denorm
                    # build mask from keep
                    N, P = caps.num_queries, caps.num_points
                    m_pred = np.ones((N, P), dtype=bool)
                    for i in keep:
                        m_pred[i] = False
                    # predicted semantic labels if available
                    if pred_sem is not None:
                        sem_np = pred_sem.detach().cpu().numpy()[0]
                        labels = np.argmax(sem_np, axis=-1).astype(np.int64)
                    else:
                        # fallback to budget order coloring
                        order = []
                        for orig in (1, 0, 2):
                            cap = int(budgets.get(orig, 0))
                            lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                            order += [lab] * max(0, cap)
                        labels = np.zeros((N,), dtype=np.int64)
                        L = min(N, len(order))
                        if L:
                            labels[:L] = np.asarray(order[:L], dtype=np.int64)
                    overlay_slots_annot(
                        osp.join(viz_dir, f'ep_{ep:04d}_cls.png'),
                        ras,
                        bounds,
                        pc,
                        m_pred,
                        labels=labels,
                        title=f'ep {ep:04d} (cls thr=0.5)'
                    )
                    # points-only overlay for readability
                    try:
                        import cv2 as _cv2
                        canvas, scale, H0, W0 = _letterbox(ras, 1024, (1024, 1024))
                        minx, miny, maxx, maxy = [float(v) for v in bounds]
                        Sx = W0 / max(maxx - minx, 1e-6)
                        Sy = H0 / max(maxy - miny, 1e-6)
                        def _to_px(arr: np.ndarray) -> np.ndarray:
                            xs = ((arr[:, 0] - minx) * Sx)
                            ys = ((maxy - arr[:, 1]) * Sy)
                            pts = np.stack([xs, ys], axis=1)
                            pts = (pts.astype(np.float32) * scale).round().astype(np.int32)
                            return pts
                        # draw kept prediction points (red)
                        for i in keep:
                            arr_w = _denorm(pc[i], bounds)
                            pts = _to_px(arr_w)
                            for (px, py) in pts:
                                _cv2.circle(canvas, (int(px), int(py)), 2, (0, 0, 255), -1)
                        # draw GT points (cyan)
                        gt_exist_idx = np.where(~ds.gt_mask.all(axis=1))[0]
                        for gi in gt_exist_idx:
                            arr_w = _denorm(ds.gt_pack[int(gi)], bounds)
                            pts = _to_px(arr_w)
                            for (px, py) in pts:
                                _cv2.circle(canvas, (int(px), int(py)), 1, (255, 255, 0), -1)
                        _cv2.imwrite(osp.join(viz_dir, f'ep_{ep:04d}_points.png'), canvas)
                    except Exception:
                        pass
                except Exception:
                    pass
                # also add matched-cls overlay (use GT classes for matched pairs, unmatched=gray)
                try:
                    from global_diffusion_map.refine.loss_refine import HungarianMatcher as _M
                    matcher2 = _M(w_center=float(args.w_center), w_dir=float(args.w_dir), w_pw=float(args.w_pw))
                    gt_exist_idx = np.where(~ds.gt_mask.all(axis=1))[0]
                    # Build GT labels per slot from budgets order (MapTR ids)
                    labels_slots = np.zeros((N,), dtype=np.int64)
                    order = []
                    for orig in (1, 0, 2):
                        cap = int(budgets.get(orig, 0))
                        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                        order += [lab] * max(0, cap)
                    L = min(N, len(order))
                    if L:
                        labels_slots[:L] = np.asarray(order[:L], dtype=np.int64)
                    labels_pred = np.full((caps.num_queries,), -1, dtype=np.int64)
                    if gt_exist_idx.size > 0:
                        pairs = matcher2(pc, ds.gt_pack[gt_exist_idx])
                        for pi, gj_loc in pairs:
                            gj = int(gt_exist_idx[gj_loc])
                            labels_pred[int(pi)] = int(labels_slots[gj])
                    overlay_slots_annot(
                        osp.join(viz_dir, f'ep_{ep:04d}_cls_matched.png'),
                        ras,
                        bounds,
                        pc,
                        m_pred,
                        labels=labels_pred,
                        title=f'ep {ep:04d} (matched cls)'
                    )
                except Exception:
                    pass
                ckpt = {'encoder': enc.state_dict(), 'net': net.state_dict(), 'P': P, 'N': N, 'budgets': budgets}
                os.makedirs(out_dir, exist_ok=True)
                torch.save(ckpt, osp.join(out_dir, f'ckpt_ep_{ep:04d}.pth'))

    # After training: add two targeted checks
    try:
        import random
        gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
        bounds = gt.get('bounds')
        ras = ds.raster
        N, P = caps.num_queries, caps.num_points
        present = ~ds.gt_mask.all(axis=1)

        # A) Deletion check: add K ghosts into empty slots
        def _rand_ghost(P: int) -> np.ndarray:
            g = np.random.uniform(-1.0, 1.0, size=(P, 2)).astype(np.float32)
            for k in range(1, P):
                g[k] = 0.7 * g[k] + 0.3 * g[k-1]
            return g
        prop = ds.gt_pack.copy()
        empty = np.where(~present)[0].tolist()
        random.shuffle(empty)
        K = min(len(empty), 5)
        for i in empty[:K]:
            prop[i] = _rand_ghost(P)
        x_t = np.clip((1.0 - args.alpha) * prop + args.alpha * np.random.normal(size=prop.shape).astype(np.float32), -1.0, 1.0)
        with torch.no_grad():
            rv = enc(torch.from_numpy(ras[None, ...]).to(device))
            out = net(torch.from_numpy(x_t[None, ...]).to(device), rv, torch.full((1,), float(args.alpha), device=device))
            if isinstance(out, (tuple, list)) and len(out) == 3:
                pc, pl, ps = out
            else:
                pc, pl = out
                ps = None
            pc = pc.cpu().numpy()[0]
            prob = 1.0 / (1.0 + np.exp(-pl.cpu().numpy()[0].reshape(-1)))
            keep = np.where(prob >= 0.5)[0].tolist()
            pred_list = [pc[i] for i in keep]
            overlay_on_raster(osp.join(viz_dir, 'check_delete.png'), ras, bounds, ds.gt_pack, pred_list)
            # save the inputs of deletion check: x0 (ghost injected) and xK (blended)
            try:
                import numpy as _np
                from global_diffusion_map.refine.single_scene_overfit import overlay_slots_annot
                N, P = caps.num_queries, caps.num_points
                # labels_in from budgets; mark ghosts as -1 (gray)
                order = []
                for orig in (1, 0, 2):
                    cap = int(budgets.get(orig, 0))
                    lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                    order += [lab] * max(0, cap)
                labels_in = _np.zeros((N,), dtype=_np.int64)
                L = min(N, len(order))
                if L:
                    labels_in[:L] = _np.asarray(order[:L], dtype=_np.int64)
                for gi in empty[:K]:
                    labels_in[int(gi)] = -1
                # mask: draw GT present and ghosts
                m_in = _np.ones((N, P), dtype=bool)
                present_np = (~ds.gt_mask).any(axis=1)
                for i in range(N):
                    if present_np[i] or (i in empty[:K]):
                        m_in[i] = False
                overlay_slots_annot(osp.join(viz_dir, 'check_delete_input_x0.png'), ras, bounds, prop, m_in, labels=labels_in, title='check_delete input x0')
                overlay_slots_annot(osp.join(viz_dir, 'check_delete_input_xK.png'), ras, bounds, x_t, m_in, labels=labels_in, title='check_delete input xK')
                with open(osp.join(viz_dir, 'check_delete_input.txt'), 'w') as f:
                    f.write(f'ghost_indices: {empty[:K]}\n')
            except Exception:
                pass
            # save class overlay for kept predictions
            try:
                from global_diffusion_map.refine.single_scene_overfit import overlay_slots_annot
                N, P = caps.num_queries, caps.num_points
                m_keep = np.ones((N, P), dtype=bool)
                for i in keep:
                    m_keep[i] = False
                if ps is not None:
                    sem_np = ps.detach().cpu().numpy()[0]
                    labels = np.argmax(sem_np, axis=-1).astype(np.int64)
                else:
                    labels = np.zeros((N,), dtype=np.int64)
                overlay_slots_annot(osp.join(viz_dir, 'check_delete_cls.png'), ras, bounds, pc, m_keep, labels=labels, title='check_delete (kept cls)')
                # highlight ghosts that were wrongly kept
                ghost_set = set(empty[:K])
                wrong = [i for i in keep if i in ghost_set]
                if wrong:
                    m_ghost = np.ones((N, P), dtype=bool)
                    for i in wrong:
                        m_ghost[i] = False
                    overlay_slots_annot(osp.join(viz_dir, 'check_delete_ghosts.png'), ras, bounds, pc, m_ghost, labels=labels, title=f'ghost kept: {len(wrong)}')
            except Exception:
                pass

        # B) Creation check: drop some GT, let model create from noise slots
        prop = ds.gt_pack.copy()
        ids = np.where(present)[0].tolist(); random.shuffle(ids)
        drop_num = max(1, int(round(len(ids) * 0.4)))
        dropped = ids[:drop_num]
        for i in dropped:
            prop[i] = np.random.normal(size=(P, 2)).astype(np.float32)
        x_t = np.clip((1.0 - args.alpha) * prop + args.alpha * np.random.normal(size=prop.shape).astype(np.float32), -1.0, 1.0)
        with torch.no_grad():
            rv = enc(torch.from_numpy(ras[None, ...]).to(device))
            out = net(torch.from_numpy(x_t[None, ...]).to(device), rv, torch.full((1,), float(args.alpha), device=device))
            if isinstance(out, (tuple, list)) and len(out) == 3:
                pc, pl, ps = out
            else:
                pc, pl = out
                ps = None
            pc = pc.cpu().numpy()[0]
            prob = 1.0 / (1.0 + np.exp(-pl.cpu().numpy()[0].reshape(-1)))
            keep = np.where(prob >= 0.5)[0].tolist()
            pred_list = [pc[i] for i in keep]
            overlay_on_raster(osp.join(viz_dir, 'check_create.png'), ras, bounds, ds.gt_pack, pred_list)
            # save the inputs of creation check: x0 (dropped->noise) and xK (blended)
            try:
                import numpy as _np
                from global_diffusion_map.refine.single_scene_overfit import overlay_slots_annot
                N, P = caps.num_queries, caps.num_points
                order = []
                for orig in (1, 0, 2):
                    cap = int(budgets.get(orig, 0))
                    lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                    order += [lab] * max(0, cap)
                labels_in = _np.zeros((N,), dtype=_np.int64)
                L = min(N, len(order))
                if L:
                    labels_in[:L] = _np.asarray(order[:L], dtype=_np.int64)
                for di in dropped:
                    labels_in[int(di)] = -1
                m_in = _np.ones((N, P), dtype=bool)
                present_np = (~ds.gt_mask).any(axis=1)
                for i in range(N):
                    if present_np[i] or (i in dropped):
                        m_in[i] = False
                overlay_slots_annot(osp.join(viz_dir, 'check_create_input_x0.png'), ras, bounds, prop, m_in, labels=labels_in, title='check_create input x0')
                overlay_slots_annot(osp.join(viz_dir, 'check_create_input_xK.png'), ras, bounds, x_t, m_in, labels=labels_in, title='check_create input xK')
                with open(osp.join(viz_dir, 'check_create_input.txt'), 'w') as f:
                    f.write(f'dropped_indices: {dropped}\n')
            except Exception:
                pass
            # save class overlay + created-only overlay
            try:
                from global_diffusion_map.refine.single_scene_overfit import overlay_slots_annot
                N, P = caps.num_queries, caps.num_points
                m_keep = np.ones((N, P), dtype=bool)
                for i in keep:
                    m_keep[i] = False
                if ps is not None:
                    sem_np = ps.detach().cpu().numpy()[0]
                    labels = np.argmax(sem_np, axis=-1).astype(np.int64)
                else:
                    labels = np.zeros((N,), dtype=np.int64)
                overlay_slots_annot(osp.join(viz_dir, 'check_create_cls.png'), ras, bounds, pc, m_keep, labels=labels, title='check_create (kept cls)')
                created = [i for i in keep if i in set(dropped)]
                if created:
                    m_new = np.ones((N, P), dtype=bool)
                    for i in created:
                        m_new[i] = False
                    overlay_slots_annot(osp.join(viz_dir, 'check_create_new.png'), ras, bounds, pc, m_new, labels=labels, title=f'created from noise: {len(created)}')
            except Exception:
                pass
    except Exception as e:
        print(f"[warn] failed to generate checks: {e}")

    print(f"[ok] dynamic overfit finished. Visualizations under {viz_dir}")


if __name__ == '__main__':
    main()
