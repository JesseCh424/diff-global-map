#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

import sys
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import pack_vectors_to_slots, pack_gt_to_slots
from global_diffusion_map.refine.single_scene_overfit import (
    RasterEncoder,
    overlay_on_raster,
    overlay_slots_annot,
    compute_bounds_from_any,
    load_pickle,
    letterbox,
    denorm_xy,
    to_px,
)
from global_diffusion_map.refine.model_refine import SlotMLPWithTime
from global_diffusion_map.refine.edm import EDMPrecondRefine, karras_schedule, edm_unrolled_train
from global_diffusion_map.refine.loss_refine import gpu_greedy_match


def set_seed(s: int = 0) -> None:
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def label_order_from_budgets(budgets: Dict[int, int]) -> List[int]:
    order: List[int] = []
    for orig in (1, 0, 2):  # divider, ped, boundary
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        for _ in range(int(budgets.get(orig, 0))):
            order.append(lab)
    return order


def maptr_to_orig(lab: int) -> int:
    # MapTR: 0=divider,1=ped,2=boundary → orig ids: 1,0,2
    return 1 if lab == 0 else (0 if lab == 1 else 2)


def save_refined_pkl(path: str, coords: np.ndarray, logits: np.ndarray, labels: np.ndarray,
                     bounds: Sequence[float], thr: float = 0.5) -> None:
    import pickle
    os.makedirs(osp.dirname(path), exist_ok=True)
    out: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    prob = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    for i in range(coords.shape[0]):
        if prob[i] < thr:
            continue
        lab = int(labels[i])
        orig = maptr_to_orig(lab)
        out[orig].append(coords[i])  # still normalized; consumer may denorm by bounds
    obj = {**out, 'bounds': list(map(float, bounds)), 'logits': prob.tolist(), 'labels': labels.tolist()}
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main() -> None:
    ap = argparse.ArgumentParser(description='Refine inference (single scene, no step/state viz)')
    ap.add_argument('--agg-pred-root', required=True)
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--start', choices=['proposal', 'noise', 'blend', 'gt_noise', 'gt_edit', 'proposal_edit'], default='proposal')
    # EDM multi-step sampler
    ap.add_argument('--steps', type=int, default=18)
    ap.add_argument('--sigma-min', type=float, default=0.002)
    ap.add_argument('--sigma-max', type=float, default=1.5)
    ap.add_argument('--rho', type=float, default=7.0)
    ap.add_argument('--second-order', action='store_true')
    # SDEdit 起点（仅用于 refine：从较小噪声开始并跳过前序大噪声步）
    ap.add_argument('--start-sigma', type=float, default=-1.0,
                    help='SDEdit 起点噪声（例如 0.3~0.5）。>0 时：对 proposal 叠加该强度高斯噪声，'
                         '并从该 sigma 起步（丢弃更大的噪声步）。<=0 表示按原始 schedule 全程执行。')
    # sdedit/gt edit
    ap.add_argument('--drop-frac', type=float, default=0.3)
    ap.add_argument('--add-ghosts', type=int, default=3)
    ap.add_argument('--edit-noise-scale', type=float, default=0.1)
    ap.add_argument('--alpha', type=float, default=0.5)
    # outputs
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/infer_refine')
    ap.add_argument('--thr', type=float, default=0.05)
    ap.add_argument('--nms-meters', type=float, default=0.0)
    ap.add_argument('--keep-only-proposal', action='store_true',
                    help='Only keep refined slots that were present in the input proposals (filter by proposal mask).')
    ap.add_argument('--topk', type=int, default=0,
                    help='Keep at most top-K slots by presence probability after NMS/filters (0=keep all).')
    ap.add_argument('--debug', action='store_true', help='打印调试信息（bounds、mask 含义、归一化范围等）')
    ap.add_argument('--no-clip-init', action='store_true', help='不对初始 proposal 进行 [-1,1] 裁剪（默认裁剪）')
    # Force-field debug visualization
    ap.add_argument('--force-field', action='store_true', help='输出力场诊断图（蓝:输入, 红箭头:实际移动, 绿箭头:理想GT方向）')
    args = ap.parse_args()

    set_seed(0)
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}

    # Load vectors and bounds (STRICT: static GT bounds)
    agg = load_pickle(osp.join(args.agg_pred_root, f'{args.scene}.pkl'))
    gt = load_pickle(osp.join(args.static_root, f'{args.scene}.pkl'))
    bounds = gt.get('bounds')
    if bounds is None:
        raise RuntimeError('bounds-missing in static GT pickle; run cropper to write canonical bounds')

    x_prop, m_prop, labs = pack_vectors_to_slots(agg, bounds, budgets, num_points=P, num_queries=N)
    # 可选：对初始 proposal 进行安全裁剪，避免超出 [-1,1] 造成 ODE 爆散
    if not args.no_clip_init:
        x_prop = np.clip(x_prop, -1.0, 1.0)
    try:
        x_gt, m_gt, present_gt = pack_gt_to_slots(gt, bounds, budgets, num_points=P, num_queries=N)
    except Exception:
        x_gt, m_gt, present_gt = None, None, None

    # 调试输出：bounds 对齐与归一化范围
    if bool(getattr(args, 'debug', False)):
        agg_bounds = agg.get('bounds', None)
        print(f"[debug] static bounds={bounds}")
        if agg_bounds is not None:
            print(f"[debug] agg    bounds={agg_bounds}")
        print(f"[debug] x_prop range: min={float(x_prop.min()):.4f}, max={float(x_prop.max()):.4f}")
        if m_prop is not None:
            # m=True 表示 padding（按当前实现），统计有效点与无效点占比
            valid_ratio = float((~m_prop).mean())
            pad_ratio = float(m_prop.mean())
            print(f"[debug] m_prop: valid_ratio={valid_ratio:.4f}, pad_ratio={pad_ratio:.4f}")

    # Build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = RasterEncoder(out_dim=256).to(device)
    base = SlotMLPWithTime(P=P, hidden=256, out_points=P, t_dim=64, num_slots=N, sem_classes=3).to(device)
    net = EDMPrecondRefine(base, sigma_data=1.0).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if 'encoder' in ckpt:
        enc.load_state_dict(ckpt['encoder'], strict=False)
    if 'net' in ckpt:
        base.load_state_dict(ckpt['net'], strict=False)
    enc.eval(); net.eval()

    # Prepare raster condition and start state
    # Match training: letterbox 10_render_gt to 1024x1024 before encoding (parity with train_edm_full)
    from global_diffusion_map.refine.single_scene_overfit import load_raster_png
    ras_raw = load_raster_png(osp.join(args.rendered_root, args.scene, '10_render_gt.png'))  # CHW RGB [0,1]
    canvas_bgr, _, _, _ = letterbox(ras_raw, 1024, (1024, 1024))  # HWC BGR uint8
    canvas_rgb = canvas_bgr[:, :, ::-1].astype(np.float32) / 255.0
    raster_enc = np.transpose(canvas_rgb, (2, 0, 1))  # CHW RGB [0,1]
    rv = enc(torch.from_numpy(raster_enc[None, ...]).to(device))

    # start state（SDEdit：若提供 start_sigma>0，则在 proposal 上加小噪声并从该步起跑）
    start_sigma = float(getattr(args, 'start_sigma', -1.0))
    if args.start == 'proposal':
        if start_sigma > 0.0:
            noise = np.random.normal(size=x_prop.shape).astype(np.float32)
            # 仅对有效点位加噪（避免对 padding/空槽注入噪声导致“鬼线条”）
            if m_prop is not None:
                valid_pts = (~m_prop).astype(np.float32)[..., None]  # [N,P,1]
                noise = noise * valid_pts
            x_t = np.clip(x_prop + start_sigma * noise, -1.0, 1.0)
        else:
            x_t = x_prop.copy()
        start_labels = labs
    elif args.start == 'gt_noise':
        if x_gt is None:
            raise RuntimeError('start=gt_noise requires GT pack (x_gt)')
        noise = np.random.normal(size=x_gt.shape).astype(np.float32)
        # 同样仅对有效 GT 点位加噪
        if m_gt is not None:
            valid_pts = (~m_gt).astype(np.float32)[..., None]
            noise = noise * valid_pts
        a = float(args.alpha)
        x_t = np.clip((1.0 - a) * x_gt + a * noise, -1.0, 1.0)
        start_labels = label_order_from_budgets(budgets)
    elif args.start == 'noise':
        x_t = np.random.normal(size=x_prop.shape).astype(np.float32)
        start_labels = labs
    else:
        # blend/proposal_edit/gt_edit not implemented in this minimal path
        x_t = x_prop.copy()
        start_labels = labs

    # Run EDM multistep (no per-step/state visualization code retained)
    x_torch = torch.from_numpy(x_t[None, ...]).to(device)
    sigmas = karras_schedule(max(1, int(args.steps)), float(args.sigma_min), float(args.sigma_max), float(args.rho)).to(device)
    # 若提供 start_sigma>0：丢弃所有大于 start_sigma 的前序步，使采样从较小噪声开始
    if start_sigma > 0.0 and sigmas.numel() > 0:
        # schedule 为降序（从 sigma_max → sigma_min）
        # 找到第一个 <= start_sigma 的索引，从此开始执行
        mask = sigmas <= (start_sigma + 1e-6)
        if bool(mask.any()):
            idx = int(torch.nonzero(mask, as_tuple=False)[0].item())
        else:
            idx = int(sigmas.numel() - 1)
        sigmas = sigmas[idx:]
    with torch.no_grad():
        coords, logits, _preds, _states = edm_unrolled_train(net, x_torch, rv, sigmas, second_order=bool(args.second_order))
    coords_np = coords.detach().cpu().numpy()[0]
    logits_np = logits.detach().cpu().numpy()[0]
    # last-step semantic logits if available
    sem_pred_np = None
    if isinstance(_preds, (list, tuple)) and len(_preds) > 0:
        last = _preds[-1]
        if isinstance(last, (list, tuple)) and len(last) >= 3 and last[2] is not None:
            sem_logits = last[2]  # [B,N,C]
            sem_logits_np = sem_logits.detach().cpu().numpy()[0]
            sem_pred_np = np.argmax(sem_logits_np, axis=-1).astype(np.int64)  # [N]

    # Save outputs
    out_dir = osp.join(args.out_root, args.scene)
    os.makedirs(out_dir, exist_ok=True)
    # final overlay (red predicted, white GT when可用)
    probs = 1.0 / (1.0 + np.exp(-logits_np.reshape(-1)))
    pred_keep = np.where(probs >= float(args.thr))[0].tolist()
    # optional NMS in meters (greedy by slot prob, distance on poly centers)
    nms_m = float(getattr(args, 'nms_meters', 0.0))
    if nms_m > 0.0 and len(pred_keep) > 1:
        minx, miny, maxx, maxy = [float(v) for v in bounds]
        Wm = max(maxx - minx, 1e-6); Hm = max(maxy - miny, 1e-6)
        cand = pred_keep[:]
        # sort by descending prob
        cand.sort(key=lambda i: float(probs[i]), reverse=True)
        centers = {}
        def _center_m(i: int):
            if i in centers:
                return centers[i]
            xy = coords_np[i]  # [P,2] normalized
            xm = (xy[:, 0] + 1.0) * 0.5 * Wm + minx
            ym = (xy[:, 1] + 1.0) * 0.5 * Hm + miny
            c = np.array([xm.mean(), ym.mean()], dtype=np.float32)
            centers[i] = c
            return c
        kept = []
        for i in cand:
            ci = _center_m(i)
            ok = True
            for j in kept:
                cj = _center_m(j)
                if float(np.linalg.norm(ci - cj)) < nms_m:
                    ok = False; break
            if ok:
                kept.append(i)
        pred_keep = kept
    # optional: keep only proposal-present slots
    if bool(getattr(args, 'keep_only_proposal', False)) and (m_prop is not None):
        prop_present = (~m_prop).any(axis=1)
        pred_keep = [i for i in pred_keep if (i < prop_present.shape[0] and bool(prop_present[i]))]
    # optional: top-K by prob
    topk = int(getattr(args, 'topk', 0))
    if topk > 0 and len(pred_keep) > topk:
        pred_keep.sort(key=lambda i: float(probs[i]), reverse=True)
        pred_keep = pred_keep[:topk]
    pred_list = [coords_np[i] for i in pred_keep]
    gt_list: List[np.ndarray] = []
    if x_gt is not None and present_gt is not None:
        keep = np.where((present_gt.reshape(-1) > 0))[0].tolist()
        gt_list = [x_gt[i] for i in keep]
    overlay_on_raster(osp.join(out_dir, 'refine_overlay.png'), ras_raw, bounds, gt_list, pred_list)
    # clean overlay（仅绘制精炼结果，去掉 GT 以降低视觉噪声）
    try:
        overlay_on_raster(osp.join(out_dir, 'refine_overlay_clean.png'), ras_raw, bounds, [], pred_list)
    except Exception:
        pass

    # force-field diagnostic: visualize movement vs ideal GT direction
    if bool(getattr(args, 'force_field', False)) and (x_gt is not None):
        try:
            import cv2
            # Prepare letterboxed canvas (same as overlay)
            canvas, scale, H0, W0 = letterbox(ras_raw, cond_max_side=1024, cond_fixed_size=(1024, 1024))
            # Build tensors for matcher
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            xin = torch.from_numpy(x_prop).to(dev).float()  # [N,P,2]
            xgt = torch.from_numpy(x_gt).to(dev).float()    # [M,P,2]
            mgt = torch.from_numpy(m_gt).to(dev).bool()     # [M,P]
            pgt = torch.ones((xgt.shape[0],), dtype=torch.long, device=dev)
            # Disable direction term to avoid eigen-decomposition spikes during inference
            pairs = gpu_greedy_match(xin, xgt, mgt, pgt, w_center=1.0, w_dir=0.0, w_pw=0.0, use_chamfer=True)
            # Helper: center in pixels
            minx, miny, maxx, maxy = [float(v) for v in bounds]
            def center_px(arr_norm: np.ndarray) -> tuple[int,int]:
                xy = denorm_xy(arr_norm, bounds)
                pts = to_px(xy, bounds, W0, H0)
                cen = pts.mean(axis=0).astype(np.float32) * scale
                return int(round(cen[0])), int(round(cen[1]))
            # Draw GT for context (thin green)
            for g in gt_list:
                pts = to_px(denorm_xy(g, bounds), bounds, W0, H0)
                pts = (pts.astype(np.float32) * scale).round().astype(np.int32)
                cv2.polylines(canvas, [pts], False, (0,255,0), 1)
            # Draw movement arrows for matched pairs
            for (pi, gj) in pairs:
                if not (0 <= pi < x_prop.shape[0] and 0 <= gj < x_gt.shape[0]):
                    continue
                c_in  = center_px(x_prop[pi])
                c_out = center_px(coords_np[pi])
                c_gt  = center_px(x_gt[gj])
                # Blue input center
                cv2.circle(canvas, c_in, 2, (255,0,0), -1)
                # Red arrow: actual motion (input -> refined)
                cv2.arrowedLine(canvas, c_in, c_out, (0,0,255), 2, tipLength=0.25)
                # Green dotted: ideal (input -> GT)
                # draw as small segments for dotted effect
                dx = c_gt[0] - c_in[0]; dy = c_gt[1] - c_in[1]
                seg = max(1, int(np.hypot(dx, dy) // 12))
                for k in range(seg):
                    t0 = k / seg; t1 = min(1.0, (k + 0.5) / seg)
                    p0 = (int(round(c_in[0] + dx * t0)), int(round(c_in[1] + dy * t0)))
                    p1 = (int(round(c_in[0] + dx * t1)), int(round(c_in[1] + dy * t1)))
                    cv2.line(canvas, p0, p1, (0,255,0), 2)
            cv2.putText(canvas, 'Force Field: red=actual, green=ideal, blue=input', (10, 1010), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(canvas, 'Force Field: red=actual, green=ideal, blue=input', (10, 1010), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imwrite(osp.join(out_dir, 'force_field.png'), canvas)
        except Exception as e:
            if bool(getattr(args, 'debug', False)):
                print(f'[warn] force-field viz failed: {e}')

    # class-annotated refined overlay（根据语义头预测的类别为槽位着色与编号标注）
    try:
        mask_ref = np.ones((coords_np.shape[0], coords_np.shape[1]), dtype=bool)
        for i in pred_keep:
            mask_ref[i, :] = False
        labels_ref = sem_pred_np if sem_pred_np is not None else None
        overlay_slots_annot(
            osp.join(out_dir, 'refine_overlay_cls.png'),
            ras_raw, bounds,
            slots=coords_np, mask=mask_ref,
            labels=labels_ref,
            title='Refined (sem cls) + GT',
            gt_slots=x_gt if x_gt is not None else None,
            gt_mask=m_gt if x_gt is not None and m_gt is not None else None,
        )
    except Exception:
        pass

    # pure input proposal overlay（未去噪，仅显示输入 proposal 与 GT）
    try:
        prop_present = None
        if m_prop is not None:
            # present if any point is valid (not masked)
            prop_present = (~m_prop).any(axis=1)
        if prop_present is None:
            prop_idx = list(range(x_prop.shape[0]))
        else:
            prop_idx = np.where(prop_present)[0].tolist()
        prop_list = [x_prop[i] for i in prop_idx]
        overlay_on_raster(osp.join(out_dir, 'input_overlay.png'), ras_raw, bounds, gt_list, prop_list)
    except Exception:
        pass
    # refined pkl (apply same NMS/keep set before save)
    if len(pred_keep) > 0:
        coords_keep = np.stack([coords_np[i] for i in pred_keep], axis=0)
        logits_keep = np.stack([logits_np[i] for i in pred_keep], axis=0) if logits_np.ndim > 1 else logits_np[pred_keep]
        labels_np = np.asarray(start_labels, dtype=np.int64)
        labels_keep = labels_np[pred_keep] if labels_np.ndim == 1 else labels_np
        save_refined_pkl(osp.join(out_dir, f'{args.scene}.pkl'), coords_keep, logits_keep, labels_keep, bounds, thr=float(args.thr))
    else:
        # fallback: save nothing but keep structure
        save_refined_pkl(osp.join(out_dir, f'{args.scene}.pkl'), coords_np[:0], logits_np[:0], np.asarray([], dtype=np.int64), bounds, thr=float(args.thr))
    print(f'[ok] inference done. Overlay: {osp.join(out_dir, "refine_overlay.png")}  PKL: {osp.join(out_dir, f"{args.scene}.pkl")}')


if __name__ == '__main__':
    main()
