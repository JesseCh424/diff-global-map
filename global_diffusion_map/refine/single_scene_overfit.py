#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import sys
import os.path as osp
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from global_diffusion_map.refine.dataset_refine import RefineCaps, _normalize_xy, _uniform_resample, pack_gt_to_slots, pack_vectors_to_slots
from global_diffusion_map.refine.loss_refine import criterion, HungarianMatcher
from global_diffusion_map.refine.model_refine import DiffusionRefiner
from global_diffusion_map.refine.single_scene_dataset import SingleSceneDataset


def set_seed(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pickle(path: str) -> Dict[str, Any]:
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_bounds_from_any(d: Dict[str, Any]) -> List[float]:
    """Strict bounds fetch: only use serialized bounds; do not fallback.

    Historically we computed bounds from raw vectors when pickles lacked
    a canonical 'bounds' entry. Per updated policy, this fallback must be
    disabled to avoid canvas drift. This helper now only returns d['bounds']
    if present; otherwise raises.
    """
    if 'bounds' in d and d['bounds'] is not None:
        b = d['bounds']
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    raise RuntimeError('bounds-missing: expected canonical bounds in pickle')


    


class RasterEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        # Simple 3-layer CNN with GroupNorm for BS=1 stability
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GroupNorm(4, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, 3, 2, 1), nn.GroupNorm(16, out_dim), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        f = self.enc(x)
        f = self.pool(f).flatten(1)
        return f  # [B,C]




class SlotMLP(nn.Module):
    def __init__(self, P: int, hidden: int = 256, out_points: int = 30) -> None:
        super().__init__()
        self.P = int(P)
        self.feat = nn.Sequential(
            nn.Linear(P * 2 + 256, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = DiffusionRefiner(hidden_dim=hidden, num_points=out_points)

    def forward(self, x_slots: torch.Tensor, raster_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_slots: [B,N,P,2], raster_vec: [B,256]
        B, N, P, _ = x_slots.shape
        xs = x_slots.view(B, N, P * 2)
        rv = raster_vec[:, None, :].expand(B, N, -1)
        feats = torch.cat([xs, rv], dim=-1)
        feats = self.feat(feats)
        coords, logits = self.head(feats)
        # tanh to keep coords in [-1,1]
        coords = torch.tanh(coords)
        return coords, logits


def load_raster_png(path: str, out_size: Tuple[int, int] | None = None) -> np.ndarray:
    """Load raster without pre-resize.
    Keep original size; downstream letterbox() mirrors check_cond_alignment.
    """
    img = Image.open(path).convert('RGB')
    if out_size is not None:
        img = img.resize((out_size[1], out_size[0]), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)  # [3,H,W]


def denorm_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    w = max(maxx - minx, 1e-6); h = max(maxy - miny, 1e-6)
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0] + 1.0) * 0.5 * w + minx
    out[..., 1] = (xy[..., 1] + 1.0) * 0.5 * h + miny
    return out


def letterbox(img: np.ndarray, cond_max_side: int | None, cond_fixed_size: Tuple[int, int] | None) -> Tuple[np.ndarray, float, int, int]:
    """Resize + letterbox to cond_fixed_size.
    Returns: (canvas_bgr, total_scale, orig_h, orig_w).
    Paste at (0,0) to mirror training dataset和推理。
    """
    import cv2
    # img: CHW in RGB [0,1]
    img_rgb = (img.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    h, w = img_rgb.shape[:2]
    if cond_max_side is not None:
        ms = max(h, w)
        if ms > 0 and ms > cond_max_side:
            s = float(cond_max_side) / float(ms)
            nh = max(1, int(round(h * s)))
            nw = max(1, int(round(w * s)))
            img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            h, w = img_rgb.shape[:2]
            s_pre = s
        else:
            s_pre = 1.0
    else:
        s_pre = 1.0
    s = 1.0
    if cond_fixed_size is not None:
        tgt_h, tgt_w = int(cond_fixed_size[0]), int(cond_fixed_size[1])
        s = min(float(tgt_w) / float(w), float(tgt_h) / float(h))
        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        if (nw, nh) != (w, h):
            img_r = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        else:
            img_r = img_rgb
        canvas_rgb = np.zeros((tgt_h, tgt_w, 3), dtype=np.uint8)
        canvas_rgb[:nh, :nw] = img_r
        # convert to BGR for cv2 drawing
        canvas_bgr = canvas_rgb[..., ::-1].copy()
        return canvas_bgr, s_pre * s, int(img.shape[1]), int(img.shape[2])
    # No fixed size: still convert color order
    canvas_bgr = img_rgb[..., ::-1].copy()
    return canvas_bgr, s_pre * s, int(img.shape[1]), int(img.shape[2])


def to_px(arr: np.ndarray, bounds: Sequence[float], W: int, H: int) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    Sx = W / max(maxx - minx, 1e-6)
    Sy = H / max(maxy - miny, 1e-6)
    xs = np.clip(np.round((arr[:, 0] - minx) * Sx), 0, W - 1)
    ys = np.clip(np.round((maxy - arr[:, 1]) * Sy), 0, H - 1)
    return np.stack([xs, ys], 1).astype(np.int32)


def overlay_on_raster(out_path: str, raster_chw: np.ndarray, bounds: Sequence[float],
                      gt_pack: np.ndarray, pred_pack: np.ndarray,
                      cond_max_side: int = 1024, cond_fixed_size: Tuple[int, int] = (1024, 1024),
                      thickness_px: int = 2, highlight_edges: bool = True) -> None:
    import cv2
    canvas, scale, H0, W0 = letterbox(raster_chw, cond_max_side, cond_fixed_size)
    H, W = canvas.shape[:2]
    # 轻微降低背景亮度以凸显叠加线条
    canvas = (canvas.astype(np.float32) * 0.9).astype(np.uint8)
    if highlight_edges:
        # 用 Canny 提取底图边缘，叠加成细的淡青色骨架，帮助阅读道路结构
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_color = (255, 255, 0)  # BGR: cyan
        canvas[edges.astype(bool)] = edge_color
    # 绘制 letterbox 有效内容边框（矩形），便于直观看到内容是否未占满画布
    cw = int(round(W0 * scale)); ch = int(round(H0 * scale))
    cw = max(1, min(W, cw)); ch = max(1, min(H, ch))
    cv2.rectangle(canvas, (0, 0), (cw - 1, ch - 1), (255, 0, 255), 1)
    # 双笔触绘制：先黑色粗描，再彩色细描，保证在粗厚道路上可见
    def _draw_polyline(arr_list: List[np.ndarray], color: Tuple[int, int, int]) -> None:
        for arr in arr_list:
            if np.allclose(arr, 0):
                continue
            xy = denorm_xy(arr, bounds)
            # 先用原始栅格尺寸投影，再乘 letterbox 缩放
            pts = to_px(xy, bounds, W0, H0)
            pts = (pts.astype(np.float32) * scale).round().astype(np.int32)
            cv2.polylines(canvas, [pts], False, (0, 0, 0), thickness=max(1, thickness_px + 2))
            cv2.polylines(canvas, [pts], False, color, thickness=max(1, thickness_px))

    # GT：用白色提高可见性；Pred：红色
    _draw_polyline([a for a in gt_pack], (255, 255, 255))
    _draw_polyline([a for a in pred_pack], (0, 0, 255))
    for arr in gt_pack:
        pass
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)




def _tangent_noise(arr: np.ndarray, sigma_t: float, sigma_n: float) -> np.ndarray:
    """为一条折线生成切向/法向噪声（逐点局部切向）。
    arr: [P,2] in [-1,1]. 返回与 arr 同形的噪声偏移。
    """
    P = arr.shape[0]
    out = np.zeros_like(arr, dtype=np.float32)
    if P <= 1:
        return out
    for k in range(P):
        k0 = max(0, k - 1)
        k1 = min(P - 1, k + 1)
        v = arr[k1] - arr[k0]
        nrm = float(np.linalg.norm(v) + 1e-8)
        t = v / nrm if nrm > 0 else np.array([1.0, 0.0], dtype=np.float32)
        n = np.array([-t[1], t[0]], dtype=np.float32)
        zt = np.random.normal(scale=sigma_t)
        zn = np.random.normal(scale=sigma_n)
        out[k] = zt * t + zn * n
    return out


def overlay_slots_annot(out_path: str,
                        raster_chw: np.ndarray,
                        bounds: Sequence[float],
                        slots: np.ndarray,        # [N,P,2] normalized
                        mask: np.ndarray,         # [N,P]
                        labels: Optional[np.ndarray] = None,  # [N] (MapTR labels)
                        title: Optional[str] = None,
                        cond_max_side: int = 1024,
                        cond_fixed_size: Tuple[int, int] = (1024, 1024),
                        thickness_px: int = 2,
                        gt_slots: Optional[np.ndarray] = None, # [N_gt,P,2] normalized
                        gt_mask: Optional[np.ndarray] = None   # [N_gt,P]
                        ) -> None:
    """在底图上绘制带编号的槽位曲线（仅存在槽位），并按类别上色。
    颜色（BGR）：divider(0)=红(0,0,255)，ped(1)=黄(0,255,255)，boundary(2)=青(255,255,0)。
    """
    import cv2
    canvas, scale, H0, W0 = letterbox(raster_chw, cond_max_side, cond_fixed_size)
    H, W = canvas.shape[:2]
    canvas = (canvas.astype(np.float32) * 0.9).astype(np.uint8)
    # Draw a thin magenta rectangle around the active letterbox content area for diagnostics
    try:
        cw = int(round(W0 * float(scale)))
        ch = int(round(H0 * float(scale)))
        cw = max(1, min(W, cw))
        ch = max(1, min(H, ch))
        import cv2
        cv2.rectangle(canvas, (0, 0), (cw - 1, ch - 1), (255, 0, 255), 1)
        cv2.putText(canvas, f"scale={scale:.4f} src={W0}x{H0}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        # Save meta alongside the image for auditability
        meta_path = out_path.replace('.png', '.meta.txt')
        with open(meta_path, 'w') as mf:
            b = [float(v) for v in bounds]
            mf.write(f"W0={W0} H0={H0}\n")
            mf.write(f"W={W} H={H}\n")
            mf.write(f"scale={scale}\n")
            mf.write(f"content_w={cw} content_h={ch}\n")
            mf.write(f"bounds=[{b[0]},{b[1]},{b[2]},{b[3]}]\n")
            mf.write(f"cond_max_side={cond_max_side} cond_fixed_size={cond_fixed_size}\n")
    except Exception:
        pass

    color_map = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 255, 0)}
    name_map = {0: 'D', 1: 'P', 2: 'B'}
    # Optionally draw GT first (white) for reference
    if gt_slots is not None and gt_mask is not None:
        for i in range(gt_slots.shape[0]):
            if gt_mask[i].all():
                continue
            arr_g = gt_slots[i]
            xy_g = denorm_xy(arr_g, bounds)
            pts_g = to_px(xy_g, bounds, W0, H0)
            pts_g = (pts_g.astype(np.float32) * scale).round().astype(np.int32)
            cv2.polylines(canvas, [pts_g], False, (0, 0, 0), thickness=max(1, thickness_px + 2))
            cv2.polylines(canvas, [pts_g], False, (255, 255, 255), thickness=max(1, thickness_px))

    idx = 0
    for i in range(slots.shape[0]):
        if mask[i].all():
            continue
        # 优先按标签上色；未知/越界时使用通用颜色（红）而非跳过
        if labels is not None:
            lab = int(labels[i])
        else:
            lab = 2
        col = color_map.get(lab, (0, 0, 255))
        arr = slots[i]
        xy = denorm_xy(arr, bounds)
        pts = to_px(xy, bounds, W0, H0)
        pts = (pts.astype(np.float32) * scale).round().astype(np.int32)
        # 双笔触
        cv2.polylines(canvas, [pts], False, (0, 0, 0), thickness=max(1, thickness_px + 2))
        cv2.polylines(canvas, [pts], False, col, thickness=max(1, thickness_px))
        # 标注编号
        cen = pts.mean(axis=0).astype(int)
        tag = f"{name_map.get(lab,'S')}{i:02d}"
        cv2.putText(canvas, tag, (int(cen[0]), int(cen[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, tag, (int(cen[0]), int(cen[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)
        idx += 1
    if title:
        cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)


def zero_loss_sanity(pred_coords: torch.Tensor, pred_logits: torch.Tensor,
                     tgt_coords: torch.Tensor, tgt_mask: torch.Tensor, tgt_present: torch.Tensor) -> Tuple[float, float]:
    from global_diffusion_map.refine.loss_refine import criterion as crit
    out = crit(pred_coords, pred_logits, tgt_coords, tgt_mask, tgt_present)
    return float(out['loss_reg'].item()), float(out['loss_cls'].item())


def main() -> None:
    ap = argparse.ArgumentParser(description='Single-scene overfit sanity check for refinement')
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--rendered-root', required=True)
    ap.add_argument('--agg-pred-root', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out-root', default='global_diffusion_map/refine/work_dirs/overfit')
    ap.add_argument('--viz-prob-thr', type=float, default=0.5, help='ep_xxxx 可视化时按 presence 概率过滤预测槽位')
    # 可视化噪声调节
    ap.add_argument('--prop-jitter', type=float, default=0.1, help='overlay_prop_on_raster 的抖动强度（仅存在槽位）')
    ap.add_argument('--alpha', type=float, default=0.03, help='overlay_xK_on_raster 的混合比例（存在槽位 xK = alpha*noise + (1-alpha)*GT）')
    ap.add_argument('--prop-jitter-tan', type=float, default=0.01, help='可选：切向噪声强度（优先于 --prop-jitter）')
    ap.add_argument('--prop-jitter-nor', type=float, default=0.06, help='可选：法向噪声强度（默认更小）')
    ap.add_argument('--xk-tan', type=float, default=None, help='xK 的切向噪声强度')
    ap.add_argument('--xk-nor', type=float, default=None, help='xK 的法向噪声强度')
    args = ap.parse_args()

    set_seed(0)
    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    P = int(stats.get('M', 20))
    N = int(stats.get('num_queries', 64))
    budgets = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 30, 2: 22}).items()}
    caps = RefineCaps(num_queries=N, num_points=P)

    # Load scene data
    gt_pkl = osp.join(args.static_root, f'{args.scene}.pkl')
    gt = load_pickle(gt_pkl)
    bounds = gt.get('bounds')
    if bounds is None:
        # AGENTS.md: 优先 static 的 canonical bounds；否则用聚合预测的 bounds；仍缺失则从聚合预测向量计算（与 check_cond_alignment 行为一致）
        ap_pkl = osp.join(args.agg_pred_root, f'{args.scene}.pkl')
        agg = load_pickle(ap_pkl)
        b2 = agg.get('bounds', None)
        if b2 is None:
            raise RuntimeError(
                'bounds-missing: aggregated pred pickle has no bounds. '
                'Per policy, do not fallback to computed bounds.')
        bounds = b2
    # Raster
    raster_path = osp.join(args.rendered_root, args.scene, '10_render_gt.png')
    raster = load_raster_png(raster_path)  # [3,H,W]

    # Pack GT to slots
    gt_pack, gt_mask, gt_present = pack_gt_to_slots(
        gt, bounds, budgets, num_points=P, num_queries=N
    )
    # Pack aggregated proposals to slots（无噪声，用于“原始提案对齐”检查）
    agg = load_pickle(osp.join(args.agg_pred_root, f'{args.scene}.pkl'))
    prop_pack, prop_mask, prop_labels = pack_vectors_to_slots(
        agg, bounds, budgets, num_points=P, num_queries=N
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Model
    enc = RasterEncoder(out_dim=256).to(device)
    net = SlotMLP(P=P, hidden=256, out_points=P).to(device)
    opt = optim.AdamW(list(enc.parameters()) + list(net.parameters()), lr=args.lr)

    out_dir = osp.join(args.out_root, args.scene)
    viz_dir = osp.join(out_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    # Training loop
    # 1) Overlay check：GT/Proposal 叠加到栅格（对齐最关键）
    overlay_on_raster(osp.join(viz_dir, 'overlay_gt_on_raster.png'), raster, bounds, gt_pack, np.zeros_like(gt_pack))
    # 原始提案（无噪声）叠加检查：红=原始提案，白=GT
    path_prop_raw = osp.join(viz_dir, 'overlay_prop_raw_on_raster.png')
    overlay_on_raster(path_prop_raw, raster, bounds, gt_pack, prop_pack)
    # 原始提案（带编号/类别）可视化，便于区分存在槽位与类别
    overlay_slots_annot(osp.join(viz_dir, 'overlay_prop_raw_slots_annot.png'), raster, bounds,
                        prop_pack, prop_mask, prop_labels, title='Raw Proposal Slots')
    # 造一个强扰动 proposal 看是否与栅格对齐（仅对“存在”的槽位加噪，避免把空槽画成噪点云）
    present_idx = ~gt_mask.all(axis=1)  # [N]
    # Proposal 抖动：优先按切/法向分量生成，否则使用各向同性高斯
    if args.prop_jitter_tan is not None:
        sig_t = float(args.prop_jitter_tan)
        sig_n = float(args.prop_jitter_nor if args.prop_jitter_nor is not None else max(1e-3, sig_t * 0.2))
        prop_vis = gt_pack.copy()
        for i in range(N):
            if not present_idx[i]:
                continue
            prop_vis[i] = np.clip(gt_pack[i] + _tangent_noise(gt_pack[i], sig_t, sig_n), -1.0, 1.0)
    else:
        noise_prop = np.random.normal(scale=float(args.prop_jitter), size=gt_pack.shape).astype(np.float32)
        prop_vis = gt_pack.copy()
        prop_vis[present_idx] = np.clip(gt_pack[present_idx] + noise_prop[present_idx], -1.0, 1.0)
    path_prop_noise = osp.join(viz_dir, 'overlay_prop_on_raster.png')
    overlay_on_raster(path_prop_noise, raster, bounds, gt_pack, prop_vis)
    # 使用 proposal 的 mask（prop_mask）而非 gt_mask，确保只显示“存在的提案槽位”
    overlay_slots_annot(osp.join(viz_dir, 'overlay_prop_on_raster_slots_annot.png'), raster, bounds,
                        prop_vis, prop_mask, labels=prop_labels, title='Noisy Proposal (existing slots only)')

    # 2) Normalization check：打印范围（必须在 [-1,1]）
    def _rng(name: str, arr: np.ndarray) -> None:
        print(f"[norm] {name}: min={arr.min():.3f} max={arr.max():.3f}")
    _rng('gt_pack', gt_pack)
    _rng('prop_vis', prop_vis)
    # 打印存在槽位统计（GT/Proposal）与类占比
    gt_exist = int((~gt_mask.all(axis=1)).sum())
    prop_exist = int((~prop_mask.all(axis=1)).sum())
    print(f"[exist] gt slots={gt_exist}/{N}  prop slots={prop_exist}/{N}")
    # 按 MapTR 标签（0:divider,1:ped,2:boundary）统计 proposal 槽位
    if prop_exist:
        import collections
        cnt = collections.Counter(int(l) for l in prop_labels[:N] if not prop_mask[int(np.where(prop_labels==l)[0][0] if isinstance(prop_labels, np.ndarray) else 0)].all())
        # 更稳妥：逐槽位检查 mask
        per_cls = {0:0,1:0,2:0}
        for i in range(N):
            if not prop_mask[i].all():
                per_cls[int(prop_labels[i])] += 1
        print(f"[exist-per-class] proposal slots: divider={per_cls.get(0,0)} ped={per_cls.get(1,0)} boundary={per_cls.get(2,0)}")

    # 3) Zero-loss sanity：Prediction=GT 时，reg≈0，cls 很小
    pc0 = torch.from_numpy(gt_pack[None, ...]).to(device)
    # 旧版：全部槽位给高置信度（用于零损 sanity 的最简版本）
    pl0 = torch.full((1, N, 1), 6.0, device=device)  # sigmoid≈0.997 代表正样本
    tm0 = torch.from_numpy(gt_mask[None, ...]).to(device)
    tp0 = torch.from_numpy(gt_present[None, ...]).to(device)
    z_reg, z_cls = zero_loss_sanity(pc0, pl0, pc0, tm0, tp0)
    print(f"[zero-loss] reg={z_reg:.8f} cls={z_cls:.8f}")

    # 4) BatchNorm trap：将任意 BatchNorm 设为 eval（本模型使用 GroupNorm，安全）
    def freeze_bn(m: nn.Module) -> None:
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
    enc.apply(freeze_bn); net.apply(freeze_bn)

    # 5) Noise visual check：构造 xK（部分加噪）并叠加
    alpha = float(args.alpha)  # 噪声比例（0.0=原样，1.0=纯噪）
    # xK 也用切/法向噪声，优先使用 --xk-tan/--xk-nor，否则退化为各向同性
    xK = gt_pack.copy()
    if args.xk_tan is not None:
        sig_t = float(args.xk_tan)
        sig_n = float(args.xk_nor if args.xk_nor is not None else max(1e-3, sig_t * 0.2))
        for i in range(N):
            if not present_idx[i]:
                continue
            noised = gt_pack[i] + _tangent_noise(gt_pack[i], sig_t, sig_n)
            xK[i] = np.clip((1.0 - alpha) * gt_pack[i] + alpha * noised, -1.0, 1.0)
    else:
        noise_xk = np.random.normal(size=gt_pack.shape).astype(np.float32)
        xK[present_idx] = np.clip(alpha * noise_xk[present_idx] + (1.0 - alpha) * gt_pack[present_idx], -1.0, 1.0)
    path_xk = osp.join(viz_dir, 'overlay_xK_on_raster.png')
    overlay_on_raster(path_xk, raster, bounds, gt_pack, xK)

    # 5.1) 定量化噪声是否合适的度量（中心距离/简化 Chamfer/方向余弦）
    def slot_labels_from_budgets(budgets: Dict[int, int], N: int) -> np.ndarray:
        # MapTR labels顺序：divider(0)×cap + ped(1)×cap + boundary(2)×cap
        order: List[int] = []
        for orig in (1, 0, 2):
            cap = int(budgets.get(orig, 0))
            lab = 0 if orig == 1 else (1 if orig == 0 else 2)
            order += [lab] * max(0, cap)
        arr = np.zeros((N,), dtype=np.int64)
        L = min(N, len(order))
        if L > 0:
            arr[:L] = np.asarray(order[:L], dtype=np.int64)
        return arr

    def principal_dir(xy: np.ndarray) -> np.ndarray:
        if xy.shape[0] < 2:
            return np.array([1.0, 0.0], dtype=np.float32)
        v = xy[-1] - xy[0]
        n = float(np.linalg.norm(v) + 1e-8)
        if n == 0.0:
            c = xy - xy.mean(0, keepdims=True)
            u, s, vh = np.linalg.svd(c, full_matrices=False)
            v = vh[0]
            n = float(np.linalg.norm(v) + 1e-8)
        return (v / n).astype(np.float32)

    def simple_chamfer(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        m = min(a.shape[0], b.shape[0], 16)
        ia = np.linspace(0, a.shape[0]-1, num=m).round().astype(int)
        ib = np.linspace(0, b.shape[0]-1, num=m).round().astype(int)
        aa, bb = a[ia], b[ib]
        Da = np.sqrt(((aa[:, None, :] - bb[None, :, :])**2).sum(-1))
        Db = Da.T
        return 0.5 * (Da.min(1).mean() + Db.min(1).mean())

    # 仅评估“存在槽位”的 xK vs GT
    exist_idx = np.where(~gt_mask.all(axis=1))[0]
    labels_slots = slot_labels_from_budgets(budgets, N)
    diag = float(np.sqrt(8.0))
    cdist, cd_cls = [], {0: [], 1: [], 2: []}
    chmf, ch_cls = [], {0: [], 1: [], 2: []}
    dcos, dc_cls = [], {0: [], 1: [], 2: []}
    for i in exist_idx:
        a = xK[i]; b = gt_pack[i]
        # 归一化空间的中心距离/Chamfer、方向差
        ca = a.mean(0); cb = b.mean(0)
        c = float(np.linalg.norm(ca - cb)) / diag
        cdist.append(c); cd_cls[labels_slots[i]].append(c)
        ch = float(simple_chamfer(a, b)) / diag
        chmf.append(ch); ch_cls[labels_slots[i]].append(ch)
        pa = principal_dir(a); pb = principal_dir(b)
        d = float(np.abs(np.dot(pa, pb)))  # cos 值
        dcos.append(d); dc_cls[labels_slots[i]].append(d)

    def Q(ls: List[float], q: float) -> float:
        return float(np.percentile(np.asarray(ls, dtype=np.float32), q)) if ls else 0.0
    print(f"[noise-metrics] center_norm p50={Q(cdist,50):.3f} p90={Q(cdist,90):.3f}  chamfer_norm p50={Q(chmf,50):.3f} p90={Q(chmf,90):.3f}  dir_cos p50={Q(dcos,50):.3f}")
    for cls_id, name in {0:'divider',1:'ped',2:'boundary'}.items():
        print(f"[noise-metrics:{name}] center_p50={Q(cd_cls[cls_id],50):.3f} chamfer_p50={Q(ch_cls[cls_id],50):.3f} dircos_p50={Q(dc_cls[cls_id],50):.3f}")

    # 进入训练（使用 DataLoader + Hungarian 匹配目标）
    ds = SingleSceneDataset(
        static_root=args.static_root,
        rendered_root=args.rendered_root,
        agg_pred_root=args.agg_pred_root,
        scene=args.scene,
        caps=caps,
        class_budgets=budgets,
        length=max(1000, args.epochs * 2),
        jitter_sigma_m=0.5,
        drop_rate=0.0,
        ghosts=0,
    )
    matcher = HungarianMatcher(w_center=1.0, w_dir=0.2, w_pw=0.5)
    enc.train(); net.train()
    iters = args.epochs
    for ep in range(1, iters + 1):
        sample = ds[ep - 1]
        x = sample['proposal'][None, ...].to(device)  # [1,N,P,2]
        r = sample['raster'][None, ...].to(device)    # [1,3,H,W]
        tgt_c = sample['tgt_coords'][None, ...].to(device)
        tgt_m = sample['tgt_mask'][None, ...].to(device)
        tgt_p = sample['tgt_present'][None, ...].to(device)

        # Forward
        rv = enc(r)
        pred_coords, pred_logits = net(x, rv)

        # 目标对齐：Hungarian 匹配（在 numpy 上跑，生成对齐后的 target）
        pc_np = pred_coords.detach().cpu().numpy()[0]  # [N,P,2]
        gt_np = tgt_c.detach().cpu().numpy()[0]
        gt_mask_np = tgt_m.detach().cpu().numpy()[0]
        # 仅使用 present==1 的 GT 做匹配；pred 用全部槽位
        valid_gt_idx = np.where(tgt_p.cpu().numpy()[0] > 0)[0].tolist()
        gt_list = [gt_np[j] for j in valid_gt_idx]
        # 若没有有效 GT，跳过匹配
        if len(gt_list) > 0:
            pairs = matcher(pc_np, np.stack(gt_list, axis=0))  # [(pi, gj_loc)]
            new_tgt = np.zeros_like(gt_np)
            new_msk = np.ones_like(gt_mask_np)
            new_pre = np.zeros_like(tgt_p.cpu().numpy()[0])
            for pi, gj_loc in pairs:
                gj = valid_gt_idx[gj_loc]
                new_tgt[pi] = gt_np[gj]
                new_msk[pi] = gt_mask_np[gj]
                new_pre[pi] = 1
            tgt_coords = torch.from_numpy(new_tgt[None, ...]).to(device)
            tgt_mask = torch.from_numpy(new_msk[None, ...]).to(device)
            tgt_present = torch.from_numpy(new_pre[None, ...]).to(device)
        else:
            tgt_coords, tgt_mask, tgt_present = tgt_c, tgt_m, tgt_p

        losses = criterion(pred_coords, pred_logits, tgt_coords, tgt_mask, tgt_present,
                           l1_weight=1.0, cls_weight=1.0)
        loss = losses['loss_cls'] + losses['loss_reg']
        opt.zero_grad(); loss.backward(); opt.step()

        if ep % 20 == 0 or ep == 1:
            print(f"[ep {ep:04d}] loss={loss.item():.6f} cls={losses['loss_cls'].item():.6f} reg={losses['loss_reg'].item():.6f}")
        if ep % 50 == 0 or ep == args.epochs:
            with torch.no_grad():
                pc = pred_coords.detach().cpu().numpy()[0]
                pl = pred_logits.detach().cpu().numpy()[0].reshape(-1)
                prob = 1.0 / (1.0 + np.exp(-pl))
                keep = np.where(prob >= float(args.viz_prob_thr))[0].tolist()
                pred_list = [pc[i] for i in keep]
                overlay_on_raster(osp.join(viz_dir, f'ep_{ep:04d}.png'), raster, bounds, gt_pack, pred_list)
                # also plot per-slot cls view (only slots with prob>=thr)
                m_pred = np.ones((N, P), dtype=bool)
                for i in keep:
                    m_pred[i] = False
                try:
                    overlay_slots_annot(
                        osp.join(viz_dir, f'ep_{ep:04d}_cls.png'),
                        raster,
                        bounds,
                        pc,
                        m_pred,
                        labels=labels_slots,
                        title=f'ep {ep:04d} (cls thr={float(args.viz_prob_thr)})'
                    )
                except Exception:
                    pass
                # 额外：可视化“原始提案 vs GT”的匹配标注，按类别统计
                # 使用 raw proposal（prop_pack）与 GT（gt_pack）进行一次匹配并绘制
                try:
                    # 准备仅存在槽位的数组
                    gt_exist_idx = np.where(~gt_mask.all(axis=1))[0]
                    prop_exist_idx = np.where(~prop_mask.all(axis=1))[0]
                    if gt_exist_idx.size > 0 and prop_exist_idx.size > 0:
                        A = prop_pack[prop_exist_idx]
                        B = gt_pack[gt_exist_idx]
                        pairs = matcher(A, B)
                        # 构造一张标注图：matched 用类别色，不匹配的提案灰色虚线
                        import cv2
                        canvas, scale, H0, W0 = letterbox(raster, 1024, (1024, 1024))
                        canvas = (canvas.astype(np.float32) * 0.9).astype(np.uint8)
                        color_map = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 255, 0)}
                        used_prop = set()
                        for pi, gj in pairs:
                            gslot = int(gt_exist_idx[gj])
                            pslot = int(prop_exist_idx[pi])
                            lab = int(prop_labels[pslot])
                            col = color_map.get(lab, (0, 0, 255))
                            # 画 prop
                            arr = prop_pack[pslot]
                            xy = denorm_xy(arr, bounds)
                            pts = to_px(xy, bounds, W0, H0)
                            pts = (pts.astype(np.float32) * scale).round().astype(np.int32)
                            cv2.polylines(canvas, [pts], False, (0, 0, 0), thickness=4)
                            cv2.polylines(canvas, [pts], False, col, thickness=2)
                            used_prop.add(pslot)
                            # 在 GT 中心标注类别
                            gxy = denorm_xy(gt_pack[gslot], bounds)
                            gpts = to_px(gxy, bounds, W0, H0)
                            gpts = (gpts.astype(np.float32) * scale).round().astype(np.int32)
                            cen = gpts.mean(axis=0).astype(int)
                            tag = f"{['D','P','B'][lab]}"
                            cv2.putText(canvas, tag, (int(cen[0]), int(cen[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(canvas, tag, (int(cen[0]), int(cen[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        col, 1, cv2.LINE_AA)
                        # 未匹配的提案用灰色虚线
                        for pslot in prop_exist_idx:
                            if int(pslot) in used_prop:
                                continue
                            arr = prop_pack[int(pslot)]
                            xy = denorm_xy(arr, bounds)
                            pts = to_px(xy, bounds, W0, H0)
                            pts = (pts.astype(np.float32) * scale).round().astype(np.int32)
                            cv2.polylines(canvas, [pts], False, (160, 160, 160), thickness=1, lineType=cv2.LINE_AA)
                        cv2.imwrite(osp.join(viz_dir, f'match_raw_prop_ep_{ep:04d}.png'), canvas)
                except Exception as _e:
                    pass

                # 额外：按 Hungarian 匹配后的 GT 类别着色（未匹配为灰色）
                try:
                    gt_exist_idx = np.where(~gt_mask.all(axis=1))[0]
                    if gt_exist_idx.size > 0:
                        pairs = matcher(pc, gt_pack[gt_exist_idx])
                        labels_pred = np.full((N,), -1, dtype=np.int64)
                        for pi, gj_loc in pairs:
                            gj = int(gt_exist_idx[gj_loc])
                            labels_pred[int(pi)] = int(labels_slots[gj])
                        m_pred2 = np.ones((N, P), dtype=bool)
                        for i in keep:
                            m_pred2[i] = False
                        overlay_slots_annot(
                            osp.join(viz_dir, f'ep_{ep:04d}_cls_matched.png'),
                            raster,
                            bounds,
                            pc,
                            m_pred2,
                            labels=labels_pred,
                            title=f'ep {ep:04d} (matched cls)'
                        )
                except Exception:
                    pass

    # Inference test A: pure noise → reconstruction
    enc.eval(); net.eval()
    with torch.no_grad():
        x_noise = torch.randn(1, N, P, 2, device=device)
        rv = enc(torch.from_numpy(raster[None, ...]).to(device))
        pc, pl = net(x_noise, rv)
        pc_np = pc.cpu().numpy()[0]
        overlay_on_raster(osp.join(viz_dir, 'testA_noise_recon.png'), raster, bounds, gt_pack, pc_np)

    # Inference test B: SDEdit-like refinement (strongly jittered proposals)
    with torch.no_grad():
        prop = gt_pack + np.random.normal(scale=0.8, size=gt_pack.shape).astype(np.float32)
        prop = np.clip(prop, -1.0, 1.0)
        x_prop = torch.from_numpy(prop[None, ...]).to(device)
        rv = enc(torch.from_numpy(raster[None, ...]).to(device))
        pc, pl = net(x_prop, rv)
        pc_np = pc.cpu().numpy()[0]
        overlay_on_raster(osp.join(viz_dir, 'testB_prop_refine.png'), raster, bounds, gt_pack, pc_np)

    print(f"[ok] overfit finished. Visualizations under {viz_dir}")


if __name__ == '__main__':
    main()
