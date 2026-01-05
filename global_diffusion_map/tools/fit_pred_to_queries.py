#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mmcv import Config


def _prep_env() -> None:
    import sys, pathlib
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    poly_root = osp.join(repo_root, 'poly-diffuse')
    if poly_root not in sys.path:
        sys.path.insert(0, poly_root)
    try:
        import torch as _t
        torch_lib = osp.join(pathlib.Path(_t.__file__).parent, 'lib')
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH','')}"
    except Exception:
        pass
    # Avoid duplicate EfficientNet registration
    try:
        from mmdet.models import BACKBONES  # type: ignore
        BACKBONES.module_dict.pop('EfficientNet', None)
    except Exception:
        pass


def _load_pickle(path: str) -> Dict[str, Any]:
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def _uniform_sample(poly: np.ndarray, M: int) -> np.ndarray:
    from shapely.geometry import LineString
    ls = LineString(poly)
    if ls.length <= 1e-6:
        p = np.array(ls.coords[0], dtype=np.float32)
        return np.tile(p[None, :], (M, 1))
    dists = np.linspace(0.0, ls.length, num=M, dtype=np.float32)
    pts = [list(ls.interpolate(float(d)).coords)[0] for d in dists]
    return np.asarray(pts, dtype=np.float32)


def _normalize_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = ((xy[:, 0] - minx) / w) * 2.0 - 1.0
    out[:, 1] = ((xy[:, 1] - miny) / h) * 2.0 - 1.0
    return out


def _denormalize_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = (xy[:, 0] + 1.0) * 0.5 * w + minx
    out[:, 1] = (xy[:, 1] + 1.0) * 0.5 * h + miny
    return out


def _compute_bounds_from_vectors(scene_dict: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    pts: List[np.ndarray] = []
    for k in (0, 1, 2):
        for arr in scene_dict.get(k, []):
            a = np.asarray(arr)
            if a.size:
                pts.append(a)
    if not pts:
        return None
    cat = np.concatenate(pts, axis=0)
    minx, miny = cat.min(0)
    maxx, maxy = cat.max(0)
    return float(minx), float(miny), float(maxx), float(maxy)


def _build_maptr_labels_and_budgets(budgets: Dict[int, int]) -> List[int]:
    # Map original ids to MapTR labels: divider=0, ped=1, boundary=2
    # Budget keys are original ids (0:ped,1:divider,2:boundary)
    order: List[int] = []
    for orig in (2, 1, 0):  # keep consistent with infer_av2 ordering
        cap = int(budgets.get(orig, 0))
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        for _ in range(cap):
            order.append(lab)
    return order


def _build_guide(cfg_path: str, guide_ckpt_path: str, num_verts: int, num_queries: int) -> torch.nn.Module:
    _prep_env()
    from mmdet3d.models import build_model  # type: ignore
    cfg = Config.fromfile(cfg_path)
    if hasattr(cfg, 'plugin') and cfg.plugin:
        assert hasattr(cfg, 'plugin_dir')
        import importlib, sys
        # Ensure maptracker root is on sys.path so 'plugin' under maptracker can be imported
        repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
        maptracker_root = osp.join(repo_root, 'maptracker')
        if maptracker_root not in sys.path:
            sys.path.insert(0, maptracker_root)
        module_path = cfg.plugin_dir.replace('/', '.')[:-1] if cfg.plugin_dir.endswith('/') else cfg.plugin_dir.replace('/', '.')
        try:
            importlib.import_module(module_path)
        except ModuleNotFoundError:
            # Try namespaced import under maptracker (e.g., maptracker.plugin)
            module_path2 = ('maptracker.' + module_path) if not module_path.startswith('maptracker.') else module_path
            importlib.import_module(module_path2)
    model = build_model(cfg.model)
    # Derive dims from MapTR
    pe_dim = int(getattr(model.pts_bbox_head.positional_encoding, 'num_feats'))
    embed_dim = int(getattr(model.pts_bbox_head.transformer, 'embed_dims'))
    from src.models.polygon_models.polygon_meta import PolyMetaModel  # type: ignore
    net_guide = PolyMetaModel(input_dim=pe_dim, embed_dim=embed_dim, max_poly=int(num_queries), num_vert=int(num_verts))
    state = torch.load(guide_ckpt_path, map_location='cpu')
    net_guide.load_state_dict(state['net'], strict=False)
    net_guide.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    return net_guide


def main() -> None:
    ap = argparse.ArgumentParser(description='Fit agg_pred vectors into fixed query space using μguide as anchor, and visualize matches')
    ap.add_argument('--agg-pred-dir', required=True, help='Aggregated prediction pkls dir')
    ap.add_argument('--config', required=True, help='Config to derive MapTR dims for guide')
    ap.add_argument('--guide-ckpt', default='global_diffusion_map/ckpts/guide/network-snapshot_m30q64.pth', help='Guide snapshot path')
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json', help='Stats JSON with M, num_queries, class_budget')
    ap.add_argument('--bounds-dir', default=None, help='Optional dir to read per-scene bounds from <dir>/<scene>.pkl["bounds"]')
    ap.add_argument('--out-dir', required=True, help='Where to save packed proposals (npz) and debug JSON')
    ap.add_argument('--viz-dir', required=True, help='Where to save visualization PNGs')
    ap.add_argument('--scenes', nargs='*', default=None, help='Optional scene ids (omit to use all from agg-pred dir)')
    # Matching controls
    ap.add_argument('--w-center', type=float, default=1.0)
    ap.add_argument('--w-dir', type=float, default=0.2)
    ap.add_argument('--w-chamfer', type=float, default=0.5)
    ap.add_argument('--tau', type=float, default=0.07, help='gate threshold in normalized units')
    ap.add_argument('--cand-radius-frac', type=float, default=0.12)
    ap.add_argument('--m-chamfer', type=int, default=8)
    args = ap.parse_args()

    stats = json.load(open(args.stats_json, 'r'))
    M = int(stats.get('M', 30))
    num_queries = int(stats.get('num_queries', 64))
    class_budget = {int(k): int(v) for k, v in stats.get('class_budget', {}).items()}
    label_order = _build_maptr_labels_and_budgets(class_budget)
    assert len(label_order) <= num_queries

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    guide = _build_guide(args.config, args.guide_ckpt, M, num_queries)

    from global_diffusion_map.lib.stable_match import stable_match_reorder

    # Collect scenes
    if args.scenes:
        scenes = list(args.scenes)
    else:
        scenes = sorted([osp.splitext(p)[0] for p in os.listdir(args.agg_pred_dir) if p.endswith('.pkl')])

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)

    for scene in scenes:
        pred_pkl = osp.join(args.agg_pred_dir, f'{scene}.pkl')
        if not osp.exists(pred_pkl):
            print(f'[skip] {scene}: missing agg_pred pkl')
            continue
        pred = _load_pickle(pred_pkl)
        # Bounds
        if args.bounds_dir:
            bd_pkl = osp.join(args.bounds_dir, f'{scene}.pkl')
            bnd = None
            if osp.exists(bd_pkl):
                d = _load_pickle(bd_pkl)
                bnd = d.get('bounds', None)
            if bnd is None:
                bnd = _compute_bounds_from_vectors(pred)
        else:
            bnd = _compute_bounds_from_vectors(pred)
        if bnd is None:
            print(f'[warn] {scene}: no bounds; skip')
            continue
        bounds = list(map(float, bnd))

        # Build initial arrays
        pts = np.zeros((1, num_queries, M, 2), dtype=np.float32)
        mask = np.ones((1, num_queries, M), dtype=bool)
        labs = np.zeros((1, num_queries), dtype=np.int64)
        # Fill lab order for first K slots
        K = min(len(label_order), num_queries)
        for i in range(K):
            labs[0, i] = int(label_order[i])
            mask[0, i] = False  # mark as valid slots; geometry to be filled after matching

        # Prepare proposals (normalize + resample)
        prop_list: List[np.ndarray] = []
        prop_labs: List[int] = []
        # Prefer longer curves first within each class; cap to ~5x budget to control time
        for orig in (2, 1, 0):
            vecs = pred.get(orig, [])
            lens = [float(getattr(v, 'shape', [0])[0]) for v in vecs]
            order = np.argsort(lens)[::-1]
            take = int(max(1, class_budget.get(orig, 0) * 5))
            for idx in list(order[:take]):
                smp = _uniform_sample(np.asarray(vecs[idx]), M)
                norm = _normalize_xy(smp, bounds)
                prop_list.append(norm)
                # map to MapTR label
                lab = 0 if orig == 1 else (1 if orig == 0 else 2)
                prop_labs.append(lab)
        if not prop_list:
            print(f'[warn] {scene}: empty proposals')
            continue
        P = len(prop_list)
        prop_np = np.stack(prop_list, axis=0)[None, ...]  # [1,P,M,2]

        # Build guide μ centers (seed with proposals per-slot to obtain scene-aware anchors)
        # For each slot j with label labs[0,j], pick next proposal of the same class as seed
        cls_to_indices: Dict[int, List[int]] = {}
        for idx, lab in enumerate(prop_labs):
            cls_to_indices.setdefault(int(lab), []).append(idx)
        # Pointers per class
        cls_ptr: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        seed = np.zeros((1, K, M, 2), dtype=np.float32)
        for j in range(K):
            cls = int(labs[0, j])
            arr = cls_to_indices.get(cls, [])
            p = cls_ptr.get(cls, 0)
            if p < len(arr):
                seed[0, j] = prop_np[0, arr[p]]
                cls_ptr[cls] = p + 1
            else:
                seed[0, j] = 0.0
        init_pts = torch.from_numpy(seed).to(device)
        guide_mask = torch.zeros((1, K, M), dtype=torch.bool, device=device)
        guide_labs = torch.from_numpy(labs[:, :K]).to(device)
        with torch.no_grad():
            guide_attn = torch.zeros_like(guide_mask)
            g_center, _ = guide(init_pts, guide_attn, guide_labs)
            g_mean = g_center.detach().cpu().numpy()[0, :K]  # [K,2]
        # Tile to M for matching target
        tgt_np = np.tile(g_mean[None, :, None, :], (1, 1, M, 1))  # [1,K,M,2]

        # Run stable match reorder
        new_pts, new_mask = stable_match_reorder(
            prop_np=prop_np,
            tgt_np=tgt_np,
            labs_np=labs,
            mask_np=mask,
            prop_labels=np.array(prop_labs, dtype=np.int64),
            w_center=float(args.w_center),
            w_dir=float(args.w_dir),
            w_pw=float(args.w_chamfer),
            thres=float(args.tau),
            cand_radius_frac=float(args.cand_radius_frac),
            m_pw=int(args.m_chamfer),
        )

        # Visualization with matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            minx, miny, maxx, maxy = bounds
            fig = plt.figure(figsize=(11, 5))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            for ax in (ax1, ax2):
                ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
                ax.set_aspect('equal', adjustable='box'); ax.axis('off')
            color = {0: 'r', 1: 'b', 2: 'g'}  # MapTR label colors

            def _edge_label_positions(K:int, bounds:Sequence[float]):
                x0, y0, x1, y1 = bounds
                W = x1 - x0; H = y1 - y0
                # Margin away from border in meters
                mx = 0.02 * max(W, 1e-6)
                my = 0.02 * max(H, 1e-6)
                # Counters per side
                cnt = { 'top':0, 'right':0, 'bottom':0, 'left':0 }
                # Estimate per-side quotas
                quotas = { 'top': (K+3)//4, 'right': (K+2)//4, 'bottom': (K+1)//4, 'left': K//4 }
                pos = []
                for j in range(K):
                    side = ['top','right','bottom','left'][j % 4]
                    i = cnt[side]; n = max(quotas[side], 1)
                    t = (i+1) / (n+1)
                    if side == 'top':
                        x = x0 + t * W; y = y1 - my
                    elif side == 'bottom':
                        x = x0 + t * W; y = y0 + my
                    elif side == 'right':
                        x = x1 - mx; y = y0 + t * H
                    else:  # left
                        x = x0 + mx; y = y0 + t * H
                    pos.append((x, y))
                    cnt[side] += 1
                return pos

            # Left panel: only matched anchors (μguide centers for matched slots) with edge labels
            matched_idx = [j for j in range(K) if not new_mask[0, j].all()]
            g_world = _denormalize_xy(g_mean, bounds)  # [K,2]
            dx = 0.5  # cross half-length in meters
            left_labels_xy = _edge_label_positions(len(matched_idx), bounds)
            for i, j in enumerate(matched_idx):
                lab = int(labs[0, j])
                name = ('divider' if lab == 0 else ('ped' if lab == 1 else 'boundary')) + f'{j:02d}'
                c = g_world[j]
                # draw small cross only for matched anchors
                ax1.plot([c[0]-dx, c[0]+dx], [c[1], c[1]], color=color.get(lab,'k'), linewidth=1.2, alpha=0.9)
                ax1.plot([c[0], c[0]], [c[1]-dx, c[1]+dx], color=color.get(lab,'k'), linewidth=1.2, alpha=0.9)
                lx, ly = left_labels_xy[i]
                # leader line from center to edge label (text at edge)
                ax1.annotate(name, xy=(c[0], c[1]), xytext=(lx, ly), textcoords='data',
                             fontsize=9, color='k', ha='center', va='center',
                             arrowprops=dict(arrowstyle='-', color='k', lw=0.6, alpha=0.7))
            ax1.set_title('matched anchors', fontsize=12)

            # Right panel: only matched curves + edge labels
            right_labels_xy = _edge_label_positions(len(matched_idx), bounds)
            for i, j in enumerate(matched_idx):
                lab = int(labs[0, j])
                name = ('divider' if lab == 0 else ('ped' if lab == 1 else 'boundary')) + f'{j:02d}'
                crv = _denormalize_xy(new_pts[0, j], bounds)
                ax2.plot(crv[:, 0], crv[:, 1], color=color.get(lab, 'k'), linewidth=1.6, alpha=0.95)
                p = crv.mean(axis=0)
                lx, ly = right_labels_xy[i]
                ax2.annotate(name, xy=(p[0], p[1]), xytext=(lx, ly), textcoords='data',
                             fontsize=9, color='k', ha='center', va='center',
                             arrowprops=dict(arrowstyle='-', color='k', lw=0.8, alpha=0.7))
            ax2.set_title('matched', fontsize=12)

            # Bottom-of-figure summary: per-class matched/empty across both panels
            from collections import Counter
            labs_list = [int(labs[0, j]) for j in range(K)]
            budget = Counter(labs_list)
            matched_counts = Counter([int(labs[0, j]) for j in matched_idx])
            names = {0:'divider', 1:'ped', 2:'boundary'}
            parts = []
            for k in [0,1,2]:
                m = int(matched_counts.get(k, 0)); b = int(budget.get(k, 0)); e = max(b - m, 0)
                parts.append(f"{names[k]}: {m}/{b} matched, {e} empty")
            fig.text(0.5, 0.02, " | ".join(parts), ha='center', va='center', fontsize=11, color='k')
            out_png = osp.join(args.viz_dir, f'{scene}.png')
            plt.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
        except Exception as e:
            print(f'[warn] viz failed for {scene}: {e}')

        # Dump debug json (slot occupancy)
        dbg = {
            'scene': scene,
            'bounds': bounds,
            'num_slots': K,
            'labels': [int(labs[0, j]) for j in range(K)],
            'filled': [not bool(new_mask[0, j].all()) for j in range(K)],
        }
        _save_json(dbg, osp.join(args.out_dir, f'{scene}.json'))
        print(f'[ok] {scene}: packed+viz -> {args.viz_dir}')


if __name__ == '__main__':
    main()
