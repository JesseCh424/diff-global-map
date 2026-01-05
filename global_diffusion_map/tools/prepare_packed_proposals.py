#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


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
    try:
        from mmdet.models import BACKBONES  # type: ignore
        BACKBONES.module_dict.pop('EfficientNet', None)
    except Exception:
        pass


def _load_pickle(path: str) -> Dict[str, Any]:
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


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


def build_guide(cfg_path: str, guide_ckpt_path: str, num_verts: int, num_queries: int) -> torch.nn.Module:
    _prep_env()
    from mmcv import Config
    from mmdet3d.models import build_model  # type: ignore
    cfg = Config.fromfile(cfg_path)
    if hasattr(cfg, 'plugin') and cfg.plugin:
        assert hasattr(cfg, 'plugin_dir')
        import importlib, sys
        repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
        maptracker_root = osp.join(repo_root, 'maptracker')
        if maptracker_root not in sys.path:
            sys.path.insert(0, maptracker_root)
        module_path = cfg.plugin_dir.replace('/', '.')[:-1] if cfg.plugin_dir.endswith('/') else cfg.plugin_dir.replace('/', '.')
        try:
            importlib.import_module(module_path)
        except ModuleNotFoundError:
            module_path2 = ('maptracker.' + module_path) if not module_path.startswith('maptracker.') else module_path
            importlib.import_module(module_path2)
    model = build_model(cfg.model)
    pe_dim = int(getattr(model.pts_bbox_head.positional_encoding, 'num_feats'))
    embed_dim = int(getattr(model.pts_bbox_head.transformer, 'embed_dims'))
    from src.models.polygon_models.polygon_meta import PolyMetaModel  # type: ignore
    net_guide = PolyMetaModel(input_dim=pe_dim, embed_dim=embed_dim, max_poly=int(num_queries), num_vert=int(num_verts))
    state = torch.load(guide_ckpt_path, map_location='cpu')
    net_guide.load_state_dict(state['net'], strict=False)
    net_guide.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    return net_guide


def prepare(
    agg_pred_dir: str,
    bounds_dir: Optional[str],
    out_root: str,
    config: str,
    guide_ckpt: str,
    stats_json: str,
    mode: str = 'train',
) -> None:
    os.makedirs(out_root, exist_ok=True)
    stats = json.load(open(stats_json, 'r'))
    M = int(stats.get('M', 30))
    num_queries = int(stats.get('num_queries', 64))
    class_budget = {int(k): int(v) for k, v in stats.get('class_budget', {}).items()}
    guide = build_guide(config, guide_ckpt, M, num_queries)

    def orig_to_maptr(orig: int) -> int:
        return 0 if orig == 1 else (1 if orig == 0 else 2)

    # Label order
    label_order: List[int] = []
    for orig in (2, 1, 0):
        cap = int(class_budget.get(orig, 0))
        lab = orig_to_maptr(orig)
        for _ in range(cap):
            label_order.append(lab)
    K = min(len(label_order), num_queries)

    # Per-class params
    if mode == 'train':
        min_len = {0: 6.0, 1: 4.0, 2: 12.0}
        per_class = {
            0: {'w_center': 1.2, 'w_dir': 0.15, 'w_pw': 0.35, 'thres': 0.25, 'cand_radius_frac': 0.50, 'pw_thres': 0.12},
            1: {'w_center': 1.1, 'w_dir': 0.15, 'w_pw': 0.35, 'thres': 0.25, 'cand_radius_frac': 0.50, 'pw_thres': 0.14},
            2: {'w_center': 1.8, 'w_dir': 0.10, 'w_pw': 0.25, 'thres': 0.25, 'cand_radius_frac': 0.40, 'pw_thres': 0.16},
        }
        cmax = {0: 5.0, 1: 5.0, 2: 6.0}
    else:
        min_len = {0: 8.0, 1: 5.0, 2: 15.0}
        per_class = {
            0: {'w_center': 1.4, 'w_dir': 0.10, 'w_pw': 0.30, 'thres': 0.25, 'cand_radius_frac': 0.60, 'pw_thres': 0.10},
            1: {'w_center': 1.3, 'w_dir': 0.10, 'w_pw': 0.30, 'thres': 0.25, 'cand_radius_frac': 0.60, 'pw_thres': 0.12},
            2: {'w_center': 2.0, 'w_dir': 0.05, 'w_pw': 0.20, 'thres': 0.25, 'cand_radius_frac': 0.50, 'pw_thres': 0.15},
        }
        cmax = {0: 4.0, 1: 4.0, 2: 5.0}

    from global_diffusion_map.lib.stable_match import stable_match_reorder

    scenes = sorted([osp.splitext(p)[0] for p in os.listdir(agg_pred_dir) if p.endswith('.pkl')])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for scene in scenes:
        out_npz = osp.join(out_root, f'{scene}.npz')
        if osp.exists(out_npz):
            continue
        pred = _load_pickle(osp.join(agg_pred_dir, f'{scene}.pkl'))
        # bounds
        if bounds_dir and osp.exists(osp.join(bounds_dir, f'{scene}.pkl')):
            b = _load_pickle(osp.join(bounds_dir, f'{scene}.pkl'))
            bounds = b.get('bounds', None)
        else:
            bounds = pred.get('bounds', None)
        if bounds is None:
            pts_all = []
            for k in (0, 1, 2):
                for arr in pred.get(k, []):
                    pts_all.append(arr)
            if not pts_all:
                continue
            cat = np.concatenate(pts_all, axis=0)
            minx, miny = cat.min(0); maxx, maxy = cat.max(0)
            bounds = [float(minx), float(miny), float(maxx), float(maxy)]

        # Build slot labels/masks
        labs = np.zeros((1, num_queries), dtype=np.int64)
        msk = np.ones((1, num_queries, M), dtype=bool)
        for i in range(K):
            labs[0, i] = int(label_order[i]); msk[0, i] = False

        # Build proposal pool (normalize + resample)
        def _curve_len(arr: np.ndarray) -> float:
            a = np.asarray(arr, dtype=np.float32)
            if a.shape[0] < 2: return 0.0
            d = a[1:] - a[:-1]
            return float(np.linalg.norm(d, axis=1).sum())
        prop_list: List[np.ndarray] = []
        prop_labs: List[int] = []
        for orig in (2, 1, 0):
            vecs = pred.get(orig, [])
            lab = orig_to_maptr(orig)
            Ls = [(_curve_len(v), i) for i, v in enumerate(vecs)]
            Ls.sort(key=lambda x: x[0], reverse=True)
            take = int(max(1, class_budget.get(orig, 0) * 3))
            picked = 0
            for L, idx in Ls:
                if L < min_len[lab]:
                    continue
                smp = _uniform_sample(np.asarray(vecs[idx]), M)
                prop_list.append(_normalize_xy(smp, bounds))
                prop_labs.append(lab)
                picked += 1
                if picked >= take:
                    break
        if not prop_list:
            continue
        prop_np = np.stack(prop_list, axis=0)[None, ...]

        # μguide anchors (seeded by class-wise props)
        # seed = class-wise round-robin from proposal pool
        cls_to_idx: Dict[int, List[int]] = {}
        for i, lab in enumerate(prop_labs):
            cls_to_idx.setdefault(int(lab), []).append(i)
        seed = np.zeros((1, K, M, 2), dtype=np.float32)
        ptr = {0: 0, 1: 0, 2: 0}
        for j in range(K):
            c = int(labs[0, j]); arr = cls_to_idx.get(c, []); p = ptr.get(c, 0)
            if p < len(arr):
                seed[0, j] = prop_np[0, arr[p]]; ptr[c] = p + 1
        # Guide centers (for anchor-fit stats only; not required as matching targets)
        init = torch.from_numpy(seed).to(device)
        g_mask = torch.zeros((1, K, M), dtype=torch.bool, device=device)
        g_labs = torch.from_numpy(labs[:, :K]).to(device)
        with torch.no_grad():
            g_att = torch.zeros_like(g_mask)
            g_center, _ = guide(init, g_att, g_labs)
            g_mean = g_center.detach().cpu().numpy()[0, :K]
        # Matching targets: use seed curves to preserve direction/shape cues
        tgt_np = seed.copy()

        # Matching and gating
        new_pts, new_mask = stable_match_reorder(
            prop_np=prop_np,
            tgt_np=tgt_np,
            labs_np=labs,
            mask_np=msk,
            prop_labels=np.array(prop_labs, dtype=np.int64),
            w_center=1.0, w_dir=0.2, w_pw=0.5, thres=0.25, cand_radius_frac=0.6, m_pw=8,
            per_class=per_class,
        )
        # Post-gating by world-space centroid distance to guide center
        g_world = _denormalize_xy(g_mean, bounds)
        for j in range(K):
            if new_mask[0, j].all():
                continue
            labj = int(labs[0, j])
            crv = _denormalize_xy(new_pts[0, j], bounds)
            p = crv.mean(axis=0)
            d = float(np.linalg.norm(p - g_world[j]))
            if d > cmax.get(labj, 5.0):
                new_mask[0, j] = True

        # Save as compressed npz
        np.savez_compressed(out_npz,
            pts=new_pts[0].astype(np.float32),
            mask=new_mask[0],
            labels=labs[0].astype(np.int64),
            bounds=np.array(bounds, dtype=np.float32),
        )
        print(f'[pack] {scene} -> {out_npz}')


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare packed proposals (matched+gated) for train/infer')
    ap.add_argument('--agg-pred-dir', required=True)
    ap.add_argument('--bounds-dir', default=None)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--guide-ckpt', required=True)
    ap.add_argument('--stats-json', required=True)
    ap.add_argument('--mode', choices=['train','infer'], default='train')
    args = ap.parse_args()
    prepare(args.agg_pred_dir, args.bounds_dir, args.out_root, args.config, args.guide_ckpt, args.stats_json, mode=args.mode)


if __name__ == '__main__':
    main()
