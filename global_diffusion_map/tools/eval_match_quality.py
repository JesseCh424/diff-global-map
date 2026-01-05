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


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = (a**2).sum(1, keepdims=True)
    bb = (b**2).sum(1, keepdims=True).T
    ab = a @ b.T
    d2 = np.maximum(aa + bb - 2.0 * ab, 0.0)
    return np.sqrt(d2, dtype=np.float32)


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [m,2], in meters
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 1e6
    D = _pairwise_dist(a, b)
    fwd = float(D.min(axis=1).mean())
    bwd = float(D.min(axis=0).mean())
    return 0.5 * (fwd + bwd)


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


def evaluate(
    agg_pred_dir: str,
    static_gt_dir: str,
    bounds_dir: Optional[str],
    config: str,
    guide_ckpt: str,
    stats_json: str,
    scenes: List[str],
    tau: float = 0.2,
    cand_radius_frac: float = 0.5,
    m_chamfer: int = 16,
    skip_pr: bool = False,
    mode: str = 'train',  # 'train' or 'infer'
) -> Dict[str, Any]:
    stats = json.load(open(stats_json, 'r'))
    M = int(stats.get('M', 30))
    num_queries = int(stats.get('num_queries', 64))
    class_budget = {int(k): int(v) for k, v in stats.get('class_budget', {}).items()}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    guide = build_guide(config, guide_ckpt, M, num_queries)
    from global_diffusion_map.lib.stable_match import stable_match_reorder

    # Map original ids to MapTR labels
    def orig_to_maptr(orig: int) -> int:
        return 0 if orig == 1 else (1 if orig == 0 else 2)

    # Label order by class budgets (slot labels)
    label_order: List[int] = []
    for orig in (2, 1, 0):
        cap = int(class_budget.get(orig, 0))
        lab = orig_to_maptr(orig)
        for _ in range(cap):
            label_order.append(lab)
    K = min(len(label_order), num_queries)

    # Accumulators
    from collections import Counter
    cover_matched = Counter()
    cover_budget = Counter()
    pr_at = {0.5: {'tp': Counter(), 'fp': Counter(), 'fn': Counter()}, 1.0: {'tp': Counter(), 'fp': Counter(), 'fn': Counter()}}
    centroid_errors: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    # Anchor-fit metrics (vs μguide seed): normalized center distance and normalized chamfer
    anchor_center_norm: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    anchor_pw_norm: Dict[int, List[float]] = {0: [], 1: [], 2: []}

    for scene in scenes:
        pred_pkl = osp.join(agg_pred_dir, f'{scene}.pkl')
        gt_pkl = osp.join(static_gt_dir, f'{scene}.pkl')
        if not (osp.exists(pred_pkl) and osp.exists(gt_pkl)):
            continue
        pred = _load_pickle(pred_pkl)
        gt = _load_pickle(gt_pkl)
        # Bounds
        if bounds_dir:
            b = _load_pickle(osp.join(bounds_dir, f'{scene}.pkl')) if osp.exists(osp.join(bounds_dir, f'{scene}.pkl')) else None
            bnds = b.get('bounds', None) if b else None
        else:
            bnds = None
        if bnds is None:
            # compute from prediction vectors
            pts_all: List[np.ndarray] = []
            for k in (0, 1, 2):
                for arr in pred.get(k, []):
                    a = np.asarray(arr); 
                    if a.size:
                        pts_all.append(a)
            if not pts_all:
                continue
            cat = np.concatenate(pts_all, axis=0)
            minx, miny = cat.min(0); maxx, maxy = cat.max(0)
            bnds = [float(minx), float(miny), float(maxx), float(maxy)]

        # Build slot labels and empty mask
        labs = np.zeros((1, num_queries), dtype=np.int64)
        mask = np.ones((1, num_queries, M), dtype=bool)
        for i in range(K):
            labs[0, i] = int(label_order[i]); mask[0, i] = False

        # Prepare proposals
        def _curve_len_m(arr: np.ndarray) -> float:
            a = np.asarray(arr, dtype=np.float32)
            if a.shape[0] < 2:
                return 0.0
            d = a[1:] - a[:-1]
            return float(np.linalg.norm(d, axis=1).sum())

        # min length per MapTR label (meters)
        if mode == 'infer':
            min_len = {0: 8.0, 1: 5.0, 2: 15.0}  # stricter for inference
        else:
            min_len = {0: 6.0, 1: 4.0, 2: 12.0}  # slightly looser for training

        prop_list: List[np.ndarray] = []
        prop_labs: List[int] = []
        for orig in (2, 1, 0):
            vecs = pred.get(orig, [])
            # Rank by world-length (meters)
            lengths = [(_curve_len_m(v), i) for i, v in enumerate(vecs)]
            lengths.sort(key=lambda x: x[0], reverse=True)
            lab = orig_to_maptr(orig)
            take = int(max(1, class_budget.get(orig, 0) * 3))
            picked = 0
            for L, idx in lengths:
                if L < min_len[lab]:
                    continue
                smp = _uniform_sample(np.asarray(vecs[idx]), M)
                norm = _normalize_xy(smp, bnds)
                prop_list.append(norm)
                prop_labs.append(lab)
                picked += 1
                if picked >= take:
                    break
        if not prop_list:
            continue
        prop_np = np.stack(prop_list, axis=0)[None, ...]

        # Build μguide centers (seed by class-wise proposals)
        cls_to_idx: Dict[int, List[int]] = {}
        for i, lab in enumerate(prop_labs):
            cls_to_idx.setdefault(int(lab), []).append(i)
        seed = np.zeros((1, K, M, 2), dtype=np.float32)
        ptr = {0: 0, 1: 0, 2: 0}
        for j in range(K):
            c = int(labs[0, j]); arr = cls_to_idx.get(c, []); p = ptr.get(c, 0)
            if p < len(arr):
                seed[0, j] = prop_np[0, arr[p]]; ptr[c] = p + 1
        init_pts = torch.from_numpy(seed).to(device)
        guide_mask = torch.zeros((1, K, M), dtype=torch.bool, device=device)
        guide_labs = torch.from_numpy(labs[:, :K]).to(device)
        with torch.no_grad():
            guide_attn = torch.zeros_like(guide_mask)
            g_center, _ = guide(init_pts, guide_attn, guide_labs)
            g_mean = g_center.detach().cpu().numpy()[0, :K]
        # Use seeded curves as target shapes for matching (better direction/shape cues)
        tgt_np = seed.copy()

        # Match & pack with class-specific overrides
        if mode == 'infer':
            per_class = {
                # MapTR labels: 0=divider, 1=ped, 2=boundary
                0: {'w_center': 1.4, 'w_dir': 0.10, 'w_pw': 0.30, 'thres': max(0.22, float(tau)-0.12), 'cand_radius_frac': max(0.40, float(cand_radius_frac)-0.30), 'pw_thres': 0.08},
                1: {'w_center': 1.3, 'w_dir': 0.10, 'w_pw': 0.30, 'thres': max(0.22, float(tau)-0.10), 'cand_radius_frac': max(0.45, float(cand_radius_frac)-0.25), 'pw_thres': 0.10},
                2: {'w_center': 2.2, 'w_dir': 0.05, 'w_pw': 0.15, 'thres': max(0.22, float(tau)-0.10), 'cand_radius_frac': max(0.35, float(cand_radius_frac)-0.35), 'pw_thres': 0.12},
            }
        else:
            per_class = {
                0: {'w_center': 1.2, 'w_dir': 0.15, 'w_pw': 0.35, 'thres': 0.25, 'cand_radius_frac': 0.50, 'pw_thres': 0.12},
                1: {'w_center': 1.1, 'w_dir': 0.15, 'w_pw': 0.35, 'thres': 0.25, 'cand_radius_frac': 0.50, 'pw_thres': 0.14},
                2: {'w_center': 1.8, 'w_dir': 0.10, 'w_pw': 0.25, 'thres': 0.25, 'cand_radius_frac': 0.40, 'pw_thres': 0.16},
            }
        new_pts, new_mask = stable_match_reorder(
            prop_np=prop_np,
            tgt_np=tgt_np,
            labs_np=labs,
            mask_np=mask,
            prop_labels=np.array(prop_labs, dtype=np.int64),
            w_center=1.0,
            w_dir=0.2,
            w_pw=0.5,
            thres=float(tau),
            cand_radius_frac=float(cand_radius_frac),
            m_pw=int(m_chamfer),
            per_class=per_class,
        )

        # Post-gate by world centroid distance to μguide center (in meters)
        # Stricter for divider/ped, looser for boundary
        cmax = {0: (5.0 if mode=='train' else 4.0), 1: (5.0 if mode=='train' else 4.0), 2: (6.0 if mode=='train' else 5.0)}
        g_world = _denormalize_xy(g_mean, bnds)
        for j in range(K):
            if new_mask[0, j].all():
                continue
            lab = int(labs[0, j])
            crv = _denormalize_xy(new_pts[0, j], bnds)
            p = crv.mean(axis=0)
            d = float(np.linalg.norm(p - g_world[j]))
            if d > cmax.get(lab, 6.0):
                new_mask[0, j] = True

        # Coverage accumulators
        from collections import Counter as Cn
        budget = Cn([int(labs[0, j]) for j in range(K)])
        matched_idx = [j for j in range(K) if not new_mask[0, j].all()]
        matched_cnt = Cn([int(labs[0, j]) for j in matched_idx])
        for k in [0, 1, 2]:
            cover_matched[k] += int(matched_cnt.get(k, 0))
            cover_budget[k] += int(budget.get(k, 0))

        # Anchor-fit stats (normalized to diag=sqrt(8) in [-1,1])
        diag = np.sqrt(8.0)
        for j in matched_idx:
            c = int(labs[0, j])
            prop = prop_np[0]  # proposals pool normalized
            # find matched prop curve by comparing to packed new_pts (closest in pool)
            pr = new_pts[0, j]
            # compute center distance vs seed curve center (normalized)
            seed_center = seed[0, j].mean(axis=0)
            pr_center = pr.mean(axis=0)
            dc = float(np.linalg.norm(pr_center - seed_center)) / float(diag)
            anchor_center_norm[c].append(dc)
            # simplified chamfer norm vs seed
            a = _uniform_sample(pr, m_chamfer)
            b = _uniform_sample(seed[0, j], m_chamfer)
            # pairwise dist in normalized space
            # reuse chamfer_distance but assume coords in [-1,1]; scale by diag
            def _ch(a,b):
                D = _pairwise_dist(a, b)
                return 0.5*float(D.min(axis=1).mean()+D.min(axis=0).mean())
            dch = _ch(a, b) / float(diag)
            anchor_pw_norm[c].append(dch)

        # Build GT arrays per class
        gt_world: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
        for orig in (0, 1, 2):
            lab = orig_to_maptr(orig)
            for arr in gt.get(orig, []):
                a = np.asarray(arr, dtype=np.float32)
                if a.size:
                    gt_world[lab].append(a)

        if not skip_pr:
            # P/R with Chamfer thresholds
            for th in [0.5, 1.0]:
                for c in [0, 1, 2]:
                    preds = [ _denormalize_xy(new_pts[0, j], bnds) for j in matched_idx if int(labs[0, j]) == c ]
                    gts = gt_world.get(c, [])
                    if not gts:
                        pr_at[th]['tp'][c] += 0
                        pr_at[th]['fp'][c] += len(preds)
                        pr_at[th]['fn'][c] += 0
                        continue
                    P = len(preds); G = len(gts)
                    if P == 0:
                        pr_at[th]['fn'][c] += G
                        continue
                    D = np.zeros((P, G), dtype=np.float32)
                    for i in range(P):
                        a = _uniform_sample(preds[i], m_chamfer)
                        for j in range(G):
                            b = _uniform_sample(gts[j], m_chamfer)
                            D[i, j] = chamfer_distance(a, b)
                    try:
                        from scipy.optimize import linear_sum_assignment  # type: ignore
                        row_ind, col_ind = linear_sum_assignment(D)
                        matched = [(i, j, float(D[i, j])) for i, j in zip(row_ind, col_ind) if D[i, j] <= th]
                        tp = len(matched)
                    except Exception:
                        flat = [(float(D[i, j]), i, j) for i in range(P) for j in range(G)]
                        flat.sort(key=lambda x: x[0])
                        used_p, used_g, matched = set(), set(), []
                        for d, i, j in flat:
                            if d > th: break
                            if i in used_p or j in used_g: continue
                            used_p.add(i); used_g.add(j); matched.append((i, j, float(d)))
                        tp = len(matched)
                    fp = max(P - tp, 0)
                    fn = max(len(gts) - tp, 0)
                    pr_at[th]['tp'][c] += tp
                    pr_at[th]['fp'][c] += fp
                    pr_at[th]['fn'][c] += fn
                    for (i, j, d) in matched:
                        pa = preds[i].mean(axis=0); pb = gts[j].mean(axis=0)
                        centroid_errors[c].append(float(np.linalg.norm(pa - pb)))

    # Summaries
    out: Dict[str, Any] = {}
    # Coverage
    cov = {k: float(cover_matched[k]) / max(int(cover_budget[k]), 1) for k in [0, 1, 2]}
    out['coverage_per_class'] = {'divider': cov[0], 'ped': cov[1], 'boundary': cov[2]}
    out['coverage_overall'] = float(sum(cover_matched.values())) / max(int(sum(cover_budget.values())), 1)
    # P/R at thresholds
    out['pr'] = {}
    if not skip_pr:
        for th in [0.5, 1.0]:
            rec = {}
            for c, name in zip([0, 1, 2], ['divider', 'ped', 'boundary']):
                tp = pr_at[th]['tp'][c]; fp = pr_at[th]['fp'][c]; fn = pr_at[th]['fn'][c]
                P = float(tp) / max(tp + fp, 1); R = float(tp) / max(tp + fn, 1)
                rec[name] = {'P': P, 'R': R, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
            tp = sum(pr_at[th]['tp'].values()); fp = sum(pr_at[th]['fp'].values()); fn = sum(pr_at[th]['fn'].values())
            rec['overall'] = {'P': float(tp)/max(tp+fp,1), 'R': float(tp)/max(tp+fn,1), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
            out['pr'][str(th)] = rec
    # Centroid error stats (m)
    out['centroid_error_m'] = {
        'divider': float(np.median(centroid_errors[0]) if centroid_errors[0] else 0.0),
        'ped': float(np.median(centroid_errors[1]) if centroid_errors[1] else 0.0),
        'boundary': float(np.median(centroid_errors[2]) if centroid_errors[2] else 0.0),
    }
    # Anchor-fit stats (normalized)
    def _pct(ls: List[float], p: float) -> float:
        if not ls:
            return 0.0
        a = np.array(ls)
        return float(np.percentile(a, p))
    out['anchor_fit_norm'] = {
        'center_p50': {
            'divider': _pct(anchor_center_norm[0], 50),
            'ped': _pct(anchor_center_norm[1], 50),
            'boundary': _pct(anchor_center_norm[2], 50),
        },
        'pw_chamfer_p50': {
            'divider': _pct(anchor_pw_norm[0], 50),
            'ped': _pct(anchor_pw_norm[1], 50),
            'boundary': _pct(anchor_pw_norm[2], 50),
        }
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate proposal↔guide matching quality vs GT')
    ap.add_argument('--agg-pred-dir', required=True)
    ap.add_argument('--static-gt-dir', required=True)
    ap.add_argument('--bounds-dir', default=None)
    ap.add_argument('--config', required=True)
    ap.add_argument('--guide-ckpt', required=True)
    ap.add_argument('--stats-json', required=True)
    ap.add_argument('--scene-list', default=None, help='Optional file with scene IDs; otherwise scan agg dir')
    ap.add_argument('--num-scenes', type=int, default=0, help='Limit to first N scenes (0=all)')
    ap.add_argument('--tau', type=float, default=0.2)
    ap.add_argument('--cand-radius-frac', type=float, default=0.5)
    ap.add_argument('--m-chamfer', type=int, default=16)
    ap.add_argument('--skip-pr', action='store_true', help='Skip PR vs GT for speed (only coverage/anchor-fit)')
    args = ap.parse_args()

    if args.scene_list and osp.exists(args.scene_list):
        scenes = [ln.strip() for ln in open(args.scene_list,'r').read().splitlines() if ln.strip()]
    else:
        scenes = sorted([osp.splitext(p)[0] for p in os.listdir(args.agg_pred_dir) if p.endswith('.pkl')])
    if args.num_scenes and args.num_scenes > 0:
        scenes = scenes[:args.num_scenes]

    out = evaluate(
        agg_pred_dir=args.agg_pred_dir,
        static_gt_dir=args.static_gt_dir,
        bounds_dir=args.bounds_dir,
        config=args.config,
        guide_ckpt=args.guide_ckpt,
        stats_json=args.stats_json,
        scenes=scenes,
        tau=args.tau,
        cand_radius_frac=args.cand_radius_frac,
        m_chamfer=args.m_chamfer,
        skip_pr=args.skip_pr,
        mode='train',
    )
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
