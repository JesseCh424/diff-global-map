#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
from typing import Dict, List, Tuple

import numpy as np

try:
    from shapely.geometry import LineString
except Exception:
    LineString = None  # fallback to raw sampling


def _load_pickle(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _sample_line(arr: np.ndarray, num: int = 100) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.shape[0] == 1:
        return np.repeat(arr, num, axis=0)
    if LineString is not None:
        try:
            ls = LineString(arr)
            if ls.length <= 1e-8:
                return np.repeat(arr[:1], num, axis=0)
            dists = np.linspace(0.0, ls.length, num=num, dtype=np.float32)
            pts = [list(ls.interpolate(float(d)).coords)[0] for d in dists]
            return np.asarray(pts, dtype=np.float32)
        except Exception:
            pass
    # fallback: linear interp along cumulative arc length of segments
    seg = arr[1:] - arr[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total < 1e-8:
        return np.repeat(arr[:1], num, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    tgrid = np.linspace(0.0, total, num=num, dtype=np.float32)
    out = np.zeros((num, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(tgrid):
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        if j >= len(seg):
            out[i] = arr[-1]
        else:
            alpha = 0.0 if seg_len[j] < 1e-8 else (t - cum[j]) / seg_len[j]
            out[i] = arr[j] + alpha * seg[j]
    return out


def _stack_points(bank: Dict[int, List[np.ndarray]], samples_per_line: int) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for cls_id in (0, 1, 2):
        pts_list: List[np.ndarray] = []
        for arr in bank.get(cls_id, []) or []:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.shape[0] < 1:
                continue
            pts_list.append(_sample_line(arr, num=samples_per_line))
        if len(pts_list) == 0:
            out[cls_id] = np.zeros((0, 2), dtype=np.float32)
        else:
            out[cls_id] = np.concatenate(pts_list, axis=0)
    return out


def chamfer(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Return (d(a->b), d(b->a), sym) using brute-force or blockwise."""
    if a.shape[0] == 0 and b.shape[0] == 0:
        return float('nan'), float('nan'), float('nan')
    if a.shape[0] == 0:
        return float('nan'), float(0.0), float('nan')
    if b.shape[0] == 0:
        return float(0.0), float('nan'), float('nan')
    # blockwise to reduce peak memory
    def nn_mean(src: np.ndarray, dst: np.ndarray, blk: int = 4096) -> float:
        n = src.shape[0]
        mins = np.empty((n,), dtype=np.float32)
        for i in range(0, n, blk):
            s = src[i:i+blk]
            # (blk, 1, 2) - (1, m, 2) => (blk, m)
            diff = s[:, None, :] - dst[None, :, :]
            d2 = (diff * diff).sum(axis=2)
            mins[i:i+blk] = np.sqrt(d2.min(axis=1))
        return float(mins.mean())

    d_ab = nn_mean(a, b)
    d_ba = nn_mean(b, a)
    sym = 0.5 * (d_ab + d_ba)
    return d_ab, d_ba, sym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-pkl', default=None)
    ap.add_argument('--gt-pkl', default=None)
    ap.add_argument('--pred-root', default=None)
    ap.add_argument('--gt-root', default=None)
    ap.add_argument('--scene', default=None)
    ap.add_argument('--samples-per-line', type=int, default=100)
    args = ap.parse_args()

    if args.pred_pkl is None:
        assert args.pred_root and args.scene
        args.pred_pkl = osp.join(args.pred_root, f'{args.scene}.pkl')
    if args.gt_pkl is None:
        assert args.gt_root and args.scene
        args.gt_pkl = osp.join(args.gt_root, f'{args.scene}.pkl')
    assert osp.exists(args.pred_pkl), args.pred_pkl
    assert osp.exists(args.gt_pkl), args.gt_pkl

    pred = _load_pickle(args.pred_pkl)
    gt = _load_pickle(args.gt_pkl)

    P = _stack_points(pred, args.samples_per_line)
    G = _stack_points(gt, args.samples_per_line)

    # per-class Chamfer + overall（并集）
    res: Dict[str, float] = {}
    for name, cls_id in [('ped', 0), ('div', 1), ('bnd', 2)]:
        d1, d2, sym = chamfer(P[cls_id], G[cls_id])
        res[f'{name}_PtoG'] = d1
        res[f'{name}_GtoP'] = d2
        res[f'{name}_Chamfer'] = sym

    Pall = np.concatenate([P[0], P[1], P[2]], axis=0) if any([P[k].size for k in (0,1,2)]) else np.zeros((0,2), np.float32)
    Gall = np.concatenate([G[0], G[1], G[2]], axis=0) if any([G[k].size for k in (0,1,2)]) else np.zeros((0,2), np.float32)
    d1, d2, sym = chamfer(Pall, Gall)
    res['all_PtoG'] = d1
    res['all_GtoP'] = d2
    res['all_Chamfer'] = sym

    # 打印结果
    print(f'# Samples per line: {args.samples_per_line}')
    print(f'# pred: {args.pred_pkl}')
    print(f'# gt  : {args.gt_pkl}')
    for k in ['ped_Chamfer','div_Chamfer','bnd_Chamfer','all_Chamfer',
              'ped_PtoG','div_PtoG','bnd_PtoG','all_PtoG',
              'ped_GtoP','div_GtoP','bnd_GtoP','all_GtoP']:
        v = res.get(k, float('nan'))
        print(f'{k}: {v:.4f}' if isinstance(v, float) and v==v else f'{k}: nan')


if __name__ == '__main__':
    main()

