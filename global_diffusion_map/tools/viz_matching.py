#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import LineString


def _load_pickle(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _uniform_sample(line: LineString, num: int) -> np.ndarray:
    if line.length <= 1e-6:
        p = np.array(line.coords[0], dtype=np.float32)
        return np.tile(p[None, :], (num, 1))
    dists = np.linspace(0.0, line.length, num=num, dtype=np.float32)
    pts = [list(line.interpolate(float(d)).coords)[0] for d in dists]
    return np.asarray(pts, dtype=np.float32)


def _normalize_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)
    out = np.empty_like(xy, dtype=np.float32)
    out[:, 0] = ((xy[:, 0] - minx) / w) * 2.0 - 1.0
    out[:, 1] = ((xy[:, 1] - miny) / h) * 2.0 - 1.0
    return out


def _principal_dir(xy: np.ndarray) -> np.ndarray:
    xy0 = xy - xy.mean(axis=0, keepdims=True)
    try:
        cov = (xy0.T @ xy0) / max(xy.shape[0] - 1, 1)
        w, v = np.linalg.eigh(cov)
        d = v[:, -1]
    except Exception:
        d = xy[-1] - xy[0]
    n = np.linalg.norm(d) + 1e-8
    return d / n


def _hungarian(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        r, c = linear_sum_assignment(C)
        return r.astype(np.int64), c.astype(np.int64)
    except Exception:
        pairs = []
        used_r = set(); used_c = set()
        P, G = C.shape
        flat = [(C[i, j], i, j) for i in range(P) for j in range(G)]
        flat.sort(key=lambda x: x[0])
        for val, i, j in flat:
            if i in used_r or j in used_c:
                continue
            pairs.append((i, j))
            used_r.add(i); used_c.add(j)
        if not pairs:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64)
        rr = np.array([p[0] for p in pairs], dtype=np.int64)
        cc = np.array([p[1] for p in pairs], dtype=np.int64)
        return rr, cc


def _pack_scene(vecs: Dict[int, List[np.ndarray]], bounds: Sequence[float], M: int, num_queries: int, class_budget: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inst_list: List[np.ndarray] = []
    label_list: List[int] = []
    for cls_id in (2, 1, 0):  # boundary, divider, ped (orig ids)
        arrs = vecs.get(cls_id, [])
        scored: List[Tuple[float, np.ndarray]] = []
        for a in arrs:
            try:
                ln = LineString(a).length
            except Exception:
                ln = 0.0
            scored.append((ln, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        cap = int(class_budget.get(cls_id, 0))
        take = scored[:cap]
        for _, pts_arr in take:
            line = LineString(pts_arr)
            sampled = _uniform_sample(line, M)
            normed = _normalize_xy(sampled, bounds)
            inst_list.append(normed)
            # Map to MapTR order: divider=0, ped=1, boundary=2
            if cls_id == 1:
                mapped = 0
            elif cls_id == 0:
                mapped = 1
            else:
                mapped = 2
            label_list.append(mapped)
    pts = np.zeros((1, num_queries, M, 2), dtype=np.float32)
    mask = np.ones((1, num_queries, M), dtype=bool)
    labels = np.zeros((1, num_queries), dtype=np.int64)
    N = min(len(inst_list), num_queries)
    for i in range(N):
        pts[0, i] = inst_list[i]
        labels[0, i] = label_list[i]
        mask[0, i] = False
    return pts, labels, mask


def match_pairs(gt_pts: np.ndarray, gt_mask: np.ndarray, gt_labels: np.ndarray,
                pr_pts: np.ndarray, pr_mask: np.ndarray, pr_labels: np.ndarray,
                w_center: float, w_dir: float, w_pw: float, thres: float) -> Dict[int, List[Tuple[int, int, float]]]:
    B, N, M, _ = gt_pts.shape
    assert B == 1
    result: Dict[int, List[Tuple[int, int, float]]] = {0: [], 1: [], 2: []}
    for c in (0, 1, 2):
        prop_idx = [int(i) for i in range(N) if (not bool(pr_mask[0, i, 0])) and int(pr_labels[0, i]) == c]
        gt_idx   = [int(i) for i in range(N) if (not bool(gt_mask[0, i, 0])) and int(gt_labels[0, i]) == c]
        if len(prop_idx) == 0 or len(gt_idx) == 0:
            continue
        C = np.zeros((len(prop_idx), len(gt_idx)), dtype=np.float32)
        for pi, ii in enumerate(prop_idx):
            for gj, jj in enumerate(gt_idx):
                cost = 0.0
                if w_center > 0:
                    cp = pr_pts[0, ii].mean(axis=0); cg = gt_pts[0, jj].mean(axis=0)
                    cost += w_center * float(np.linalg.norm(cp - cg))
                if w_dir > 0:
                    dp = _principal_dir(pr_pts[0, ii]); dg = _principal_dir(gt_pts[0, jj])
                    dot = float(abs((dp * dg).sum()) / (np.linalg.norm(dp) * np.linalg.norm(dg) + 1e-8))
                    cost += w_dir * (1.0 - dot)
                if w_pw > 0:
                    cost += w_pw * float(np.linalg.norm(pr_pts[0, ii] - gt_pts[0, jj], axis=-1).mean())
                C[pi, gj] = cost
        rr, cc = _hungarian(C)
        pairs = []
        for k in range(len(rr)):
            if C[int(rr[k]), int(cc[k])] <= thres:
                pairs.append((prop_idx[int(rr[k])], gt_idx[int(cc[k])], float(C[int(rr[k]), int(cc[k])])))
        # sort pairs by gt length for stable numbering
        pairs_sorted = []
        for (pi, gi, co) in pairs:
            g_xy = gt_pts[0, gi]
            ln = float(np.linalg.norm(g_xy[1:] - g_xy[:-1], axis=-1).sum())
            pairs_sorted.append((pi, gi, ln))
        pairs_sorted.sort(key=lambda x: x[2], reverse=True)
        result[c] = pairs_sorted
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--static-root', required=True)
    ap.add_argument('--aggregated-root', required=True)
    ap.add_argument('--stats-json', default='global_diffusion_map/work_dirs/av2_stats.json')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--scenes', nargs='*', default=None)
    ap.add_argument('--max-scenes', type=int, default=5)
    ap.add_argument('--w-center', type=float, default=1.0)
    ap.add_argument('--w-dir', type=float, default=0.2)
    ap.add_argument('--w-pw', type=float, default=0.5)
    ap.add_argument('--match-thres', type=float, default=0.6)
    ap.add_argument('--dpi', type=int, default=120)
    args = ap.parse_args()

    with open(args.stats_json, 'r') as f:
        stats = json.load(f)
    M = int(stats.get('M', 20))
    num_queries = int(stats.get('num_queries', 50))
    class_budget = {int(k): int(v) for k, v in stats.get('class_budget', {0: 8, 1: 17, 2: 25}).items()}

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted([osp.splitext(x)[0] for x in os.listdir(args.static_root) if x.endswith('.pkl')])
    if args.max_scenes and len(scenes) > args.max_scenes:
        scenes = scenes[:args.max_scenes]

    os.makedirs(args.out_dir, exist_ok=True)

    for s in scenes:
        gt_pkl = osp.join(args.static_root, f'{s}.pkl')
        ap_pkl = osp.join(args.aggregated_root, f'{s}.pkl')
        if not (osp.exists(gt_pkl) and osp.exists(ap_pkl)):
            continue
        gt = _load_pickle(gt_pkl)
        apd = _load_pickle(ap_pkl)
        bounds = gt.get('bounds', None)
        if bounds is None:
            pts_all = []
            for k in (0, 1, 2):
                for arr in gt.get(k, []):
                    pts_all.append(arr)
            if not pts_all:
                continue
            cat = np.concatenate(pts_all, axis=0)
            minx, miny = cat.min(0); maxx, maxy = cat.max(0)
            bounds = [float(minx), float(miny), float(maxx), float(maxy)]

        gt_pts, gt_labs, gt_mask = _pack_scene(gt, bounds, M, num_queries, class_budget)
        pr_pts, pr_labs, pr_mask = _pack_scene(apd, bounds, M, num_queries, class_budget)

        pairs = match_pairs(gt_pts, gt_mask, gt_labs, pr_pts, pr_mask, pr_labs,
                            w_center=args.w_center, w_dir=args.w_dir, w_pw=args.w_pw, thres=args.match_thres)

        # Prepare split panels
        minx, miny, maxx, maxy = map(float, bounds)
        fig = plt.figure(figsize=(7, 12))
        ax_gt = fig.add_subplot(2, 1, 1)
        ax_pr = fig.add_subplot(2, 1, 2)
        for ax in (ax_gt, ax_pr):
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
            ax.set_aspect('equal', adjustable='box')
            ax.set_facecolor('white'); ax.axis('off')
        ax_gt.set_title(f'GT — {s}', fontsize=10)
        ax_pr.set_title('Proposal', fontsize=10)

        cls_name = {0: 'divider', 1: 'ped', 2: 'boundary'}
        color_gt = {0: 'r', 1: 'b', 2: 'g'}
        color_prop = {0: '#ffaaaa', 1: '#aaaaff', 2: '#aaffaa'}

        # Draw matched couples per class with same tag on both panels
        for c in (0, 1, 2):
            items = pairs.get(c, [])
            for idx, (pi, gi, _ln) in enumerate(items, start=1):
                tag = f"{cls_name.get(c, 'cls')}{idx:02d}"
                # GT polyline
                g_xy = gt_pts[0, gi]
                g_den = np.empty_like(g_xy)
                g_den[:, 0] = (g_xy[:, 0] + 1.0) * 0.5 * (maxx - minx) + minx
                g_den[:, 1] = (g_xy[:, 1] + 1.0) * 0.5 * (maxy - miny) + miny
                ax_gt.plot(g_den[:, 0], g_den[:, 1], color=color_gt.get(c, 'k'), linewidth=2.0, alpha=0.95)
                # Proposal polyline
                p_xy = pr_pts[0, pi]
                p_den = np.empty_like(p_xy)
                p_den[:, 0] = (p_xy[:, 0] + 1.0) * 0.5 * (maxx - minx) + minx
                p_den[:, 1] = (p_xy[:, 1] + 1.0) * 0.5 * (maxy - miny) + miny
                ax_pr.plot(p_den[:, 0], p_den[:, 1], color=color_prop.get(c, '#cccccc'), linewidth=1.5, alpha=0.9, linestyle='--')
                # Labels
                gc = g_xy.mean(axis=0)
                gxy = np.array([(gc[0] + 1.0) * 0.5 * (maxx - minx) + minx, (gc[1] + 1.0) * 0.5 * (maxy - miny) + miny])
                pc = p_xy.mean(axis=0)
                pxy = np.array([(pc[0] + 1.0) * 0.5 * (maxx - minx) + minx, (pc[1] + 1.0) * 0.5 * (maxy - miny) + miny])
                offset = np.array([0.01 * (maxx - minx), 0.01 * (maxy - miny)])
                ax_gt.text(*(gxy + offset), tag, fontsize=9, color='black', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.2'))
                ax_gt.plot([gxy[0]+offset[0], gxy[0]], [gxy[1]+offset[1], gxy[1]], color='black', linewidth=0.8, alpha=0.7)
                ax_pr.text(*(pxy + offset), tag, fontsize=9, color='black', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.2'))
                ax_pr.plot([pxy[0]+offset[0], pxy[0]], [pxy[1]+offset[1], pxy[1]], color='black', linewidth=0.8, alpha=0.7)

        # Draw unmatched GT-only and proposal-only
        for c in (0, 1, 2):
            matched_g = set(gi for _, gi, _ in pairs.get(c, []))
            matched_p = set(pi for pi, _, _ in pairs.get(c, []))
            # GT-only
            gt_only = [i for i in range(num_queries) if (not bool(gt_mask[0, i, 0])) and int(gt_labs[0, i]) == c and i not in matched_g]
            for uidx, gi in enumerate(gt_only, start=1):
                tag = f"{cls_name.get(c,'cls')}_GT_only{uidx:02d}"
                g_xy = gt_pts[0, gi]
                g_den = np.empty_like(g_xy)
                g_den[:, 0] = (g_xy[:, 0] + 1.0) * 0.5 * (maxx - minx) + minx
                g_den[:, 1] = (g_xy[:, 1] + 1.0) * 0.5 * (maxy - miny) + miny
                ax_gt.plot(g_den[:, 0], g_den[:, 1], color=color_gt.get(c, 'k'), linewidth=2.0, alpha=0.95)
                gc = g_xy.mean(axis=0)
                gxy = np.array([(gc[0] + 1.0) * 0.5 * (maxx - minx) + minx, (gc[1] + 1.0) * 0.5 * (maxy - miny) + miny])
                offset = np.array([0.01 * (maxx - minx), 0.01 * (maxy - miny)])
                ax_gt.text(*(gxy + offset), tag, fontsize=9, color='black', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.2'))
                ax_gt.plot([gxy[0]+offset[0], gxy[0]], [gxy[1]+offset[1], gxy[1]], color='black', linewidth=0.8, alpha=0.7)
            # Proposal-only
            pr_only = [i for i in range(num_queries) if (not bool(pr_mask[0, i, 0])) and int(pr_labs[0, i]) == c and i not in matched_p]
            for uidx, pi in enumerate(pr_only, start=1):
                tag = f"{cls_name.get(c,'cls')}_PROP_only{uidx:02d}"
                p_xy = pr_pts[0, pi]
                p_den = np.empty_like(p_xy)
                p_den[:, 0] = (p_xy[:, 0] + 1.0) * 0.5 * (maxx - minx) + minx
                p_den[:, 1] = (p_xy[:, 1] + 1.0) * 0.5 * (maxy - miny) + miny
                ax_pr.plot(p_den[:, 0], p_den[:, 1], color=color_prop.get(c, '#cccccc'), linewidth=1.5, alpha=0.9, linestyle='--')
                pc = p_xy.mean(axis=0)
                pxy = np.array([(pc[0] + 1.0) * 0.5 * (maxx - minx) + minx, (pc[1] + 1.0) * 0.5 * (maxy - miny) + miny])
                offset = np.array([0.01 * (maxx - minx), 0.01 * (maxy - miny)])
                ax_pr.text(*(pxy + offset), tag, fontsize=9, color='black', ha='left', va='bottom', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.2'))
                ax_pr.plot([pxy[0]+offset[0], pxy[0]], [pxy[1]+offset[1], pxy[1]], color='black', linewidth=0.8, alpha=0.7)

        out_png = osp.join(args.out_dir, f'{s}_match.png')
        os.makedirs(args.out_dir, exist_ok=True)
        plt.tight_layout(); fig.savefig(out_png, dpi=args.dpi); plt.close(fig)
        print(f'[ok] saved {out_png}')


if __name__ == '__main__':
    main()
