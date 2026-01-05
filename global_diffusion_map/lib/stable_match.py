#!/usr/bin/env python
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _centroid(curve: np.ndarray) -> np.ndarray:
    return np.asarray(curve, dtype=np.float32).mean(axis=0)


def _direction(curve: np.ndarray) -> np.ndarray:
    pts = np.asarray(curve, dtype=np.float32)
    if pts.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    v = pts[-1] - pts[0]
    n = float(np.linalg.norm(v) + 1e-8)
    if n == 0.0:
        # PCA fallback
        c = pts - pts.mean(axis=0, keepdims=True)
        u, s, vh = np.linalg.svd(c, full_matrices=False)
        v = vh[0]
        n = float(np.linalg.norm(v) + 1e-8)
    return (v / n).astype(np.float32)


def _downsample(curve: np.ndarray, m: int) -> np.ndarray:
    pts = np.asarray(curve, dtype=np.float32)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((m, 2), dtype=np.float32)
    if n == m:
        return pts.copy()
    idx = np.linspace(0, n - 1, num=m)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    t = (idx - lo).astype(np.float32)[:, None]
    return (1.0 - t) * pts[lo] + t * pts[hi]


def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [Pa,2], b: [Pb,2]
    aa = (a**2).sum(1, keepdims=True)
    bb = (b**2).sum(1, keepdims=True).T
    ab = a @ b.T
    d2 = np.maximum(aa + bb - 2.0 * ab, 0.0)
    return np.sqrt(d2, dtype=np.float32)


def _simplified_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: [m,2]
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    D = _pairwise_dist(a, b)
    fwd = float(D.min(axis=1).mean())
    bwd = float(D.min(axis=0).mean())
    return 0.5 * (fwd + bwd)


def _build_class_indices(labels: np.ndarray, valid: np.ndarray) -> Dict[int, List[int]]:
    cls2idx: Dict[int, List[int]] = {}
    for j in range(labels.shape[0]):
        if not valid[j]:
            continue
        c = int(labels[j])
        cls2idx.setdefault(c, []).append(j)
    return cls2idx


def stable_match_reorder(
    prop_np: np.ndarray,  # [B,N,M,2] proposals in normalized coords; first N may be valid
    tgt_np: np.ndarray,   # [B,N,M,2] targets (e.g., tiled mu centers)
    labs_np: np.ndarray,  # [B,num_queries] MapTR labels per slot (0:divider,1:ped,2:boundary)
    mask_np: np.ndarray,  # [B,num_queries,M] True for padding
    prop_labels: Optional[np.ndarray] = None,  # [N] MapTR labels per proposal (0/1/2)
    w_center: float = 1.0,
    w_dir: float = 0.2,
    w_pw: float = 0.5,
    thres: float = 0.20,
    cand_radius_frac: float = 0.50,
    m_pw: int = 8,
    per_class: Optional[Dict[int, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder proposals per class to match target slot order using Hungarian.

    - Costs computed in normalized space [-1,1].
    - thres in normalized units (fraction of diag, ~2.828), but applied to raw cost sum here.
    - Returns new_pts [B,num_queries,M,2], new_mask [B,num_queries,M].
    """
    B, N, M, _ = prop_np.shape
    assert B == 1, "Only batch=1 supported for now"
    Q = labs_np.shape[1]
    new_pts = np.zeros((B, Q, M, 2), dtype=np.float32)
    new_mask = np.ones((B, Q, M), dtype=bool)

    # Valid slots are those with any False in mask (i.e., not all padded)
    valid_slots = np.array([not mask_np[0, j].all() for j in range(Q)], dtype=bool)
    labels = labs_np[0]  # [Q]

    # Extract valid proposals (first N by construction)
    prop_list = [prop_np[0, i] for i in range(N)]
    if prop_labels is None:
        prop_labs = labels[:N]
    else:
        prop_labs = np.asarray(prop_labels).astype(int)
    # Targets per slot
    tgt_list = [tgt_np[0, j] for j in range(Q)]

    # By class matching
    cls2slots = _build_class_indices(labels, valid_slots)
    for cls, slot_ids in cls2slots.items():
        # proposals of this class
        prop_ids = [i for i in range(N) if int(prop_labs[i]) == int(cls)]
        if not prop_ids:
            continue
        # Build costs
        P = len(prop_ids)
        S = len(slot_ids)
        C = np.zeros((P, S), dtype=np.float32)
        # Per-class overrides
        wc = float(per_class.get(int(cls), {}).get('w_center', w_center)) if per_class else float(w_center)
        wd = float(per_class.get(int(cls), {}).get('w_dir', w_dir)) if per_class else float(w_dir)
        wp = float(per_class.get(int(cls), {}).get('w_pw', w_pw)) if per_class else float(w_pw)
        thr = float(per_class.get(int(cls), {}).get('thres', thres)) if per_class else float(thres)
        cr = float(per_class.get(int(cls), {}).get('cand_radius_frac', cand_radius_frac)) if per_class else float(cand_radius_frac)
        # Precompute per proposal features
        p_cent = np.stack([_centroid(prop_list[i]) for i in prop_ids], axis=0)
        p_dir = np.stack([_direction(prop_list[i]) for i in prop_ids], axis=0)
        # For candidate radius filter, compute target centers
        t_cent = np.stack([_centroid(tgt_list[j]) for j in slot_ids], axis=0)
        # Normalize distance by normalized-space diagonal ~2.828
        diag = math.sqrt(8.0)
        # Candidate filter: large initial cost
        C[:] = 1e3
        for pi, i in enumerate(prop_ids):
            for sj, j in enumerate(slot_ids):
                dc = float(np.linalg.norm(p_cent[pi] - t_cent[sj])) / diag
                if dc > max(cr, thr * 2.0):
                    continue
                # Direction term
                t_dir = _direction(tgt_list[j])
                cosv = float(np.abs(np.dot(p_dir[pi], t_dir)))
                dtheta = 1.0 - cosv
                # Simplified Chamfer on downsampled curves
                a = _downsample(prop_list[i], m_pw)
                b = _downsample(tgt_list[j], m_pw)
                dch = _simplified_chamfer(a, b) / diag
                C[pi, sj] = wc * dc + wd * dtheta + wp * dch

        # Hungarian matching (rectangular): scipy if available; else greedy fallback
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            row_ind, col_ind = linear_sum_assignment(C)
            pairs = [(prop_ids[r], slot_ids[c], float(C[r, c])) for r, c in zip(row_ind, col_ind)]
        except Exception:
            # Greedy fallback
            pairs: List[Tuple[int, int, float]] = []
            used_p: set = set()
            used_s: set = set()
            flat = [(float(C[pi, sj]), pi, sj) for pi in range(P) for sj in range(S)]
            flat.sort(key=lambda x: x[0])
            for cost, pi, sj in flat:
                if pi in used_p or sj in used_s:
                    continue
                if cost >= 1e3:
                    break
                used_p.add(pi); used_s.add(sj)
                pairs.append((prop_ids[pi], slot_ids[sj], float(cost)))

        # Apply threshold gate and assign
        for i_id, j_id, cost in pairs:
            if not math.isfinite(cost) or cost > thr or cost >= 1e3:
                continue
            # Optional shape gate by pairwise (normalized) Chamfer
            pw_gate = None
            if per_class and int(cls) in per_class:
                pw_gate = per_class[int(cls)].get('pw_thres', None)
            if pw_gate is not None:
                a = _downsample(prop_list[i_id], m_pw)
                b = _downsample(tgt_list[j_id if j_id < len(tgt_list) else slot_ids[0]], m_pw)
                dch = _simplified_chamfer(a, b) / math.sqrt(8.0)
                if float(dch) > float(pw_gate):
                    continue
            new_pts[0, j_id] = prop_list[i_id]
            new_mask[0, j_id] = False

    return new_pts, new_mask
