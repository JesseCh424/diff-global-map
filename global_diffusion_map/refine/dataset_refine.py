from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RefineCaps:
    num_queries: int  # N
    num_points: int   # P


@dataclass
class RefineDataItem:
    # Fixed-size hybrid tensor (proposal + gaussian noise), normalized to [-1,1]
    x_init: np.ndarray         # [N, P, 2]
    mask: np.ndarray           # [N, P] True for padding
    slot_labels: np.ndarray    # [N] MapTR-style labels per slot (0:divider,1:ped,2:boundary)
    # Targets (optional during train)
    gt_pts: Optional[np.ndarray] = None  # [N, P, 2]
    gt_mask: Optional[np.ndarray] = None # [N, P]
    gt_cls: Optional[np.ndarray] = None  # [N] 1 for matched, 0 for background


def _normalize_xy(xy: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)
    out = xy.astype(np.float32).copy()
    out[..., 0] = ((xy[..., 0] - minx) / w) * 2.0 - 1.0
    out[..., 1] = ((xy[..., 1] - miny) / h) * 2.0 - 1.0
    return out


def _uniform_resample(poly: np.ndarray, m: int) -> np.ndarray:
    # Linear arc-length resample (lightweight)
    pts = np.asarray(poly, dtype=np.float32)
    if pts.shape[0] <= 1:
        return np.tile(pts[:1], (m, 1)) if pts.size else np.zeros((m, 2), np.float32)
    d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1]) if s[-1] > 1e-9 else 1.0
    targets = np.linspace(0.0, total, num=m, dtype=np.float32)
    out = np.zeros((m, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < s.size and s[j + 1] < t:
            j += 1
        if j + 1 >= pts.shape[0]:
            out[i] = pts[-1]
        else:
            a, b = pts[j], pts[j + 1]
            u = (t - s[j]) / max(s[j + 1] - s[j], 1e-9)
            out[i] = (1.0 - u) * a + u * b
    return out


def _jitter(poly: np.ndarray, sigma_m: float) -> np.ndarray:
    return poly + np.random.normal(scale=sigma_m, size=poly.shape).astype(np.float32)


def simulate_upstream_error(
    gt_polys: Dict[int, List[np.ndarray]],
    bounds: Sequence[float],
    caps: RefineCaps,
    drop_rate: float = 0.3,
    jitter_sigma_m: float = 0.5,
    ghosts: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-size hybrid tensor with simulated upstream characteristics.

    Returns:
      x_init  [N,P,2]  proposals (resampled+jitter) then gaussian noise
      mask    [N,P]
      labels  [N]
      gt_mask [N,P]  True for padding in targets (after packing by budgets)
    """
    N, P = caps.num_queries, caps.num_points
    # Build proposal list per class (drop & jitter)
    prop_pts: List[np.ndarray] = []
    prop_labs: List[int] = []
    for orig in (2, 1, 0):  # boundary, divider, ped (orig ids)
        vecs = gt_polys.get(orig, [])
        for arr in vecs:
            if random.random() < drop_rate:
                continue  # simulate miss
            res = _uniform_resample(np.asarray(arr), P)
            res = _jitter(res, jitter_sigma_m)
            prop_pts.append(_normalize_xy(res, bounds))
            # MapTR labels: divider=0, ped=1, boundary=2
            lab = 0 if orig == 1 else (1 if orig == 0 else 2)
            prop_labs.append(lab)
    # Inject ghosts (random curves)
    for _ in range(max(0, ghosts)):
        base = np.random.uniform(low=[-1, -1], high=[1, 1], size=(P, 2)).astype(np.float32)
        prop_pts.append(base)
        prop_labs.append(np.random.choice([0, 1, 2]))

    # Pack to fixed slots (truncate or pad with noise)
    x = np.zeros((N, P, 2), dtype=np.float32)
    m = np.ones((N, P), dtype=bool)
    labs = np.zeros((N,), dtype=np.int64)
    k = min(N, len(prop_pts))
    for i in range(k):
        x[i] = prop_pts[i]
        labs[i] = int(prop_labs[i])
        m[i] = False
    # Gaussian noise slots (generation zone)
    if k < N:
        x[k:] = np.random.normal(scale=1.0, size=(N - k, P, 2)).astype(np.float32)
        # labels for noise slots can be left at 0; they are background before matching

    # Target mask: by default unknown here（由上层装载 GT 后再匹配）
    gt_mask = np.ones((N, P), dtype=bool)
    return x, m, labs, gt_mask


def pack_gt_to_slots(
    gt_polys: Dict[int, List[np.ndarray]],
    bounds: Sequence[float],
    budgets: Dict[int, int],
    num_points: int,
    num_queries: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack GT polylines into fixed slots by per-class budgets.

    Slot label order (MapTR): divider(0)×cap + ped(1)×cap + boundary(2)×cap.
    Returns (coords [N,P,2], mask [N,P], present [N]).
    """
    N, P = int(num_queries), int(num_points)
    order: List[int] = []
    for orig in (1, 0, 2):  # to MapTR labels
        cap = int(budgets.get(orig, 0))
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        order += [lab] * max(0, cap)
    x = np.zeros((N, P, 2), dtype=np.float32)
    m = np.ones((N, P), dtype=bool)
    present = np.zeros((N,), dtype=np.int64)
    ptr = {0: 0, 1: 0, 2: 0}
    for slot, lab in enumerate(order[:N]):
        orig = 1 if lab == 0 else (0 if lab == 1 else 2)
        arrs = gt_polys.get(orig, [])
        if ptr[lab] >= len(arrs):
            continue
        pts = _uniform_resample(np.asarray(arrs[ptr[lab]]), P)
        x[slot] = _normalize_xy(pts, bounds)
        m[slot] = False
        present[slot] = 1
        ptr[lab] += 1
    return x, m, present


def pack_vectors_to_slots(
    vecs: Dict[int, List[np.ndarray]],
    bounds: Sequence[float],
    budgets: Dict[int, int],
    num_points: int,
    num_queries: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack arbitrary vectors (orig ids: 0:ped,1:divider,2:boundary) into fixed slots.

    Returns (coords [N,P,2], mask [N,P], labels [N]) where labels are MapTR
    order per slot (0:divider,1:ped,2:boundary). Unused slots are masked.
    """
    N, P = int(num_queries), int(num_points)
    order: List[int] = []
    for orig in (1, 0, 2):  # to MapTR labels
        cap = int(budgets.get(orig, 0))
        lab = 0 if orig == 1 else (1 if orig == 0 else 2)
        order += [lab] * max(0, cap)
    x = np.zeros((N, P, 2), dtype=np.float32)
    m = np.ones((N, P), dtype=bool)
    labs = np.zeros((N,), dtype=np.int64)
    # per class pointer
    ptr = {0: 0, 1: 0, 2: 0}
    pool = {0: vecs.get(1, []), 1: vecs.get(0, []), 2: vecs.get(2, [])}
    for slot, lab in enumerate(order[:N]):
        arrs = pool.get(lab, [])
        if ptr[lab] >= len(arrs):
            continue
        pts = _uniform_resample(np.asarray(arrs[ptr[lab]]), P)
        x[slot] = _normalize_xy(pts, bounds)
        m[slot] = False
        labs[slot] = lab
        ptr[lab] += 1
    return x, m, labs
