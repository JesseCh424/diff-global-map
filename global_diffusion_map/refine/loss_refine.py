from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import os
import os

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


def _pairwise_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: [P,2]
    return (a - b).abs().mean()


def bidirectional_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Polyline orientation agnostic L1.
    pred/target: [P,2]
    """
    fwd = _pairwise_l1(pred, target)
    bwd = _pairwise_l1(pred, torch.flip(target, dims=[0]))
    return torch.minimum(fwd, bwd)


class HungarianMatcher:
    """Lightweight matcher on per-slot normalized curves.
    Uses center, direction and simplified Chamfer as costs.
    """

    def __init__(self, w_center: float = 1.0, w_dir: float = 0.2, w_pw: float = 0.5) -> None:
        self.wc = float(w_center)
        self.wd = float(w_dir)
        self.wp = float(w_pw)

    @staticmethod
    def _principal_dir(xy: np.ndarray) -> np.ndarray:
        # Robust principal direction with NaN-safe handling.
        # Filter out non-finite points first
        xy = np.asarray(xy, dtype=np.float32)
        mask = np.isfinite(xy).all(axis=-1)
        xyv = xy[mask]
        if xyv.shape[0] < 2:
            return np.array([1.0, 0.0], dtype=np.float32)
        v = xyv[-1] - xyv[0]
        n = float(np.linalg.norm(v))
        # Fallback to PCA if degenerate or non-finite
        if (not np.isfinite(n)) or (n < 1e-8):
            c = xyv - xyv.mean(0, keepdims=True)
            try:
                _u, _s, vh = np.linalg.svd(c, full_matrices=False)
                v = vh[0]
                n = float(np.linalg.norm(v))
            except Exception:
                return np.array([1.0, 0.0], dtype=np.float32)
        if (not np.isfinite(n)) or (n < 1e-8):
            return np.array([1.0, 0.0], dtype=np.float32)
        return (v / max(n, 1e-8)).astype(np.float32)

    @staticmethod
    def _simple_chamfer(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        # Filter non-finite rows to avoid NaNs propagating
        a = a[np.isfinite(a).all(axis=-1)]
        b = b[np.isfinite(b).all(axis=-1)]
        if a.size == 0 or b.size == 0:
            return 0.0
        # downsample to reduce cost
        m = min(a.shape[0], b.shape[0], 16)
        ids_a = np.linspace(0, a.shape[0] - 1, num=m).round().astype(int)
        ids_b = np.linspace(0, b.shape[0] - 1, num=m).round().astype(int)
        aa, bb = a[ids_a], b[ids_b]
        da = np.sqrt(((aa[:, None, :] - bb[None, :, :]) ** 2).sum(-1)).min(1).mean()
        db = np.sqrt(((bb[:, None, :] - aa[None, :, :]) ** 2).sum(-1)).min(1).mean()
        return float(0.5 * (da + db))

    def __call__(self, pred_pts: np.ndarray, gt_pts: np.ndarray) -> List[Tuple[int, int]]:
        # pred_pts / gt_pts: [N,P,2], normalized
        N = pred_pts.shape[0]
        M = gt_pts.shape[0]
        if N == 0 or M == 0:
            return []
        C = np.zeros((N, M), dtype=np.float32)
        diag = np.sqrt(8.0)
        use_opt = os.environ.get('REFINE_OPT_MATCH', '0') == '1'
        if use_opt:
            # Optimized: precompute GT centers and principal directions once (仍然是 Hungarian)
            gt_centers = np.zeros((M, 2), dtype=np.float32)
            gt_dirs = np.zeros((M, 2), dtype=np.float32)
            for j in range(M):
                gj = gt_pts[j]
                if gj.size == 0:
                    gt_centers[j] = np.array([0.0, 0.0], dtype=np.float32)
                    if self.wd > 0.0:
                        gt_dirs[j] = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    gt_centers[j] = np.nanmean(gj, axis=0).astype(np.float32)
                    if self.wd > 0.0:
                        gt_dirs[j] = self._principal_dir(gj)
            for i in range(N):
                pi = pred_pts[i]
                pc = (np.nanmean(pi, axis=0).astype(np.float32)
                      if pi.size > 0 else np.array([0.0, 0.0], dtype=np.float32))
                if self.wd > 0.0:
                    pd = self._principal_dir(pi)
                for j in range(M):
                    dc = float(np.linalg.norm(pc - gt_centers[j])) / diag
                    dtheta = 0.0
                    if self.wd > 0.0:
                        dtheta = 1.0 - float(abs(float(np.dot(pd, gt_dirs[j]))))
                    dch = self._simple_chamfer(pi, gt_pts[j]) / diag
                    C[i, j] = self.wc * dc + self.wd * dtheta + self.wp * dch
        else:
            # 原始：逐对计算中心和主方向（方便对齐训练1复现）
            for i in range(N):
                pi = pred_pts[i]
                pc = (np.nanmean(pi, axis=0).astype(np.float32)
                      if pi.size > 0 else np.array([0.0, 0.0], dtype=np.float32))
                # Only compute principal direction if it's used
                if self.wd > 0.0:
                    pd = self._principal_dir(pi)
                for j in range(M):
                    gj = gt_pts[j]
                    gc = (np.nanmean(gj, axis=0).astype(np.float32)
                          if gj.size > 0 else np.array([0.0, 0.0], dtype=np.float32))
                    if self.wd > 0.0:
                        gd = self._principal_dir(gj)
                    dc = float(np.linalg.norm(pc - gc)) / diag
                    dtheta = 0.0
                    if self.wd > 0.0:
                        dtheta = 1.0 - float(abs(float(np.dot(pd, gd))))
                    dch = self._simple_chamfer(pi, gj) / diag
                    C[i, j] = self.wc * dc + self.wd * dtheta + self.wp * dch
        # Try multiple backends for Hungarian matching with safe fallbacks
        # 1) Forced greedy (environment toggle)
        if os.environ.get('REFINE_GREEDY_MATCH', '0') == '1':
            r = c = None  # type: ignore[assignment]
        else:
            # 2) Optional CuPy-based path (requires cupyx SciPy optimize; not always available)
            if os.environ.get('REFINE_CUPY_MATCH', '0') == '1':
                try:
                    import cupy as _cp  # type: ignore
                    from cupyx.scipy.optimize import linear_sum_assignment as _gpu_lsa  # type: ignore
                    _Cg = _cp.asarray(C)
                    _rg, _cg = _gpu_lsa(_Cg)
                    r = _cp.asnumpy(_rg)
                    c = _cp.asnumpy(_cg)
                except Exception:
                    r = c = None  # fall through to next backend
            else:
                r = c = None

            # 3) lap (Jonker–Volgenant) CPU backend — typically faster than SciPy
            if (r is None) or (c is None):
                try:
                    import lap  # type: ignore
                    # lap.lapjv returns (cost, x, y), where x maps rows->cols, -1 for unassigned
                    _cost, x, _y = lap.lapjv(C.astype(np.float64))  # type: ignore
                    r = np.arange(N, dtype=np.int64)
                    c = np.asarray(x, dtype=np.int64)
                except Exception:
                    r = c = None

            # 4) SciPy CPU backend
            if (r is None) or (c is None):
                try:
                    from scipy.optimize import linear_sum_assignment  # type: ignore
                    r, c = linear_sum_assignment(C)
                except Exception:
                    r = c = None

        if (r is None) or (c is None):
            # 5) Greedy fallback (deterministic, non-optimal but fast and dependency-free)
            used_r, used_c = set(), set()
            rc = []
            flat = [(C[i, j], i, j) for i in range(N) for j in range(M)]
            flat.sort(key=lambda x: x[0])
            for _, i, j in flat:
                if i in used_r or j in used_c:
                    continue
                used_r.add(i); used_c.add(j); rc.append((i, j))
            return rc
        # Normalize outputs to a list of (row, col) pairs using true row indices
        r = np.asarray(r).reshape(-1)
        c = np.asarray(c).reshape(-1)
        pairs: List[Tuple[int, int]] = []
        for ri, cj in zip(r, c):
            i = int(ri)
            j = int(cj)
            if 0 <= i < N and 0 <= j < M:
                pairs.append((i, j))
        return pairs
        return [(int(i), int(j)) for i, j in zip(r, c)]


def _build_gt_permutations(gt_pts: torch.Tensor, gt_mask: torch.Tensor,
                           modes: Optional[Sequence[str]] = None) -> Tuple[torch.Tensor, int]:
    """Build simple permutation set for GT polylines.
    - Always include forward and reversed along the point dimension.
    - masks remain aligned per permutation.
    Returns tensor of shape [M, K, P, 2], and K (num perms).
    """
    if modes is None:
        modes = ("forward", "reverse")
    M, P, _ = gt_pts.shape
    perms: List[torch.Tensor] = []
    for m in modes:
        if m == "forward":
            perms.append(gt_pts)
        elif m == "reverse":
            perms.append(torch.flip(gt_pts, dims=[1]))
        else:
            # unsupported mode, skip
            continue
    gt_perm = torch.stack(perms, dim=1)  # [M,K,P,2]
    return gt_perm, gt_perm.shape[1]


def hungarian_match_perm(
    pred_pts: torch.Tensor,             # [N,P,2]
    pred_sem_logits: Optional[torch.Tensor],  # [N,C] or None
    gt_pts: torch.Tensor,               # [M,P,2]
    gt_mask: torch.Tensor,              # [M,P] True=pad
    gt_labels: Optional[torch.Tensor],  # [M] in {0..C-1} or None
    cls_weight: float = 5.0,
    reg_weight: float = 50.0,
    use_l1_beta: float = 0.0,
) -> Tuple[List[Tuple[int,int]], np.ndarray]:
    """Permutation-invariant Hungarian matching (MapTR-style).
    - Builds forward/reverse permutations for GT.
    - Regression cost: Lines L1 per point (orientation-agnostic via perms)
      normalized by P (valid points only), similar to LinesL1Cost.
    - Classification cost (optional): -sigmoid(sem_logits)[gt_label]
    Returns:
      pairs: list of (pred_idx, gt_idx)
      gt_perm_idx: numpy array of shape [M] default -1, where for each matched GT,
                   the chosen permutation index is stored; for unmatched, -1.
    """
    device = pred_pts.device
    N, P, _ = pred_pts.shape
    M = gt_pts.shape[0]
    if N == 0 or M == 0:
        return [], np.full((M,), -1, dtype=np.int64)
    # Build permutations (forward, reverse)
    gt_perm, K = _build_gt_permutations(gt_pts, gt_mask)
    # Regression cost per permutation
    # Shapes: pred_flat [N,2P], gt_perm_flat [M,K,2P]
    pred_flat = pred_pts.reshape(N, -1).float()
    gt_perm_flat = gt_perm.reshape(M, K, -1).float()
    # Compute per-pair L1 normalized by valid points (use mask to count)
    # Expand for broadcasting
    pred_exp = pred_flat[:, None, None, :]  # [N,1,1,2P]
    gt_exp = gt_perm_flat[None, :, :, :]    # [1,M,K,2P]
    # L1 per element then sum last dim
    l1 = torch.abs(pred_exp - gt_exp).sum(dim=-1)  # [N,M,K]
    # Normalize by number of valid points P_v per GT (same for both perms)
    valid_points = (~gt_mask).float().sum(dim=1).clamp_min(1.0)  # [M]
    l1 = l1 / valid_points[None, :, None]
    # Take min over K perms for regression cost and record argmin (perm index)
    reg_cost, perm_idx = l1.min(dim=2)  # [N,M], [N,M]
    cost = reg_weight * reg_cost
    # Classification cost if provided
    if (pred_sem_logits is not None) and (gt_labels is not None):
        # sigmoided confidence for correct class
        prob = torch.sigmoid(pred_sem_logits.float())  # [N,C]
        # index class per GT: build [N,M] matrix of -p[n, label_m]
        C = prob.shape[-1]
        labels = gt_labels.long().clamp(min=0, max=C-1)
        cls_sel = prob[:, labels] if prob.dim() == 2 else prob
        cls_cost = -cls_sel  # [N,M]
        cost = cost + cls_weight * cls_cost
    # Hungarian solve on CPU
    C_np = cost.detach().cpu().numpy().astype(np.float64)
    r = c = None
    try:
        import lap  # type: ignore
        _cost, x, _y = lap.lapjv(C_np)
        r = np.arange(N, dtype=np.int64)
        c = np.asarray(x, dtype=np.int64)
    except Exception:
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            r, c = linear_sum_assignment(C_np)
        except Exception:
            r = c = None
    pairs: List[Tuple[int,int]] = []
    gt_perm_choice = np.full((M,), -1, dtype=np.int64)
    if (r is None) or (c is None):
        # Greedy fallback
        flat = [(C_np[i, j], i, j) for i in range(N) for j in range(M)]
        flat.sort(key=lambda x: x[0])
        used_r, used_c = set(), set()
        for _, i, j in flat:
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j); pairs.append((i, j))
            # record chosen perm index at (i,j)
            k = int(perm_idx[i, j].detach().cpu().item())
            gt_perm_choice[j] = k
    else:
        r = np.asarray(r).reshape(-1)
        c = np.asarray(c).reshape(-1)
        for i, j in zip(r, c):
            i = int(i); j = int(j)
            if 0 <= i < N and 0 <= j < M:
                pairs.append((i, j))
                k = int(perm_idx[i, j].detach().cpu().item())
                gt_perm_choice[j] = k
    return pairs, gt_perm_choice


def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
                       reduction: str = "mean") -> torch.Tensor:
    """Binary focal loss on logits.
    inputs: [*, 1] or [*] logits
    targets: [*] in {0,1}
    """
    inputs = inputs.view(-1)
    targets = targets.view(-1).float()
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def _principal_dir_torch(xy: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    """Compute principal direction for batched 2D point sets on the current device.
    xy: [K,P,2], mask: [K,P] True=valid (optional). Returns unit vectors [K,2].
    Uses covariance eigenvector; falls back to [1,0] if degenerate.
    """
    K, P, _ = xy.shape
    device = xy.device
    if mask is None:
        valid = torch.ones((K, P), dtype=torch.bool, device=device)
    else:
        valid = mask
    v = valid.float().unsqueeze(-1)  # [K,P,1]
    cnt = v.sum(dim=1).clamp_min(1.0)  # [K,1]
    mean = (xy * v).sum(dim=1, keepdim=True) / cnt.unsqueeze(1)  # [K,1,2]
    xm = (xy - mean) * v  # [K,P,2]
    cov = torch.einsum('kpi,kpj->kij', xm, xm) / cnt.view(K, 1, 1)  # [K,2,2]
    try:
        evals, evecs = torch.linalg.eigh(cov)  # [K,2], [K,2,2]
        idx = evals.argmax(dim=1)
        dirs = torch.stack([evecs[k, :, idx[k]] for k in range(K)], dim=0)  # [K,2]
    except Exception:
        dirs = torch.zeros((K, 2), dtype=xy.dtype, device=device)
        dirs[:, 0] = 1.0
    n = torch.linalg.norm(dirs, dim=1, keepdim=True).clamp_min(eps)
    dirs = dirs / n
    bad = ~torch.isfinite(dirs).all(dim=1)
    if bad.any():
        dirs[bad] = torch.tensor([1.0, 0.0], dtype=xy.dtype, device=device)
    return dirs


def gpu_greedy_match(
    pred_pts: torch.Tensor,   # [N,P,2] on CUDA
    gt_pts: torch.Tensor,     # [N,P,2] packed on CUDA
    gt_mask: torch.Tensor,    # [N,P] True for padding on CUDA
    gt_present: torch.Tensor, # [N]
    w_center: float = 1.0,
    w_dir: float = 0.2,
    w_pw: float = 0.5,
    use_chamfer: bool = True,
    max_center_dist: float | None = None,
) -> List[Tuple[int, int]]:
    """Greedy one-to-one assignment on GPU using torch ops.
    - Computes cost matrix C[N, M] combining center, direction, and simplified Chamfer.
    - Iteratively picks global min, masks row/col, until K=min(N,M).
    - Returns list of (pred_index, gt_index_within_valid) pairs.
    """
    with torch.cuda.amp.autocast(enabled=False):
        assert pred_pts.dim() == 3 and pred_pts.size(-1) == 2
        # force compute in FP32 for numerical stability under AMP
        pred_pts = pred_pts.float()
        gt_pts = gt_pts.float()
        gt_mask = gt_mask.bool()
        gt_present = gt_present.bool()
        device = pred_pts.device
        N, P, _ = pred_pts.shape
        valid_idx = torch.nonzero(gt_present, as_tuple=False).view(-1)
        M = int(valid_idx.numel())
        if N == 0 or M == 0:
            return []
        gt = gt_pts.index_select(0, valid_idx)        # [M,P,2]
        gmask = gt_mask.index_select(0, valid_idx)    # [M,P] True=pad
        diag = float(np.sqrt(8.0))
        # centers
        pc = pred_pts.mean(dim=1)                     # [N,2]
        vmask = (~gmask).float().unsqueeze(-1)        # [M,P,1]
        gc = (gt * vmask).sum(dim=1) / vmask.sum(dim=1).clamp_min(1.0)  # [M,2]
        # directions via principal eigenvector for robustness
        # Guard: if direction weight is 0, skip expensive eigh to avoid VRAM spikes
        if float(w_dir) > 0.0:
            pd = _principal_dir_torch(pred_pts, None)     # [N,2]
            gd = _principal_dir_torch(gt, ~gmask)         # [M,2]
        else:
            pd = None
            gd = None
        # costs
        # pairwise L2 distances in FP32 (avoid cdist AMP type issues)
        diff_c = pc[:, None, :].contiguous() - gc[None, :, :].contiguous()  # [N,M,2]
        d_center = torch.sqrt(torch.clamp((diff_c ** 2).sum(dim=-1), min=0.0))  # [N,M] in normalized coords
        dc = d_center / diag  # [N,M]
        if float(w_dir) > 0.0 and (pd is not None) and (gd is not None):
            cos = torch.clamp(pd @ gd.t(), min=-1.0, max=1.0).abs()  # [N,M]
            dtheta = (1.0 - cos)
            C = float(w_center) * dc + float(w_dir) * dtheta
        else:
            C = float(w_center) * dc
        # Pre-match gating: mask pairs beyond center distance threshold to prevent bad consumption
        if (max_center_dist is not None) and (float(max_center_dist) > 0.0):
            gate_mask = d_center > float(max_center_dist)
            if gate_mask.any():
                C = C.masked_fill(gate_mask, float('inf'))
        if bool(use_chamfer) and (float(w_pw) > 0.0):
            m = int(min(P, 16))
            idx = torch.linspace(0, P - 1, steps=m, device=device).round().long()
            a = pred_pts.index_select(1, idx)              # [N,m,2]
            b = gt.index_select(1, idx)                    # [M,m,2]
            bmask = gmask.index_select(1, idx)             # [M,m]
            # distances [N,M,m,m]
            diff = a[:, None, :, None, :] - b[None, :, None, :, :]
            d = torch.sqrt(torch.clamp((diff ** 2).sum(-1), min=0.0))  # [N,M,m,m]
            inf = torch.tensor(float('inf'), device=device, dtype=d.dtype)
            # pred->gt: mask invalid gt ds points on the gt-sample axis
            mask_gt_invalid = bmask[None, :, None, :]  # [1,M,1,m]
            d_p2g = d.masked_fill(mask_gt_invalid, inf)
            d_pred_to_gt = d_p2g.min(dim=3).values.mean(dim=2)  # [N,M]
            # gt->pred: min over pred samples; mean only over valid gt ds points
            d_g2p = d_p2g.min(dim=2).values  # [N,M,m]
            denom = (~bmask).float().sum(dim=1).clamp_min(1.0)  # [M]
            d_gt_to_pred = (d_g2p.sum(dim=2) / denom[None, :])  # [N,M]
            dch = 0.5 * (d_pred_to_gt + d_gt_to_pred) / diag
            C = C + float(w_pw) * dch
    # greedy selection
    C_work = C.clone()
    pairs: List[Tuple[int, int]] = []
    K = int(min(N, M))
    big = torch.tensor(float('inf'), device=device, dtype=C_work.dtype)
    for _ in range(K):
        val, flat = torch.min(C_work.view(-1), dim=0)
        if not torch.isfinite(val):
            break
        i = int(flat.item() // M)
        j = int(flat.item() % M)
        pairs.append((i, j))
        C_work[i, :] = big
        C_work[:, j] = big
    return pairs


def criterion(
    pred_coords: torch.Tensor,  # [B,N,P,2]
    pred_logits: torch.Tensor,  # [B,N,1]
    tgt_coords: torch.Tensor,   # [B,N,P,2] (packed)
    tgt_mask: torch.Tensor,     # [B,N,P]
    tgt_present: torch.Tensor,  # [B,N]
    l1_weight: float = 1.0,
    cls_weight: float = 1.0,
    use_focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pred_sem_logits: torch.Tensor | None = None,  # [B,N,3]
    tgt_sem_labels: torch.Tensor | None = None,   # [B,N] in {0,1,2} or -1 for ignore
    sem_weight: float = 1.0,
    smooth_weight: float = 0.0,
    reg_len_exp: float = 0.0,
    smooth_inv_len_exp: float = 0.0,
    dir_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    B, N, P, _ = pred_coords.shape
    # Classification loss on all slots (BCE/Focal)
    if use_focal:
        loss_cls = sigmoid_focal_loss(pred_logits.view(B, N), tgt_present, alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
    else:
        loss_cls = F.binary_cross_entropy_with_logits(pred_logits.view(B, N), tgt_present.float(), reduction="mean")

    # Vectorized path (env toggle)
    if os.environ.get('REFINE_VEC_LOSS', '0') == '1':
        present = (tgt_present > 0)                                # [B,N]
        V = (~tgt_mask) & present.unsqueeze(-1)                    # [B,N,P]
        Vf = torch.flip(V, dims=[2])
        # Forward masked mean L1
        diff_fwd = (pred_coords - tgt_coords).abs().sum(dim=-1)    # [B,N,P]
        denom_fwd = V.float().sum(dim=2).clamp_min(1.0)            # [B,N]
        s_fwd = (diff_fwd * V.float()).sum(dim=2) / denom_fwd      # [B,N]
        # Backward (orientation-agnostic)
        tgt_flip = torch.flip(tgt_coords, dims=[2])
        diff_bwd = (pred_coords - tgt_flip).abs().sum(dim=-1)
        denom_bwd = Vf.float().sum(dim=2).clamp_min(1.0)
        s_bwd = (diff_bwd * Vf.float()).sum(dim=2) / denom_bwd
        l_inst = torch.minimum(s_fwd, s_bwd)                       # [B,N]
        # Target-based length per instance for weighting
        edges = tgt_coords[:, :, 1:, :] - tgt_coords[:, :, :-1, :]             # [B,N,P-1,2]
        mask_e = V[:, :, 1:] & V[:, :, :-1]                                     # [B,N,P-1]
        L = torch.sqrt(torch.clamp((edges.pow(2).sum(dim=-1)), min=0.0))        # [B,N,P-1]
        L = (L * mask_e.float()).sum(dim=2)                                     # [B,N]
        # aggregate over present instances
        M = present.float()
        L_mean = (L * M).sum() / M.sum().clamp_min(1.0)
        if float(reg_len_exp) != 0.0:
            w = torch.pow(torch.clamp(L / L_mean, min=1e-6), reg_len_exp)
        else:
            w = torch.ones_like(L)
        w = w * M
        sum_w = w.sum().clamp_min(1e-6)
        loss_reg = (w * l_inst).sum() / sum_w
    else:
        # Original (loop-based) path
        def _inst_length(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] < 2:
                return torch.as_tensor(0.0, device=x.device, dtype=x.dtype)
            d = (x[1:] - x[:-1]).pow(2).sum(-1).sqrt()
            return d.sum()
        loss_reg = pred_coords.new_zeros([])
        sum_w_reg = pred_coords.new_zeros([])
        for b in range(B):
            idx = (tgt_present[b] > 0)
            if not torch.any(idx):
                continue
            pc = pred_coords[b, idx]
            tc = tgt_coords[b, idx]
            tm = tgt_mask[b, idx]
            lengths = []
            for i in range(tc.shape[0]):
                valid = ~tm[i]
                if not torch.any(valid):
                    lengths.append(torch.as_tensor(0.0, device=tc.device, dtype=tc.dtype))
                else:
                    lengths.append(_inst_length(tc[i, valid]))
            if lengths:
                L = torch.stack(lengths)
                L_mean = torch.clamp(L.mean(), min=1e-6)
            else:
                L = None
                L_mean = torch.as_tensor(1.0, device=tc.device, dtype=tc.dtype)
            for i in range(pc.shape[0]):
                valid = ~tm[i]
                if not torch.any(valid):
                    continue
                l_inst = bidirectional_l1(pc[i, valid], tc[i, valid])
                if float(reg_len_exp) != 0.0:
                    w = torch.pow(torch.clamp(L[i] / L_mean, min=1e-6), reg_len_exp)
                else:
                    w = torch.as_tensor(1.0, device=pc.device, dtype=pc.dtype)
                loss_reg = loss_reg + w * l_inst
                sum_w_reg = sum_w_reg + w
        if float(sum_w_reg.item() if sum_w_reg.numel() > 0 else 0.0) > 0.0:
            loss_reg = loss_reg / sum_w_reg
        else:
            loss_reg = pred_coords.new_zeros([])
    # Semantic class loss (only on matched pairs; unmatched label should be -1)
    loss_sem = pred_coords.new_zeros([])
    if (pred_sem_logits is not None) and (tgt_sem_labels is not None):
        B2, N2, C = pred_sem_logits.shape
        assert C == 3 and B2 == B and N2 == N
        # 仅对“存在”的槽位做类别监督，避免对填充/空槽施加错误类别约束
        mask = (tgt_sem_labels >= 0) & (tgt_present > 0)
        if torch.any(mask):
            ps = pred_sem_logits[mask]
            ts = tgt_sem_labels[mask].long()
            loss_sem = F.cross_entropy(ps, ts, reduction="mean")
        else:
            loss_sem = pred_coords.new_zeros([])

    # Smoothness regularization on present slots only (second difference / curvature proxy)
    loss_smooth = pred_coords.new_zeros([])
    if float(smooth_weight) > 0.0:
        if os.environ.get('REFINE_VEC_LOSS', '0') == '1':
            present = (tgt_present > 0)
            V = (~tgt_mask) & present.unsqueeze(-1)
            if P >= 3:
                v1 = pred_coords[:, :, 1:-1, :] - pred_coords[:, :, :-2, :]
                v2 = pred_coords[:, :, 2:, :] - pred_coords[:, :, 1:-1, :]
                a = v2 - v1
                m_triple = V[:, :, :-2] & V[:, :, 1:-1] & V[:, :, 2:]
                a2 = a.pow(2).sum(dim=-1)
                e = (a2 * m_triple.float()).sum(dim=2) / m_triple.float().sum(dim=2).clamp_min(1.0)  # [B,N]
                # length from predicted coords
                edge = pred_coords[:, :, 1:, :] - pred_coords[:, :, :-1, :]
                m_edge = V[:, :, 1:] & V[:, :, :-1]
                Lp = torch.sqrt(torch.clamp((edge.pow(2).sum(dim=-1)), min=0.0))
                Lp = (Lp * m_edge.float()).sum(dim=2)
                M = present.float()
                L_mean = (Lp * M).sum() / M.sum().clamp_min(1.0)
                if float(smooth_inv_len_exp) != 0.0:
                    w = torch.pow(torch.clamp(L_mean / Lp.clamp_min(1e-6), min=1e-6), smooth_inv_len_exp)
                else:
                    w = torch.ones_like(Lp)
                w = w * M
                loss_smooth = (w * e).sum() / w.sum().clamp_min(1e-6)
            else:
                loss_smooth = pred_coords.new_zeros([])
        else:
            acc = pred_coords.new_zeros([])
            sum_w_sm = pred_coords.new_zeros([])
            for b in range(B):
                idx = (tgt_present[b] > 0)
                if not torch.any(idx):
                    continue
                pc = pred_coords[b, idx]
                tm = tgt_mask[b, idx]
                if pc.shape[0] == 0:
                    continue
                valid = ~tm
                if pc.shape[1] < 3:
                    continue
                v1 = pc[:, 1:-1, :] - pc[:, :-2, :]
                v2 = pc[:, 2:, :] - pc[:, 1:-1, :]
                a = v2 - v1
                m_triple = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
                if torch.any(m_triple):
                    a2 = a.pow(2).sum(-1)
                    Np = pc.shape[0]
                    lengths = []
                    for i in range(Np):
                        vmask = valid[i]
                        xi = pc[i][vmask]
                        if xi.shape[0] < 2:
                            lengths.append(torch.as_tensor(0.0, device=pc.device, dtype=pc.dtype))
                        else:
                            d = (xi[1:] - xi[:-1]).pow(2).sum(-1).sqrt()
                            lengths.append(d.sum())
                    if lengths:
                        L = torch.stack(lengths)
                        L_mean = torch.clamp(L.mean(), min=1e-6)
                    else:
                        L = None
                        L_mean = torch.as_tensor(1.0, device=pc.device, dtype=pc.dtype)
                    for i in range(Np):
                        mask_i = m_triple[i]
                        if not torch.any(mask_i):
                            continue
                        e_i = a2[i][mask_i].mean()
                        if float(smooth_inv_len_exp) != 0.0:
                            w = torch.pow(torch.clamp(L_mean / torch.clamp(L[i], min=1e-6), min=1e-6), smooth_inv_len_exp)
                        else:
                            w = torch.as_tensor(1.0, device=pc.device, dtype=pc.dtype)
                        acc = acc + w * e_i
                        sum_w_sm = sum_w_sm + w
            if float(sum_w_sm.item() if sum_w_sm.numel() > 0 else 0.0) > 0.0:
                loss_smooth = acc / sum_w_sm
            else:
                loss_smooth = pred_coords.new_zeros([])
    
    # Directional consistency loss: penalize reversed first-segment direction
    loss_dir = pred_coords.new_zeros([])
    if float(dir_weight) > 0.0:
        acc = pred_coords.new_zeros([])
        cnt = 0
        eps = 1e-8
        for b in range(B):
            idx = (tgt_present[b] > 0)
            if not torch.any(idx):
                continue
            pc = pred_coords[b, idx]  # [Np,P,2]
            tc = tgt_coords[b, idx]   # [Np,P,2]
            tm = tgt_mask[b, idx]     # [Np,P]
            Np = pc.shape[0]
            for i in range(Np):
                valid = ~tm[i]  # [P]
                if valid.long().sum() < 2:
                    continue
                # first two valid indices
                ids = torch.nonzero(valid, as_tuple=False).view(-1)
                k0 = int(ids[0].item()); k1 = int(ids[1].item())
                v_pred = pc[i, k1] - pc[i, k0]
                v_tgt  = tc[i, k1] - tc[i, k0]
                n_pred = torch.clamp(torch.linalg.norm(v_pred), min=eps)
                n_tgt  = torch.clamp(torch.linalg.norm(v_tgt),  min=eps)
                cos = (v_pred * v_tgt).sum() / (n_pred * n_tgt)
                # 1 - cos ∈ [0,2]，反向时≈2；同向趋近0
                acc = acc + (1.0 - cos)
                cnt += 1
        if cnt > 0:
            loss_dir = acc / float(cnt)
        else:
            loss_dir = pred_coords.new_zeros([])

    # Package losses once; callers can sum/select as needed
    out = {
        "loss_cls": cls_weight * loss_cls,
        "loss_reg": l1_weight * loss_reg,
        "loss_sem": sem_weight * loss_sem,
        "loss_smooth": float(smooth_weight) * loss_smooth,
        "loss_dir": float(dir_weight) * loss_dir,
    }
    return out

    
