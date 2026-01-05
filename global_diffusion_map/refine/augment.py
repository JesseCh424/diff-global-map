from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


@torch.no_grad()
def augment_planA_from_gt_torch(
    prop: torch.Tensor,      # [B,N,P,2] normalized [-1,1]
    gt_mask: torch.Tensor,   # [B,N,P] True for padding
    jitter_sigma: float = 0.0,
    drop_lo: float = 0.3,
    drop_hi: float = 0.5,
    ghosts_lo: int = 0,
    ghosts_hi: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MapTR/Refine 训练使用的 Plan-A 增强（与训练保持一致）。

    - 对存在的曲线按 jitter_sigma 添加点级噪声（默认 0，禁用）
    - 随机 drop 一部分存在的实例（比例 [drop_lo, drop_hi]）
    - 在空槽注入 ghosts（数量均匀抽样 [ghosts_lo, ghosts_hi]），用连续随机游走生成

    返回: (prop_aug, present) 其中 present 为 [B,N] bool，表示 proposal 中“存在”的槽位
    """
    assert prop.dim() == 4 and prop.size(-1) == 2
    B, N, P, _ = prop.shape
    device = prop.device
    prop = prop.clone()
    present = (~gt_mask).logical_not().all(dim=-1).logical_not().clone()  # [B,N]

    # jitter（仅对有效点），默认关闭
    if float(jitter_sigma) > 0.0:
        mask_points = (~gt_mask).to(prop.dtype).unsqueeze(-1)  # [B,N,P,1]
        noise = torch.randn_like(prop) * float(jitter_sigma)
        prop = torch.clamp(prop + noise * mask_points, -1.0, 1.0)

    # 逐 batch 执行 drop / ghost 以获得随机性
    for b in range(B):
        # drop：从存在的槽随机丢弃一部分（用噪声占位，并标记 present=False，避免可视化时绘制）
        ids = torch.nonzero(present[b], as_tuple=False).view(-1)
        if ids.numel() > 0 and (float(drop_hi) > 0.0):
            frac = float(torch.empty((), device=device).uniform_(float(drop_lo), float(drop_hi)).item())
            drop_num = int(round(ids.numel() * max(0.0, frac)))
            drop_num = max(0, min(int(ids.numel()), drop_num))
            if drop_num > 0:
                sel = ids[torch.randperm(ids.numel(), device=device)[:drop_num]]
                rn = torch.randn((sel.numel(), P, 2), device=device)
                prop[b, sel] = rn.clamp(-1.0, 1.0)
                present[b, sel] = False

        # ghost：在空槽注入少量连续随机游走曲线，并标记 present=True
        empty = torch.nonzero(~present[b], as_tuple=False).view(-1)
        if empty.numel() > 0 and (int(ghosts_hi) > 0):
            take = int(torch.randint(low=int(max(0, ghosts_lo)), high=int(max(ghosts_hi, ghosts_lo + 1)), size=(1,), device=device).item())
            take = max(0, min(int(empty.numel()), int(take)))
            if take > 0:
                idx = empty[torch.randperm(empty.numel(), device=device)[:take]]
                g = torch.rand((take, P, 2), device=device) * 2.0 - 1.0
                for t in range(1, P):
                    g[:, t] = 0.7 * g[:, t] + 0.3 * g[:, t - 1]
                prop[b, idx] = g
                present[b, idx] = True

    return prop, present


def augment_planA_from_gt_numpy(
    prop: np.ndarray,      # [N,P,2]
    gt_mask: np.ndarray,   # [N,P] True for padding
    jitter_sigma: float = 0.0,
    drop_lo: float = 0.3,
    drop_hi: float = 0.5,
    ghosts_lo: int = 0,
    ghosts_hi: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy 包装，便于可视化调用，与 augment_planA_from_gt_torch 行为一致。"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.from_numpy(prop[None, ...]).to(device)
    m = torch.from_numpy(gt_mask[None, ...]).to(device)
    xa, pres = augment_planA_from_gt_torch(x, m, jitter_sigma, drop_lo, drop_hi, ghosts_lo, ghosts_hi)
    return xa[0].detach().cpu().numpy(), pres[0].detach().cpu().numpy().astype(bool)

