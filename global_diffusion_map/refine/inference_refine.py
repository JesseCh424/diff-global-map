from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


@torch.no_grad()
def sdedit_refine(
    denoiser: torch.nn.Module,
    x_init: torch.Tensor,      # [B,N,P,2] normalized
    raster_feat: Optional[torch.Tensor],
    steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 1.5,
    rho: float = 7.0,
    second_order: bool = True,
    guide_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """SDEdit-like refinement from intermediate noise level (proposal as init).

    Follows an EDM/Karras schedule and keeps raster features cached between steps
    (cache policy left to the caller/denoiser if supported).
    """
    device = x_init.device
    net = denoiser

    # Karras schedule
    step_idx = torch.arange(steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1.0 / rho) + step_idx / max(steps - 1, 1) *
               (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(step_idx[:1])])
    t_steps[-1] = t_steps[-2] * 0.5

    x_next = x_init.to(torch.float64)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        kwargs = dict(guide_kwargs or {})
        kwargs["cache_image_feat"] = (i == 0)
        kwargs["use_cached_feat"] = (i > 0)
        den = net(x_next, t_cur, raster_feat, **kwargs).to(torch.float64)
        den = den[-1]
        d_cur = (x_next - den) / t_cur
        x_next = x_next + (t_next - t_cur) * d_cur
        if second_order and i < steps - 1:
            kwargs["cache_image_feat"] = False
            kwargs["use_cached_feat"] = True
            den2 = net(x_next, t_next, raster_feat, **kwargs).to(torch.float64)
            den2 = den2[-1]
            d_prime = (x_next - den2) / t_next
            x_next = x_next + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next.to(torch.float32)

