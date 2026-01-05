from __future__ import annotations

"""EDM-style preconditioning and sampler utilities for refine.

This module mirrors the essential bits of official PolyDiffuse EDM:
- Preconditioning with c_in and optional mu_guide blending
- Karras (EDM) noise schedule and Heun/Euler solver

It adapts to our light SlotMLPWithTime backbone expecting:
  net(x_slots: [B,N,P,2], raster_vec: [B,C], t_scalar: [B]) -> coords, logits, [sem]
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class EDMConfig:
    sigma_data: float = 1.0
    second_order: bool = True
    rho: float = 7.0
    S_churn: float = 0.0
    S_min: float = 0.0
    S_max: float = float("inf")
    S_noise: float = 1.0


class EDMPrecondRefine(torch.nn.Module):
    """EDM preconditioning wrapper for refine backbone.

    It computes c_in from sigma and blends x with an optional mu_guide
    (here set to zeros by default to keep the contract minimal), then
    calls the underlying backbone with time embedding.
    """

    def __init__(self, backbone: torch.nn.Module, sigma_data: float = 1.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.sigma_data = float(sigma_data)

    def forward(self, x_slots: torch.Tensor, sigma: torch.Tensor, raster_vec: torch.Tensor, **kwargs):
        # x_slots: [B,N,P,2]; sigma: [B] or [B,1]; raster_vec: [B,C]
        sigma = sigma.view(-1)
        # preconditioning
        c_in = 1.0 / torch.sqrt(torch.as_tensor(self.sigma_data, device=sigma.device, dtype=torch.float32) ** 2 + sigma**2)
        c_in = c_in.to(x_slots.dtype)
        # mu_guide: keep zero for minimal integration
        x_in = x_slots * c_in.view(-1, 1, 1, 1)
        # time embedding: follow EDM practice to use log(sigma) as noise input
        t_scalar = torch.log(sigma.clamp_min(1e-8))
        out = self.backbone(x_in, raster_vec, t_scalar, **kwargs)
        return out  # coords, logits, [sem]

    def round_sigma(self, sigma: torch.Tensor | float) -> torch.Tensor:
        return torch.as_tensor(sigma, dtype=torch.float32)


def karras_schedule(num_steps: int, sigma_min: float, sigma_max: float, rho: float = 7.0) -> torch.Tensor:
    """Karras et al. EDM noise schedule (without the final 0).
    Returns: [num_steps] torch tensor descending from sigma_max to sigma_min.
    """
    device = torch.device('cpu')
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    s = (sigma_max ** (1.0 / rho) + step_indices / max(num_steps - 1, 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    return s.to(torch.float32)


def edm_unrolled_train(
    net: EDMPrecondRefine,
    x_start: torch.Tensor,            # [B,N,P,2] sdedit start (xK)
    raster_vec: torch.Tensor,         # [B,C]
    sigmas: List[float] | torch.Tensor,
    second_order: bool = True,
    cond_prior: torch.Tensor | None = None,
    input_labels: torch.Tensor | None = None,
    collect_states: bool = True,
    collect_preds: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]], List[torch.Tensor]]:
    """Run unrolled EDM steps for training; returns last pred and intermediate preds.

    The update follows Heun/Euler; gradients only flow through the model,
    not through the state update (detach derivative), mirroring prior code.
    """
    if not torch.is_tensor(sigmas):
        sigmas = torch.tensor(list(sigmas), dtype=torch.float32, device=x_start.device)
    B = x_start.shape[0]
    x_t = x_start
    preds: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    last_pred: Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None
    states: List[torch.Tensor] = ([x_t.clone()] if collect_states else [])

    for si, sigma in enumerate(sigmas):
        sigma_b = torch.full((B,), float(sigma), device=x_start.device, dtype=torch.float32)
        out = net(x_t, sigma_b, raster_vec, x_prior=cond_prior, input_labels=input_labels)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            pred_coords, pred_logits, pred_sem = out
        else:
            pred_coords, pred_logits = out
            pred_sem = None
        last_pred = (pred_coords, pred_logits, pred_sem)
        if collect_preds:
            preds.append(last_pred)

        # ODE update: x' = (x - x0)/sigma
        if si < len(sigmas) - 1:
            sigma_next = float(sigmas[si + 1])
            t_hat = float(sigma)
            # Euler step
            d_cur = (x_t - pred_coords).detach() / max(t_hat, 1e-6)
            x_next = x_t + (sigma_next - t_hat) * d_cur
            if second_order:
                # Heun correction — out2 is only used to update the state; it does not
                # contribute to the loss graph. Avoid building gradients to reduce memory.
                sigma_b2 = torch.full((B,), sigma_next, device=x_start.device, dtype=torch.float32)
                with torch.no_grad():
                    out2 = net(x_next, sigma_b2, raster_vec, x_prior=cond_prior, input_labels=input_labels)
                    pc2 = out2[0] if isinstance(out2, (tuple, list)) else out2
                d_prime = (x_next - pc2).detach() / max(sigma_next, 1e-6)
                x_next = x_t + (sigma_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            x_t = x_next.clamp(-1.0, 1.0)
            if collect_states:
                states.append(x_t.clone())

    if not collect_preds and last_pred is not None:
        preds = [last_pred]
    return pred_coords, pred_logits, preds, states
