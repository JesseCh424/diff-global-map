from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DiffusionRefiner(nn.Module):
    """Minimal denoising head with dual outputs (coords + presence).

    This is a light-weight skeleton suitable to be integrated with an existing
    EDM preconditioner. It expects per-slot features and outputs:
      - coords: [B, N, P, 2]
      - logits: [B, N, 1] (presence score for deletion/generation)
    """

    def __init__(self, hidden_dim: int, num_points: int, sem_classes: int = 0) -> None:
        super().__init__()
        self.num_points = int(num_points)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_points * 2),
        )
        # Upgraded MLP classification head for stronger presence discrimination
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Optional semantic head (e.g., 3-way: divider/ped/boundary). When sem_classes<=0, disabled.
        self.sem_head: nn.Module | None = None
        if int(sem_classes) > 0:
            self.sem_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, int(sem_classes)),
            )

    def forward(self, slot_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
          slot_feats: [B, N, C] pooled features per slot.

        Returns:
          coords: [B, N, P, 2]
          logits: [B, N, 1]
        """
        B, N, C = slot_feats.shape
        reg = self.reg_head(slot_feats).view(B, N, self.num_points, 2)
        logits = self.cls_head(slot_feats)
        if self.sem_head is not None:
            sem_logits = self.sem_head(slot_feats)
            return reg, logits, sem_logits
        return reg, logits


class SinusoidalTimeEmbedding(nn.Module):
    """Simple sinusoidal embedding for a scalar time/noise level.

    Produces a fixed-size vector that can be concatenated to slot features.
    """

    def __init__(self, dim: int = 64, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] scalar in continuous range (e.g., sigma or normalized time)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.linspace(0, 1, steps=half, device=device) * torch.log(torch.tensor(self.max_period, device=device))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=-1)
        return emb  # [B,dim]


class SlotMLPWithTime(nn.Module):
    """Slot MLP + time embedding → dual-head outputs.

    Minimal structure conditioned on a scalar noise level.
    """

    def __init__(self, P: int, hidden: int = 256, out_points: int = 30, t_dim: int = 64, num_slots: int | None = None, sem_classes: int = 0) -> None:
        super().__init__()
        self.P = int(P)
        self.num_slots = int(num_slots) if num_slots is not None else None
        self.t_emb = SinusoidalTimeEmbedding(dim=t_dim)
        in_dim = P * 2 + 256 + t_dim
        self.feat = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = DiffusionRefiner(hidden_dim=hidden, num_points=out_points, sem_classes=int(sem_classes))
        # Learnable per-slot IDs to avoid mode collapse in dynamic creation
        if self.num_slots is not None:
            self.query_embed = nn.Embedding(self.num_slots, hidden)
        # Optional class embedding for input prior labels (e.g., divider/ped/boundary)
        self.class_emb: nn.Embedding | None = None
        if int(sem_classes) > 0:
            self.class_emb = nn.Embedding(int(sem_classes), hidden)
        # Optional prior encoder to inject proposal shape prior
        self.prior_mlp = nn.Sequential(
            nn.Linear(P * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        # Gates (start from 0.0 to avoid perturbing old checkpoints before finetune)
        # 将门控初始化为一个微小正数，使先验在训练早期即可参与（避免“完全切断”导致学习滞后）
        self.prior_gate = nn.Parameter(torch.tensor(1.0e-3))
        self.class_gate = nn.Parameter(torch.tensor(1.0e-3))

        
    def forward(self, x_slots: torch.Tensor, raster_vec: torch.Tensor, t_scalar: torch.Tensor,
                x_prior: torch.Tensor | None = None,
                input_labels: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_slots: [B,N,P,2], raster_vec: [B,256], t_scalar: [B]
        B, N, P, _ = x_slots.shape
        xs = x_slots.view(B, N, P * 2)
        rv = raster_vec[:, None, :].expand(B, N, -1)
        te = self.t_emb(t_scalar).unsqueeze(1).expand(B, N, -1)
        feats = torch.cat([xs, rv, te], dim=-1)
        feats = self.feat(feats)
        # add per-slot query embedding if available
        if hasattr(self, 'query_embed'):
            ids = torch.arange(N, device=feats.device)
            if self.num_slots is not None and N > self.num_slots:
                ids = ids % self.num_slots
            qe = self.query_embed(ids)[None, :, :].expand(B, N, -1)
            feats = feats + qe
        # fuse prior shape features if provided
        if x_prior is not None:
            xp = x_prior.view(B, N, P * 2)
            pf = self.prior_mlp(xp)
            feats = feats + self.prior_gate.tanh() * pf
        # fuse class prior if provided
        if (input_labels is not None) and (self.class_emb is not None):
            # input_labels: [B,N]
            # When label == -1 (unknown), do NOT inject any class prior.
            # Clamp for embedding lookup, but zero-out with a valid mask.
            valid_mask = (input_labels >= 0).unsqueeze(-1).to(feats.dtype)  # [B,N,1]
            cf = self.class_emb(input_labels.clamp(min=0))
            feats = feats + self.class_gate.tanh() * cf * valid_mask
        out = self.head(feats)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            coords, logits, sem_logits = out
            coords = torch.tanh(coords)
            return coords, logits, sem_logits
        else:
            coords, logits = out  # type: ignore[assignment]
            coords = torch.tanh(coords)
            return coords, logits
