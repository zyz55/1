from __future__ import annotations

import torch
import torch.nn as nn

from src.physics.finite_diff import central_grad3d, divergence_flux, laplacian3d


class PhysicsLoss(nn.Module):
    def __init__(self, dt: float, dx: float, dy: float, dz: float, d_eff: float):
        super().__init__()
        self.dt = dt
        self.spacing = (dx, dy, dz)
        self.d_eff = d_eff

    def forward(self, c_pred: torch.Tensor, u_future: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # c_pred: (B,T,1,D,H,W), u_future: (B,T,3,D,H,W), mask: (B,1,1,D,H,W) or (B,1,D,H,W)
        if mask.dim() == 5:
            mask = mask.unsqueeze(1)
        m = mask

        losses = []
        for t in range(1, c_pred.shape[1]):
            c_t = c_pred[:, t : t + 1, ...].squeeze(1)
            c_tm1 = c_pred[:, t - 1 : t, ...].squeeze(1)
            u_t = u_future[:, t, ...]

            dcdt = (c_t - c_tm1) / self.dt

            ux = u_t[:, 0:1]
            uy = u_t[:, 1:2]
            uz = u_t[:, 2:3]

            flux_div = divergence_flux(ux * c_t, uy * c_t, uz * c_t, self.spacing)
            lap = laplacian3d(c_t, self.spacing)

            residual = dcdt + flux_div - self.d_eff * lap
            residual = residual * m[:, 0]
            losses.append((residual ** 2).mean())

        if not losses:
            return torch.tensor(0.0, device=c_pred.device)
        return torch.stack(losses).mean()
