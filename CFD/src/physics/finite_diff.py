from __future__ import annotations

import torch
import torch.nn.functional as F


def _kernel_1d(values, axis: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.tensor(values, device=device, dtype=dtype)
    shape = [1, 1, 1, 1, 1]
    shape[2 + axis] = len(values)
    return k.view(*shape)


def central_grad3d(x: torch.Tensor, spacing: tuple[float, float, float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx, dy, dz = spacing
    xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")

    kz = _kernel_1d([-0.5 / dz, 0.0, 0.5 / dz], axis=0, device=x.device, dtype=x.dtype)
    ky = _kernel_1d([-0.5 / dy, 0.0, 0.5 / dy], axis=1, device=x.device, dtype=x.dtype)
    kx = _kernel_1d([-0.5 / dx, 0.0, 0.5 / dx], axis=2, device=x.device, dtype=x.dtype)

    gz = F.conv3d(xp, kz)
    gy = F.conv3d(xp, ky)
    gx = F.conv3d(xp, kx)
    return gx, gy, gz


def laplacian3d(x: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
    dx, dy, dz = spacing
    xp = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")

    kz = _kernel_1d([1.0 / (dz * dz), -2.0 / (dz * dz), 1.0 / (dz * dz)], axis=0, device=x.device, dtype=x.dtype)
    ky = _kernel_1d([1.0 / (dy * dy), -2.0 / (dy * dy), 1.0 / (dy * dy)], axis=1, device=x.device, dtype=x.dtype)
    kx = _kernel_1d([1.0 / (dx * dx), -2.0 / (dx * dx), 1.0 / (dx * dx)], axis=2, device=x.device, dtype=x.dtype)

    return F.conv3d(xp, kx) + F.conv3d(xp, ky) + F.conv3d(xp, kz)


def divergence_flux(uxc: torch.Tensor, uyc: torch.Tensor, uzc: torch.Tensor, spacing: tuple[float, float, float]) -> torch.Tensor:
    dfx_dx, _, _ = central_grad3d(uxc, spacing)
    _, dfy_dy, _ = central_grad3d(uyc, spacing)
    _, _, dfz_dz = central_grad3d(uzc, spacing)
    return dfx_dx + dfy_dy + dfz_dz
