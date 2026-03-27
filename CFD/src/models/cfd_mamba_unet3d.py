from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from src.models.blocks import ConvBlock3D, TimeDistributed

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba is None:
            self.use_fallback = True
            self.fallback = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.use_fallback = False
            self.norm = nn.LayerNorm(d_model)
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fallback:
            return x + self.fallback(x)
        return x + self.mamba(self.norm(x))


class CFDMambaUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        mamba_dim: int = 256,
        mamba_layers: int = 4,
        mamba_state_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        enc_channels: List[int] = []

        ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(TimeDistributed(ConvBlock3D(ch, out_ch, dropout=dropout)))
            self.pools.append(TimeDistributed(nn.MaxPool3d(2)))
            enc_channels.append(out_ch)
            ch = out_ch

        self.bottleneck = TimeDistributed(ConvBlock3D(ch, ch * 2, dropout=dropout))
        bottleneck_ch = ch * 2

        self.to_mamba = nn.Linear(bottleneck_ch, mamba_dim)
        self.mamba_blocks = nn.ModuleList([MambaBlock(mamba_dim, d_state=mamba_state_dim) for _ in range(mamba_layers)])
        self.from_mamba = nn.Linear(mamba_dim, bottleneck_ch)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_ch = bottleneck_ch
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            self.upconvs.append(TimeDistributed(nn.ConvTranspose3d(dec_ch, skip_ch, kernel_size=2, stride=2)))
            self.decoders.append(TimeDistributed(ConvBlock3D(skip_ch * 2, skip_ch, dropout=dropout)))
            dec_ch = skip_ch

        self.head = TimeDistributed(nn.Conv3d(base_channels, out_channels, kernel_size=1))

    def _temporal_mamba(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, d, h, w = x.shape
        seq = rearrange(x, "b t c d h w -> (b d h w) t c")
        seq = self.to_mamba(seq)
        for blk in self.mamba_blocks:
            seq = blk(seq)
        seq = self.from_mamba(seq)
        return rearrange(seq, "(b d h w) t c -> b t c d h w", b=b, d=d, h=h, w=w)

    def forward(self, x: torch.Tensor, pred_steps: int | None = None) -> torch.Tensor:
        skips = []
        out = x
        for enc, pool in zip(self.encoders, self.pools):
            out = enc(out)
            skips.append(out)
            out = pool(out)

        out = self.bottleneck(out)
        out = self._temporal_mamba(out)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            out = up(out)
            if out.shape[-3:] != skip.shape[-3:]:
                md = min(out.shape[-3], skip.shape[-3])
                mh = min(out.shape[-2], skip.shape[-2])
                mw = min(out.shape[-1], skip.shape[-1])
                out = out[..., :md, :mh, :mw]
                skip = skip[..., :md, :mh, :mw]
            out = torch.cat([out, skip], dim=2)
            out = dec(out)

        out = self.head(out)
        if pred_steps is not None:
            if out.shape[1] >= pred_steps:
                out = out[:, -pred_steps:]
            else:
                extra = pred_steps - out.shape[1]
                out = torch.cat([out, out[:, -1:].repeat(1, extra, 1, 1, 1, 1)], dim=1)
        return out
