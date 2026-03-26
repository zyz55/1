from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from src.models.cfd_mamba_unet3d import CFDMambaUNet3D
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--input_h5", type=str, default="dataset/val.h5")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/inference")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    model = CFDMambaUNet3D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
        depth=model_cfg["depth"],
        mamba_dim=model_cfg["mamba_dim"],
        mamba_layers=model_cfg["mamba_layers"],
        mamba_state_dim=model_cfg["mamba_state_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    with h5py.File(args.input_h5, "r") as f:
        x = f["x"][args.sample_idx]
        y = f["y"][args.sample_idx]
        mask = f["mask"][args.sample_idx]

    x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(x_t, pred_steps=y.shape[0]).squeeze(0).cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "pred.npy", pred)
    np.save(out_dir / "target.npy", y)
    np.save(out_dir / "mask.npy", mask)

    mse = float(np.mean((pred - y) ** 2))
    with (out_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"sample_idx={args.sample_idx}\n")
        f.write(f"mse={mse:.8e}\n")

    print(f"Inference done. mse={mse:.6e}, saved to {out_dir}")


if __name__ == "__main__":
    main()
