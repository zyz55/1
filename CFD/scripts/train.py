from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.data.dataset import GasDispersionH5Dataset
from src.models.cfd_mamba_unet3d import CFDMambaUNet3D
from src.train.engine import train_model
from src.utils.config import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    train_ds = GasDispersionH5Dataset(train_cfg["train_data_h5"])
    val_ds = GasDispersionH5Dataset(train_cfg["val_data_h5"])

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

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

    train_model(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    main()
