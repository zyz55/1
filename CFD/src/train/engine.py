from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.physics.losses import PhysicsLoss


def lambda_schedule(epoch: int, warmup_epochs: int, lambda_final: float) -> float:
    if epoch < warmup_epochs:
        return 0.0
    frac = (epoch - warmup_epochs + 1) / max(1, warmup_epochs)
    return min(lambda_final, lambda_final * frac)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    data_criterion: nn.Module,
    physics_criterion: PhysicsLoss,
    device: torch.device,
    lambda_phy: float,
    scaler: GradScaler,
    train: bool,
    amp: bool,
    grad_clip_norm: float | None,
) -> Dict[str, float]:
    model.train(train)
    loss_sum = 0.0
    data_sum = 0.0
    phy_sum = 0.0

    for batch in tqdm(loader, desc="train" if train else "val", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u_future = batch["u_future"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(train):
            with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x, pred_steps=y.shape[1])
                l_data = data_criterion(pred, y)
                l_phy = physics_criterion(pred, u_future, mask)
                loss = l_data + lambda_phy * l_phy

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        loss_sum += float(loss.detach().cpu())
        data_sum += float(l_data.detach().cpu())
        phy_sum += float(l_phy.detach().cpu())

    n = max(1, len(loader))
    return {"loss": loss_sum / n, "loss_data": data_sum / n, "loss_phy": phy_sum / n}


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: Dict, device: torch.device) -> None:
    train_cfg = cfg["train"]
    phy_cfg = cfg["physics"]

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    data_criterion = nn.MSELoss()
    physics_criterion = PhysicsLoss(
        dt=phy_cfg["dt"],
        dx=phy_cfg["dx"],
        dy=phy_cfg["dy"],
        dz=phy_cfg["dz"],
        d_eff=phy_cfg["d_eff"],
    )
    scaler = GradScaler(enabled=train_cfg.get("amp", True) and device.type == "cuda")

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for epoch in range(train_cfg["epochs"]):
        lam = lambda_schedule(epoch, train_cfg["warmup_epochs"], train_cfg["lambda_physics_final"])

        tr = run_epoch(
            model,
            train_loader,
            optimizer,
            data_criterion,
            physics_criterion,
            device,
            lam,
            scaler,
            train=True,
            amp=train_cfg.get("amp", True),
            grad_clip_norm=train_cfg.get("grad_clip_norm", None),
        )
        va = run_epoch(
            model,
            val_loader,
            optimizer,
            data_criterion,
            physics_criterion,
            device,
            lam,
            scaler,
            train=False,
            amp=train_cfg.get("amp", True),
            grad_clip_norm=None,
        )

        scheduler.step(va["loss"])

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train": tr,
            "val": va,
            "lambda_phy": lam,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if va["loss"] < best:
            best = va["loss"]
            torch.save(state, ckpt_dir / "best.pt")

        print(f"Epoch {epoch+1}/{train_cfg['epochs']} | train={tr['loss']:.4e} val={va['loss']:.4e} lambda_phy={lam:.4f}")
