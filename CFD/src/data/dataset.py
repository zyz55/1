from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class GasDispersionH5Dataset(Dataset):
    def __init__(self, h5_path: str | Path):
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["x"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            x = torch.from_numpy(f["x"][idx]).float()
            y = torch.from_numpy(f["y"][idx]).float()
            u_future = torch.from_numpy(f["u_future"][idx]).float()
            mask = torch.from_numpy(f["mask"][idx]).float()
        return {"x": x, "y": y, "u_future": u_future, "mask": mask}
