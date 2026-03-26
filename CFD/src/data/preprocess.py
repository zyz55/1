from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np


@dataclass
class NormStats:
    c_max: float
    lambda_log: float
    u_mean: float
    u_std: float
    p_mean: float
    p_std: float


def log_normalize_concentration(c: np.ndarray, c_max: float, lambda_log: float, eps: float = 1e-12) -> np.ndarray:
    return np.log(lambda_log * c + 1.0) / (np.log(lambda_log * c_max + 1.0) + eps)


def standardize(x: np.ndarray, mean: float, std: float, eps: float = 1e-12) -> np.ndarray:
    return (x - mean) / (std + eps)


def compute_stats(case_dirs: list[Path], lambda_log: float) -> NormStats:
    c_max, u_vals, p_vals = 0.0, [], []
    for d in case_dirs:
        c = np.load(d / "C.npy")
        u = np.load(d / "U.npy")
        p = np.load(d / "p.npy")
        c_max = max(c_max, float(c.max()))
        u_vals.append(u.reshape(-1))
        p_vals.append(p.reshape(-1))
    u_all = np.concatenate(u_vals)
    p_all = np.concatenate(p_vals)
    return NormStats(c_max=c_max, lambda_log=lambda_log, u_mean=float(u_all.mean()), u_std=float(u_all.std()), p_mean=float(p_all.mean()), p_std=float(p_all.std()))


def build_h5_dataset(
    case_dirs: list[Path],
    output_h5: Path,
    stats: NormStats,
    tin: int = 4,
    tout: int = 4,
    stride: int = 1,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    X_list, Y_list, U_future_list, M_list = [], [], [], []
    for d in case_dirs:
        C = np.load(d / "C.npy").astype(np.float32)
        U = np.load(d / "U.npy").astype(np.float32)
        P = np.load(d / "p.npy").astype(np.float32)
        M = np.load(d / "mask.npy").astype(np.float32)

        Cn = log_normalize_concentration(C, c_max=stats.c_max, lambda_log=stats.lambda_log)
        Un = standardize(U, stats.u_mean, stats.u_std)
        Pn = standardize(P, stats.p_mean, stats.p_std)

        T = C.shape[0]
        for t0 in range(0, T - (tin + tout) + 1, stride):
            past_c = Cn[t0:t0 + tin]
            past_u = Un[t0:t0 + tin]
            past_p = Pn[t0:t0 + tin]
            past_m = np.repeat(M, tin, axis=0)
            x = np.concatenate([past_c, past_u, past_p, past_m], axis=1)

            y = Cn[t0 + tin:t0 + tin + tout]
            u_future = Un[t0 + tin:t0 + tin + tout]

            X_list.append(x)
            Y_list.append(y)
            U_future_list.append(u_future)
            M_list.append(M)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    Uf = np.stack(U_future_list, axis=0)
    Ms = np.stack(M_list, axis=0)

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("x", data=X, compression="gzip")
        f.create_dataset("y", data=Y, compression="gzip")
        f.create_dataset("u_future", data=Uf, compression="gzip")
        f.create_dataset("mask", data=Ms, compression="gzip")
        f.attrs["c_max"] = stats.c_max
        f.attrs["lambda_log"] = stats.lambda_log
        f.attrs["u_mean"] = stats.u_mean
        f.attrs["u_std"] = stats.u_std
        f.attrs["p_mean"] = stats.p_mean
        f.attrs["p_std"] = stats.p_std
