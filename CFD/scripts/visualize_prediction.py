from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_slice_triplet(pred: np.ndarray, gt: np.ndarray, out_png: Path, title: str) -> None:
    err = np.abs(pred - gt)
    vmax = max(float(pred.max()), float(gt.max()), 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(pred, cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title("Prediction")
    axes[1].imshow(gt, cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(err, cmap="magma")
    axes[2].set_title("Abs Error")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default="outputs/inference/pred.npy")
    parser.add_argument("--target", type=str, default="outputs/inference/target.npy")
    parser.add_argument("--out_dir", type=str, default="outputs/inference/vis")
    parser.add_argument("--time_idx", type=int, default=0)
    parser.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"])
    parser.add_argument("--slice_idx", type=int, default=-1)
    args = parser.parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.target)

    t = min(max(args.time_idx, 0), pred.shape[0] - 1)
    pred_t = pred[t, 0]
    gt_t = gt[t, 0]

    if args.axis == "z":
        s = pred_t.shape[0] // 2 if args.slice_idx < 0 else min(max(args.slice_idx, 0), pred_t.shape[0] - 1)
        p2d, g2d = pred_t[s], gt_t[s]
        title = f"t={t}, z-slice={s}"
    elif args.axis == "y":
        s = pred_t.shape[1] // 2 if args.slice_idx < 0 else min(max(args.slice_idx, 0), pred_t.shape[1] - 1)
        p2d, g2d = pred_t[:, s, :], gt_t[:, s, :]
        title = f"t={t}, y-slice={s}"
    else:
        s = pred_t.shape[2] // 2 if args.slice_idx < 0 else min(max(args.slice_idx, 0), pred_t.shape[2] - 1)
        p2d, g2d = pred_t[:, :, s], gt_t[:, :, s]
        title = f"t={t}, x-slice={s}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_slice_triplet(p2d, g2d, out_dir / "slice_compare.png", title)

    mse_curve = ((pred - gt) ** 2).mean(axis=(1, 2, 3, 4))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(mse_curve)), mse_curve, marker="o")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("MSE")
    ax.set_title("Temporal MSE")
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_mse.png", dpi=160)
    plt.close(fig)

    print(f"Saved visualization to {out_dir}")


if __name__ == "__main__":
    main()
