from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preprocess import build_h5_dataset, compute_stats
from src.utils.config import load_config


def split_cases(case_dirs: list[Path], val_ratio: float = 0.2) -> tuple[list[Path], list[Path]]:
    n = len(case_dirs)
    n_val = max(1, int(n * val_ratio)) if n > 1 else 0
    return case_dirs[:-n_val] if n_val > 0 else case_dirs, case_dirs[-n_val:] if n_val > 0 else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tin", type=int, default=4)
    parser.add_argument("--tout", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    case_dirs = sorted([p for p in Path(args.input_dir).iterdir() if p.is_dir()])
    train_cases, val_cases = split_cases(case_dirs, val_ratio=0.2)

    stats = compute_stats(train_cases, lambda_log=float(cfg["preprocess"]["lambda_log"]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_h5_dataset(train_cases, out_dir / "train.h5", stats, tin=args.tin, tout=args.tout, stride=args.stride)
    if val_cases:
        build_h5_dataset(val_cases, out_dir / "val.h5", stats, tin=args.tin, tout=args.tout, stride=args.stride)
    else:
        build_h5_dataset(train_cases, out_dir / "val.h5", stats, tin=args.tin, tout=args.tout, stride=args.stride)

    print("Dataset preprocessing completed.")


if __name__ == "__main__":
    main()
