from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.config import load_config


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_infer", action="store_true")
    parser.add_argument("--skip_vis", action="store_true")
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)

    py = sys.executable
    if not args.skip_generate:
        _run([py, "scripts/generate_data.py", "--config", args.config])

    if not args.skip_preprocess:
        _run(
            [
                py,
                "scripts/preprocess_dataset.py",
                "--config",
                args.config,
                "--input_dir",
                str(Path(cfg["data_generation"]["work_dir"]) / "processed"),
                "--output_dir",
                "dataset",
            ]
        )

    if not args.skip_train:
        _run([py, "scripts/train.py", "--config", args.config])

    ckpt = str(Path(cfg["train"]["checkpoint_dir"]) / "best.pt")
    infer_out = "outputs/inference"
    if not args.skip_infer:
        _run(
            [
                py,
                "scripts/inference.py",
                "--config",
                args.config,
                "--checkpoint",
                ckpt,
                "--input_h5",
                cfg["train"]["val_data_h5"],
                "--sample_idx",
                str(args.sample_idx),
                "--out_dir",
                infer_out,
            ]
        )

    if not args.skip_vis:
        _run(
            [
                py,
                "scripts/visualize_prediction.py",
                "--pred",
                f"{infer_out}/pred.npy",
                "--target",
                f"{infer_out}/target.npy",
                "--out_dir",
                f"{infer_out}/vis",
            ]
        )

    print("All done.")


if __name__ == "__main__":
    main()
