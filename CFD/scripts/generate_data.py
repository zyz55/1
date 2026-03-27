from __future__ import annotations

import argparse
from pathlib import Path

from src.data.openfoam_pipeline import OpenFOAMPipeline
from src.data.vtk_extractor import VTKFieldExtractor
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = OpenFOAMPipeline(cfg)
    case_dirs = pipeline.run_all()

    grid_shape = tuple(cfg["data_generation"]["mesh"]["cartesian_grid"])
    domain = cfg["data_generation"]["domain"]
    extractor = VTKFieldExtractor(grid_shape=grid_shape, domain=domain)

    processed_root = Path(cfg["data_generation"]["work_dir"]) / "processed"
    for cd in case_dirs:
        out = processed_root / cd.name
        extractor.extract_case(cd, out)

    print(f"Generated and processed {len(case_dirs)} cases.")


if __name__ == "__main__":
    main()
