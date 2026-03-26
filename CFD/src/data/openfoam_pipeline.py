from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data.city_generator import ParametricCityGenerator
from src.data.openfoam_template import bootstrap_openfoam_base_case
from src.utils.io import read_text, save_json, write_text


@dataclass
class CaseParams:
    case_id: str
    wind_speed: float
    wind_dir_deg: float
    leak_rate: float
    leak_pos: Tuple[float, float, float]


class OpenFOAMPipeline:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.gen_cfg = cfg["data_generation"]
        self.base_case_dir = Path(self.gen_cfg["base_case_dir"])
        self.work_dir = Path(self.gen_cfg["work_dir"])
        self.stl_dir = Path(self.gen_cfg["stl_dir"])
        self.n_cases = int(self.gen_cfg["n_cases"])
        self.wind_speeds = list(self.gen_cfg["wind_speeds"])
        self.wind_dirs = list(self.gen_cfg["wind_directions_deg"])
        self.leak_rate_range = tuple(self.gen_cfg["leak_rate_range"])
        self.leak_h_range = tuple(self.gen_cfg["leak_height_range"])
        self.solver = self.gen_cfg["solver"]
        self.n_procs = int(self.gen_cfg["n_procs"])
        self.rng = np.random.default_rng(cfg["project"]["seed"])

        domain = self.gen_cfg["domain"]
        build = self.gen_cfg["building"]
        self.city_gen = ParametricCityGenerator(
            tuple(domain["x"]),
            tuple(domain["y"]),
            tuple(domain["z"]),
            tuple(build["n_buildings"]),
            tuple(build["width"]),
            tuple(build["depth"]),
            tuple(build["height"]),
            seed=cfg["project"]["seed"],
        )

    def _run(self, cmd: str, cwd: Path) -> None:
        result = subprocess.run(cmd, cwd=str(cwd), shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    def _sample_case_params(self, case_idx: int) -> CaseParams:
        ws = float(self.rng.choice(self.wind_speeds))
        wd = float(self.rng.choice(self.wind_dirs))
        lr = float(self.rng.uniform(*self.leak_rate_range))
        domain = self.gen_cfg["domain"]
        x = float(self.rng.uniform(domain["x"][0] + 5.0, domain["x"][1] - 5.0))
        y = float(self.rng.uniform(domain["y"][0] + 5.0, domain["y"][1] - 5.0))
        z = float(self.rng.uniform(*self.leak_h_range))
        return CaseParams(case_id=f"case_{case_idx:05d}", wind_speed=ws, wind_dir_deg=wd, leak_rate=lr, leak_pos=(x, y, z))

    def _inject_boundary_conditions(self, case_dir: Path, params: CaseParams) -> None:
        ux = params.wind_speed * math.cos(math.radians(params.wind_dir_deg))
        uy = params.wind_speed * math.sin(math.radians(params.wind_dir_deg))

        u_file = case_dir / "0" / "U"
        gas_file = case_dir / "0" / "gas"

        if u_file.exists():
            txt = read_text(u_file)
            txt = txt.replace("__INLET_UX__", f"{ux:.6f}")
            txt = txt.replace("__INLET_UY__", f"{uy:.6f}")
            txt = txt.replace("__INLET_UZ__", "0.0")
            write_text(u_file, txt)

        if gas_file.exists():
            txt = read_text(gas_file)
            txt = txt.replace("__LEAK_RATE__", f"{params.leak_rate:.8f}")
            txt = txt.replace("__LEAK_X__", f"{params.leak_pos[0]:.4f}")
            txt = txt.replace("__LEAK_Y__", f"{params.leak_pos[1]:.4f}")
            txt = txt.replace("__LEAK_Z__", f"{params.leak_pos[2]:.4f}")
            write_text(gas_file, txt)

    def _ensure_base_case(self) -> None:
        if (not self.base_case_dir.exists()) or (not any(self.base_case_dir.iterdir())):
            template_cfg = self.gen_cfg.get("template", {})
            template_cfg.setdefault("domain", self.gen_cfg["domain"])
            template_cfg.setdefault("solver", self.solver)
            bootstrap_openfoam_base_case(self.base_case_dir, template_cfg)

    def prepare_case(self, case_idx: int) -> Tuple[Path, CaseParams]:
        self._ensure_base_case()

        params = self._sample_case_params(case_idx)
        case_dir = self.work_dir / params.case_id
        if case_dir.exists():
            shutil.rmtree(case_dir)
        shutil.copytree(self.base_case_dir, case_dir)

        stl_path = self.stl_dir / f"{params.case_id}.stl"
        meta_path = case_dir / "layout.json"
        self.city_gen.export(stl_path, meta_path)

        tri_surface_dir = case_dir / "constant" / "triSurface"
        tri_surface_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stl_path, tri_surface_dir / "buildings.stl")

        self._inject_boundary_conditions(case_dir, params)

        save_json(
            {
                "case_id": params.case_id,
                "wind_speed": params.wind_speed,
                "wind_dir_deg": params.wind_dir_deg,
                "leak_rate": params.leak_rate,
                "leak_pos": list(params.leak_pos),
            },
            case_dir / "case_params.json",
        )
        return case_dir, params

    def run_case(self, case_dir: Path) -> None:
        self._run("blockMesh", case_dir)
        self._run("snappyHexMesh -overwrite", case_dir)

        decompose_dict = case_dir / "system" / "decomposeParDict"
        if decompose_dict.exists() and self.n_procs > 1:
            txt = read_text(decompose_dict)
            txt = txt.replace("__NPROCS__", str(self.n_procs))
            write_text(decompose_dict, txt)
            self._run("decomposePar -force", case_dir)
            self._run(f"mpirun -np {self.n_procs} {self.solver} -parallel", case_dir)
            self._run("reconstructPar", case_dir)
        else:
            self._run(self.solver, case_dir)

    def run_all(self) -> List[Path]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.stl_dir.mkdir(parents=True, exist_ok=True)
        cases = []
        for i in range(self.n_cases):
            case_dir, _ = self.prepare_case(i)
            self.run_case(case_dir)
            cases.append(case_dir)
        return cases
