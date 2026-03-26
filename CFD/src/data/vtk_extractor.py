from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv


class VTKFieldExtractor:
    def __init__(self, grid_shape: Tuple[int, int, int], domain: Dict[str, List[float]]):
        self.grid_shape = grid_shape
        self.domain = domain

    def _uniform_grid(self) -> pv.ImageData:
        nx, ny, nz = self.grid_shape
        x0, x1 = self.domain["x"]
        y0, y1 = self.domain["y"]
        z0, z1 = self.domain["z"]

        grid = pv.ImageData()
        grid.dimensions = (nx, ny, nz)
        grid.origin = (x0, y0, z0)
        grid.spacing = ((x1 - x0) / max(nx - 1, 1), (y1 - y0) / max(ny - 1, 1), (z1 - z0) / max(nz - 1, 1))
        return grid

    def extract_case(self, case_dir: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        vtk_files = sorted((case_dir / "VTK").glob("*.vtk"))
        if not vtk_files:
            # try foamToVTK convention
            vtk_files = sorted(case_dir.glob("**/VTK/*.vtk"))
        if not vtk_files:
            raise FileNotFoundError(f"No VTK files found under {case_dir}")

        target = self._uniform_grid()
        c_list, u_list, p_list = [], [], []

        for vf in vtk_files:
            mesh = pv.read(vf)
            sampled = target.sample(mesh)

            # Fallback aliases commonly used by OpenFOAM exports
            c_name = "gas" if "gas" in sampled.array_names else ("C" if "C" in sampled.array_names else None)
            p_name = "p" if "p" in sampled.array_names else None
            u_name = "U" if "U" in sampled.array_names else None

            if c_name is None or p_name is None or u_name is None:
                raise KeyError(f"Required fields missing in {vf.name}. got={sampled.array_names}")

            c = sampled[c_name].reshape(self.grid_shape, order="F")
            p = sampled[p_name].reshape(self.grid_shape, order="F")
            u = sampled[u_name].reshape((*self.grid_shape, 3), order="F")

            c_list.append(c)
            p_list.append(p)
            u_list.append(np.moveaxis(u, -1, 0))  # (3, D,H,W)

        C = np.stack(c_list, axis=0)[:, None, ...]  # (T,1,D,H,W)
        P = np.stack(p_list, axis=0)[:, None, ...]  # (T,1,D,H,W)
        U = np.stack(u_list, axis=0)  # (T,3,D,H,W)

        # mask: fluid=1, solid=0 (placeholder by occupancy check via concentration NaN handling)
        mask = np.isfinite(C[0:1]).astype(np.float32)

        C = np.nan_to_num(C, nan=0.0).astype(np.float32)
        P = np.nan_to_num(P, nan=0.0).astype(np.float32)
        U = np.nan_to_num(U, nan=0.0).astype(np.float32)

        np.save(out_dir / "C.npy", C)
        np.save(out_dir / "p.npy", P)
        np.save(out_dir / "U.npy", U)
        np.save(out_dir / "mask.npy", mask)
