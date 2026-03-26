from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh


@dataclass
class Building:
    center: Tuple[float, float]
    width: float
    depth: float
    height: float


@dataclass
class CityLayout:
    domain_x: Tuple[float, float]
    domain_y: Tuple[float, float]
    domain_z: Tuple[float, float]
    buildings: List[Building]


class ParametricCityGenerator:
    def __init__(
        self,
        domain_x: Tuple[float, float],
        domain_y: Tuple[float, float],
        domain_z: Tuple[float, float],
        n_buildings_range: Tuple[int, int],
        width_range: Tuple[float, float],
        depth_range: Tuple[float, float],
        height_range: Tuple[float, float],
        seed: int = 42,
    ) -> None:
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.domain_z = domain_z
        self.n_buildings_range = n_buildings_range
        self.width_range = width_range
        self.depth_range = depth_range
        self.height_range = height_range
        self.rng = np.random.default_rng(seed)

    def sample_layout(self) -> CityLayout:
        n = int(self.rng.integers(self.n_buildings_range[0], self.n_buildings_range[1] + 1))
        buildings: List[Building] = []
        for _ in range(n):
            w = float(self.rng.uniform(*self.width_range))
            d = float(self.rng.uniform(*self.depth_range))
            h = float(self.rng.uniform(*self.height_range))
            x = float(self.rng.uniform(self.domain_x[0] + w / 2, self.domain_x[1] - w / 2))
            y = float(self.rng.uniform(self.domain_y[0] + d / 2, self.domain_y[1] - d / 2))
            buildings.append(Building(center=(x, y), width=w, depth=d, height=h))
        return CityLayout(self.domain_x, self.domain_y, self.domain_z, buildings)

    def layout_to_mesh(self, layout: CityLayout) -> trimesh.Trimesh:
        parts = []
        for b in layout.buildings:
            mesh = trimesh.creation.box(extents=[b.width, b.depth, b.height])
            mesh.apply_translation([b.center[0], b.center[1], b.height / 2])
            parts.append(mesh)
        if not parts:
            return trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        return trimesh.util.concatenate(parts)

    def export(self, out_stl: Path, out_json: Path) -> CityLayout:
        layout = self.sample_layout()
        mesh = self.layout_to_mesh(layout)
        out_stl.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_stl)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "domain_x": list(layout.domain_x),
                    "domain_y": list(layout.domain_y),
                    "domain_z": list(layout.domain_z),
                    "buildings": [asdict(b) for b in layout.buildings],
                },
                f,
                indent=2,
            )
        return layout
