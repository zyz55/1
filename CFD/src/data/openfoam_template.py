from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.utils.io import write_text


def _g(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def _control_dict(cfg: Dict[str, Any]) -> str:
    t = cfg.get("time", {})
    solver = _g(cfg, "solver", "buoyantPimpleFoam")
    return (
        f"application {solver};\n"
        f"startTime {_g(t, 'start_time', 0)};\n"
        f"endTime {_g(t, 'end_time', 100)};\n"
        f"deltaT {_g(t, 'delta_t', 0.5)};\n"
        "writeControl timeStep;\n"
        f"writeInterval {_g(t, 'write_interval', 5)};\n"
        "runTimeModifiable true;\n"
    )


def _block_mesh_dict(cfg: Dict[str, Any]) -> str:
    d = cfg["domain"]
    nx, ny, nz = cfg.get("mesh_cells", [80, 80, 40])
    x0, x1 = d["x"]
    y0, y1 = d["y"]
    z0, z1 = d["z"]
    return (
        "convertToMeters 1;\n"
        f"vertices (({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0}) "
        f"({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1}));\n"
        f"blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));\n"
        "boundary (inlet { type patch; faces ((0 3 7 4)); } "
        "outlet { type patch; faces ((1 2 6 5)); } "
        "walls { type wall; faces ((0 1 5 4)(3 2 6 7)(0 1 2 3)); } "
        "top { type patch; faces ((4 5 6 7)); });\n"
    )


def _fv_schemes(cfg: Dict[str, Any]) -> str:
    div_u = _g(cfg, "div_phi_u", "Gauss linearUpwind grad(U)")
    div_c = _g(cfg, "div_phi_gas", "Gauss upwind")
    grad = _g(cfg, "grad_default", "Gauss linear")
    return (
        f"gradSchemes {{ default {grad}; }}\n"
        "ddtSchemes { default Euler; }\n"
        f"divSchemes {{ default none; div(phi,U) {div_u}; div(phi,gas) {div_c}; }}\n"
        "laplacianSchemes { default Gauss linear corrected; }\n"
        "interpolationSchemes { default linear; }\n"
        "snGradSchemes { default corrected; }\n"
    )


def _fv_solution(cfg: Dict[str, Any]) -> str:
    return (
        "solvers { "
        "p { solver PCG; preconditioner DIC; tolerance 1e-6; relTol 0.1; } "
        "U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-6; relTol 0.1; } "
        "gas { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }"
        " }\n"
        "PIMPLE { "
        f"nOuterCorrectors {_g(cfg, 'n_outer_correctors', 2)}; "
        f"nCorrectors {_g(cfg, 'n_correctors', 2)}; "
        f"nNonOrthogonalCorrectors {_g(cfg, 'n_non_ortho_correctors', 1)};"
        " }\n"
    )


def _snappy_dict(cfg: Dict[str, Any]) -> str:
    s = cfg.get("snappy", {})
    max_local = _g(s, "max_local_cells", 2_000_000)
    max_global = _g(s, "max_global_cells", 6_000_000)
    refine_min = _g(s, "surface_refinement_min", 2)
    refine_max = _g(s, "surface_refinement_max", 3)
    loc = _g(s, "location_in_mesh", [10, 10, 2])
    return (
        "castellatedMesh true; snap true; addLayers false;\n"
        "geometry { buildings.stl { type triSurfaceMesh; name buildings; } }\n"
        "castellatedMeshControls { "
        f"maxLocalCells {max_local}; maxGlobalCells {max_global}; "
        "minRefinementCells 0; nCellsBetweenLevels 2; features (); "
        f"refinementSurfaces {{ buildings {{ level ({refine_min} {refine_max}); }} }} "
        f"locationInMesh ({loc[0]} {loc[1]} {loc[2]});"
        " }\n"
        "snapControls { nSmoothPatch 3; tolerance 2.0; nSolveIter 30; nRelaxIter 5; }\n"
        "meshQualityControls { maxNonOrtho 65; maxInternalSkewness 4; minVol 1e-13; }\n"
    )


def bootstrap_openfoam_base_case(base_case_dir: str | Path, cfg: Dict[str, Any]) -> Path:
    base = Path(base_case_dir)
    (base / "0").mkdir(parents=True, exist_ok=True)
    (base / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (base / "system").mkdir(parents=True, exist_ok=True)

    write_text(base / "0" / "U", "internalField uniform (__INLET_UX__ __INLET_UY__ __INLET_UZ__);\n")
    write_text(base / "0" / "gas", "internalField uniform 0;\n// leak __LEAK_RATE__ @ (__LEAK_X__ __LEAK_Y__ __LEAK_Z__)\n")
    write_text(base / "0" / "p", "internalField uniform 0;\n")

    write_text(base / "constant" / "transportProperties", "transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] 1.5e-05;\n")
    write_text(base / "constant" / "turbulenceProperties", "simulationType RAS;\n")

    write_text(base / "system" / "controlDict", _control_dict(cfg))
    write_text(base / "system" / "fvSchemes", _fv_schemes(cfg))
    write_text(base / "system" / "fvSolution", _fv_solution(cfg))
    write_text(base / "system" / "decomposeParDict", "numberOfSubdomains __NPROCS__;\nmethod scotch;\n")
    write_text(base / "system" / "blockMeshDict", _block_mesh_dict(cfg))
    write_text(base / "system" / "snappyHexMeshDict", _snappy_dict(cfg))

    return base
