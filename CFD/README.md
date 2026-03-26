# GasDispersion_3DMambaPINN

3D CNN + Mamba + Physics-Informed loss framework for urban gas dispersion forecasting.

## Install

```bash
pip install -r requirements.txt
```

## One-click full pipeline

```bash
python scripts/run_all.py --config configs/default.yaml
```

Skip selected stages if artifacts already exist:

```bash
python scripts/run_all.py --config configs/default.yaml --skip_generate --skip_preprocess --skip_train
```

## OpenFOAM template parameterization

`configs/default.yaml` 中的 `data_generation.template` 用于自动生成基础 OpenFOAM case（当 `base_case_dir` 为空或不存在时自动触发）。

关键参数：

- `solver`: 求解器名称（如 `buoyantPimpleFoam`）
- `domain.{x,y,z}`: 计算域范围（用于 `blockMeshDict`）
- `mesh_cells`: 背景六面体网格分辨率
- `time.{start_time,end_time,delta_t,write_interval}`: 时间推进与写出控制
- `n_outer_correctors`, `n_correctors`, `n_non_ortho_correctors`: PIMPLE 配置（`fvSolution`）
- `grad_default`, `div_phi_u`, `div_phi_gas`: `fvSchemes` 梯度/对流离散格式
- `snappy.max_local_cells`, `snappy.max_global_cells`, `snappy.surface_refinement_min`, `snappy.surface_refinement_max`, `snappy.location_in_mesh`: `snappyHexMeshDict` 细化参数

## Step-by-step run

```bash
python scripts/generate_data.py --config configs/default.yaml
python scripts/preprocess_dataset.py --config configs/default.yaml --input_dir generated_cases/processed --output_dir dataset
python scripts/train.py --config configs/default.yaml
python scripts/inference.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --input_h5 dataset/val.h5 --sample_idx 0 --out_dir outputs/inference
python scripts/visualize_prediction.py --pred outputs/inference/pred.npy --target outputs/inference/target.npy --out_dir outputs/inference/vis
```
