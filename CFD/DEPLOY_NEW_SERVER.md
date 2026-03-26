# 全新服务器手动部署指南（GasDispersion_3DMambaPINN）

本指南用于在一台全新 Linux 服务器上手动部署并运行本项目。

---

## 1. 系统准备

推荐：Ubuntu 20.04/22.04，NVIDIA GPU（已安装驱动/CUDA）。

```bash
sudo apt update
sudo apt install -y git wget curl build-essential cmake unzip htop tmux
```

检查 GPU：

```bash
nvidia-smi
```

---

## 2. 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 3. 拉取项目

```bash
git clone <你的仓库地址> GasDispersion_3DMambaPINN
cd GasDispersion_3DMambaPINN
```

---

## 4. 创建 Python 环境并安装依赖

```bash
conda create -n gas3d python=3.10 -y
conda activate gas3d
```

按 CUDA 版本安装 PyTorch（示例 cu121）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装项目依赖：

```bash
pip install -r requirements.txt
pip install mamba-ssm einops h5py pyyaml matplotlib trimesh pyvista vtk ninja
```

验证：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 5. 安装 OpenFOAM

示例安装 openfoam10：

```bash
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository -y http://dl.openfoam.org/ubuntu
sudo apt update
sudo apt install -y openfoam10
```

加入环境变量：

```bash
echo 'source /opt/openfoam10/etc/bashrc' >> ~/.bashrc
source ~/.bashrc
```

验证：

```bash
which blockMesh
which snappyHexMesh
```

---

## 6. 修改配置

编辑 `configs/default.yaml`（至少改以下项）：

- `data_generation.base_case_dir`: 建议 `./base_case_auto`
- `data_generation.n_cases`: 首次联调建议 `5~10`
- `data_generation.n_procs`: 按 CPU 设置，如 `8`
- `train.batch_size`: 按显存大小调整，建议先 `1`
- `train.epochs`: 联调先设 `2~5`

---

## 7. 运行方式

### 7.1 一键全流程

```bash
python scripts/run_all.py --config configs/default.yaml
```

### 7.2 分步运行

```bash
python scripts/generate_data.py --config configs/default.yaml
python scripts/preprocess_dataset.py --config configs/default.yaml --input_dir generated_cases/processed --output_dir dataset
python scripts/train.py --config configs/default.yaml
python scripts/inference.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt --input_h5 dataset/val.h5 --sample_idx 0 --out_dir outputs/inference
python scripts/visualize_prediction.py --pred outputs/inference/pred.npy --target outputs/inference/target.npy --out_dir outputs/inference/vis
```

---

## 8. 后台运行（推荐）

```bash
tmux new -s gas3d
conda activate gas3d
python scripts/run_all.py --config configs/default.yaml
```

分离：`Ctrl+B` 后按 `D`

恢复：

```bash
tmux attach -t gas3d
```

---

## 9. 常见问题

### 9.1 `blockMesh: command not found`

```bash
source /opt/openfoam10/etc/bashrc
```

### 9.2 GPU 不可用

检查驱动/CUDA 与 PyTorch 版本匹配。

### 9.3 `mamba_ssm` 导入失败

```bash
pip install --upgrade pip setuptools wheel ninja
pip install mamba-ssm
```

### 9.4 训练 OOM

- 降低 `train.batch_size`
- 减少 `model.base_channels`
- 降低网格分辨率

---

## 10. 建议上线流程

1. 先用 `n_cases=5` 小规模跑通。
2. 检查 `outputs/inference/vis` 可视化是否合理。
3. 再放大到正式规模（如 `n_cases=100+`）。
