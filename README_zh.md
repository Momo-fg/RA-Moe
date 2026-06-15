# RA-MoE

本仓库是论文 **RA-MoE: Region-Aware Mixture-of-Experts for AU-Guided Micro-expression Recognition** 的开源代码整理版本，面向 ECML PKDD 2026 论文发布。

当前开源主线聚焦 CASME II 数据集，保留论文中实际有效的三条分支：

- `S`：全局静态外观分支。
- `T`：基于光流的时序运动分支。
- `AU`：由 RegionNet 引导的区域分支。

## 仓库结构

```text
RA-MoE/
  configs/                  # 实验配置
  docs/                     # 数据、权重、复现实验说明
  scripts/
    preprocess/             # CASME II 预处理流程
    train_regionnet.py      # RegionNet 训练入口
    train_ramoe.py          # RA-MoE 训练入口
  src/ramoe/
    datasets/               # 数据读取
    models/                 # RA-MoE 与 RegionNet 模型
    training/               # 训练逻辑
  third_party/              # 第三方代码
  checkpoints/              # 训练生成的 RegionNet 权重
  pretrained/               # 外部预训练权重
```

## 环境安装

```bash
conda env create -f environment.yml
conda activate ramoe
```

或者：

```bash
pip install -r requirements.txt
```

注意：光流生成依赖 `cv2.optflow.DualTVL1OpticalFlow_create()`，因此需要安装 `opencv-contrib-python`。

## 数据准备

本仓库不重新分发 CASME II 数据。请自行从官方渠道申请/下载 CASME II，并将数据整理为：

```text
data/casme2/
  CASME2_RAW_selected/
  CASME2-coding-20140508.xlsx
```

完整预处理流程见 [docs/dataset_preparation.md](docs/dataset_preparation.md)。

## 预训练权重

默认需要将外部权重放到：

```text
pretrained/
  SKD-TSTSAN.pth
  Resnet50_Final.pth
  magnet.pth
  shape_predictor_68_face_landmarks.dat
```

说明见 [docs/model_assets.md](docs/model_assets.md)。

## 训练

训练 RegionNet：

```bash
python scripts/train_regionnet.py \
  --data-root data/casme2/CASME2_LOSO_with_5channel_region \
  --output-dir checkpoints/regionnet
```

训练 RA-MoE：

```bash
python scripts/train_ramoe.py --config configs/casme2.yaml
```

默认实验输出目录为：

```text
outputs/casme2_ramoe/
```

## 引用

```bibtex
@inproceedings{huang2026ramoe,
  title     = {RA-MoE: Region-Aware Mixture-of-Experts for AU-Guided Micro-expression Recognition},
  author    = {Huang, Mingru},
  booktitle = {Proceedings of ECML PKDD},
  year      = {2026}
}
```

## 许可证

本项目使用 MIT License。详见 [LICENSE](LICENSE)。
