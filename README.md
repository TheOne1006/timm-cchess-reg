# timm-cchess-reg

中国象棋棋盘识别 — 全卷积网格预测（Fully Convolutional Grid Prediction）。

输入棋盘图像，输出 `10×9×16` 分类矩阵（90 个棋位，每位置 16 类）。基于 timm ConvNeXt Atto 骨干网络，目标部署平台为 **iOS CoreML + Apple Neural Engine (ANE)**。

## 架构

```
输入 [B, 3, 640, 576]
  → ConvNeXt Atto backbone          → [B, 320, 20, 18]
  → stride-2 conv (320→128)         → [B, 128, 10, 9]
  → ContextModule (双分支空洞卷积)   → [B, 128, 10, 9]
  → 3 层 1×1 分类头 (128→64→32→16) → [B, 16, 10, 9]
```

**16 类标签：** `.`, `x`, `K`, `A`, `B`, `N`, `R`, `C`, `P`, `k`, `a`, `b`, `n`, `r`, `c`, `p`

**CoreML/ANE 设计约束：**
- 所有通道数为 32 的倍数
- 仅使用 Conv2d、BatchNorm2d、ReLU、Add、Concat
- 固定输入形状 `[1, 3, 640, 576]`（Height=640, Width=576）

## 项目结构

```
timm-cchess-reg/
├── src/
│   ├── model.py            # CChessNet 模型定义
│   ├── dataset.py          # CChessDataset + FEN 解析
│   ├── train.py            # HuggingFace Trainer 训练入口
│   ├── evaluate.py         # 多级评估指标（AP/mAP/P/R/F1）
│   ├── inference.py        # 推理演示
│   └── transforms/         # 棋盘感知数据增强
│       ├── pipeline.py     # train_transform / val_transform 预定义管线
│       ├── flip.py         # 水平/垂直/对角翻转 + 半板翻转
│       ├── mixup.py        # PNG 棋子粘贴增强
│       ├── perspective.py  # 随机透视变换
│       ├── copy_half.py    # 缓存式半板复制
│       ├── augment.py      # ColorJitter, GaussianBlur, RandomErasing
│       └── base.py         # Compose, Resize, ToTensorNormalize
├── datasets/
│   ├── demo/               # 示例数据（.jpg + .txt 配对）
│   └── single_cls2_png/    # 棋子 PNG 资源（14 类，用于 PiecePaste 增强）
├── convert_coreml.py       # CoreML 模型转换脚本
├── pyproject.toml          # 项目配置与依赖
└── uv.lock                 # uv 锁文件
```

## 安装

项目使用 [uv](https://docs.astral.sh/uv/) 作为包管理器，要求 Python ≥ 3.12。

```bash
# 克隆仓库
git clone <repo-url>
cd timm-cchess-reg

# 安装依赖
uv sync
```

## 使用

### 训练

```bash
# 使用 demo 数据训练 100 个 epoch
uv run python -m src.train --data_dir datasets/full --epochs 100 --batch_size 8

# 使用完整数据集，启用 FP16 和棋子粘贴增强
uv run python -m src.train \
  --data_dir /path/to/full_dataset \
  --png_dir datasets/single_cls2_png \
  --fp16

# 更多训练参数
uv run python -m src.train --help
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | （必填） | 数据集目录（.jpg + .txt 配对） |
| `--png_dir` | `None` | 棋子 PNG 目录（启用 PiecePaste 增强） |
| `--backbone` | `convnext_atto` | 骨干网络名称 |
| `--epochs` | `100` | 训练轮数 |
| `--batch_size` | `8` | 批大小 |
| `--lr` | `1e-3` | 学习率 |
| `--fp16` | `False` | 启用混合精度训练 |
| `--output_dir` | `outputs/` | 模型输出目录 |

### 推理

```bash
uv run python -m src.inference
```

### CoreML 转换

```bash
# 需要在 macOS 上运行
uv run python convert_coreml.py
```

转换结果保存到项目根目录的 `CChessNet.mlpackage/`。

## 数据格式

数据集目录包含成对的 `.jpg` 和 `.txt` 文件：

- **`.jpg`** — 棋盘图像（固定缩放至 640×576）
- **`.txt`** — FEN 标签，10 行 × 9 列，每行 9 个字符

标签字符含义：

| 字符 | `.` | `x` | `K` | `A` | `B` | `N` | `R` | `C` | `P` |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 含义 | 空位 | 其他 | 红帅 | 红仕 | 红相 | 红马 | 红车 | 红炮 | 红兵 |

| 字符 | `k` | `a` | `b` | `n` | `r` | `c` | `p` |
|------|-----|-----|-----|-----|-----|-----|-----|
| 含义 | 黑将 | 黑士 | 黑象 | 黑马 | 黑车 | 黑炮 | 黑卒 |

## 数据增强

训练管线按以下顺序执行：

1. Resize(640, 576)
2. PiecePaste — 从 PNG 资源在空位粘贴棋子
3. CachedCopyHalf — 缓存式半板复制
4. RandomFlip — 水平/垂直/对角翻转（至多执行一个方向）
5. HorizontalHalfFlip — 水平半板翻转
6. VerticalHalfFlip — 垂直半板翻转
7. GaussianBlur
8. ColorJitter
9. RandomErasing × 2
10. RandomPerspective
11. ToTensorNormalize

验证管线仅包含：Resize → ToTensorNormalize。

所有翻转操作只改变棋子的空间位置，不改变类别（K 永远是红帅，k 永远是黑将）。

## 评估指标

模型评估包含以下多级指标：

- **Class-level AP** — 16 类各自的 Average Precision + mAP
- **Position-level AP** — 90 个棋位各自的 mAP
- **Full accuracy** — 完全正确率 + errK 容错（err1/err3/err5）
- **P/R/F1** — 每类及 Macro/Micro 精确率/召回率/F1
- **Piece-only mAP** — 排除空位和"其他"类后的 mAP

## 依赖

- Python ≥ 3.12
- PyTorch ≥ 2.5
- timm ≥ 1.0.26
- transformers ≥ 4.40
- accelerate ≥ 1.1.0
- coremltools ≥ 8.0（CoreML 转换，仅 macOS）
- opencv-python-headless ≥ 4.13
- Pillow ≥ 10.0
