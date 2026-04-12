# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在处理本仓库代码时提供指导 (Guidance)。

## 项目概览 (Project Overview)

中国象棋棋盘识别 (Chinese Chess Board Recognition)——全卷积网格预测 (Fully Convolutional Grid Prediction)。输入棋盘图像 (Image)，输出 10x9x16 分类矩阵 (Classification Matrix)（90 个棋位，每位置 16 类）。目标部署平台 (Target Deployment Platform) 为 iOS CoreML + Apple 神经网络引擎 (Apple Neural Engine, ANE)。

使用 `timm` 库的 ConvNeXt Atto 模型替代上一代 SwinV2 + MMPretrain 方案 (Scheme)（参考路径 (Reference Path)：`/Users/theone/theone/Programme/my-apps/cchess-apps/chinese-chess-recognition/cchess_reg`）。

## 命令 (Commands)

```bash
# 环境管理 (Environment Management)（使用 uv）
uv sync                              # 安装依赖 (Install Dependencies)
uv run python src/model.py           # 验证模型构建 (Verify Model Build) + 前向传播 (Forward Pass)
uv run python src/inference.py       # 模拟推理演示 (Mock Inference Demo)（需进入目录 `cd src/`）
uv run python convert_coreml.py      # CoreML 格式转换 (CoreML Conversion)（仅限 macOS (macOS only)）

# 训练 (Training)
uv run python -m src.train --data_dir datasets/demo --epochs 100 --batch_size 8
uv run python -m src.train --data_dir /path/to/full_dataset --png_dir datasets/single_cls2_png --fp16

# 恢复训练 (Resume Training)
uv run python -m src.train --data_dir /path/to/dataset --resume_from outputs/checkpoint-XXX

# 可视化数据增强效果 (Visualize Transform Effects)
uv run python -m src.visualize_transforms --image datasets/demo/xxx.jpg --png_dir datasets/single_cls2_png
```

## 架构 (Architecture)

**数据流 (Data Flow)：** `[B,3,640,576]` → ConvNeXt Atto 主干网络 (Backbone) → `[B,320,20,18]` → 步长为 2 的卷积 (Stride-2 Conv) → `[B,128,10,9]` → 膨胀上下文模块 (Dilated Context Module) → 3 层分类器 (3-layer Classifier) (128→64→32→16) → `[B,16,10,9]`

**关键设计约束 (Key Design Constraints)（针对 CoreML/ANE）：**
- 所有通道数 (Channel Numbers) 必须是 32 的倍数
- 只允许使用 2D 卷积 (`Conv2d`)、2D 批归一化 (`BatchNorm2d`)、激活函数 (`ReLU`/`GELU`)、加法 (`Add`)、拼接 (`Concat`)
- 禁止动态形状 (Dynamic Shapes)、自适应池化 (`Adaptive Pooling`)、重塑形状 (`reshape(-1,...)`)
- 固定输入形状 (Fixed Input Shape): `[1, 3, 640, 576]`（高度 Height=640, 宽度 Width=576）

**16 类标签 (16 Class Labels)：** `.`, `x`, K, A, B, N, R, C, P, k, a, b, n, r, c, p

**训练流程 (Training Pipeline)：** 使用 HuggingFace 的训练器 (`Trainer`)。模型的前向传播 `forward()` 在提供标签 (`labels`) 时返回 `{"loss": ..., "logits": ...}`，无标签时返回 Softmax 概率 (Probabilities)。数据加载器 (`DataLoader`) 通过中国象棋训练器 `CChessTrainer`（继承自 `Trainer`）直接控制创建，保留自定义的批次整理函数 (`collate_fn`) 和丢弃末尾批次 (`drop_last`)。支持通过 `--resume_from` 参数从检查点 (Checkpoint) 恢复训练。

## 关键文件 (Key Files)

- `src/model.py` — 中国象棋网络 `CChessNet`（上下文模块 Context Module + 下采样 Downsampling + 3 层 1x1 分类头 Classification Head），主干网络通道数从 `self.backbone.num_features` 动态获取，常量从 `dataset.py` 导入
- `src/dataset.py` — 中国象棋数据集 `CChessDataset` + FEN 解析器 (Parser)，定义所有常量（类别数 `NUM_CLASSES=16`, 图像高度 `IMG_HEIGHT=640`, 图像宽度 `IMG_WIDTH=576`, 裁剪宽度 `CROP_WIDTH=400`, 裁剪高度 `CROP_HEIGHT=450` 等）
- `src/train.py` — HF Trainer 训练入口 (Training Entry)，带数据增强的子集 `SubsetWithTransform` 支持训练集 (train) / 验证集 (val) 分别叠加不同的数据增强管线 (Transform Pipeline)
- `src/evaluate.py` — 中国象棋评估器 `CChessEvaluator`（类别平均精度 Class AP, 位置平均精度 Position AP, 均值平均精度 mAP, 全准确率 Full Accuracy, 帅/将错误率 errK, 精确率/召回率/F1分数 P/R/F1, 仅棋子均值平均精度 piece_only_mAP）
- `src/visualize_transforms.py` — 可视化数据增强各阶段效果 (Visualize transform stages)（用于调试增强管线 Debugging Pipeline）
- `src/transforms/` — 棋盘感知数据增强模块 (Board-aware Data Augmentation Module)
  - `pipeline.py` — 训练增强 `train_transform` / 验证增强 `val_transform` 预定义管线（中心裁剪 `CenterCrop` → 调整大小 `Resize` → 增强 `Augmentation` → 归一化 `Normalize`）
  - `flip.py` — 随机翻转 `CChessRandomFlip`（水平 Horizontal / 垂直 Vertical / 对角 Diagonal，只改变空间位置 Spatial Position 不改变类别 Class）, 半板翻转 `CChessHalfFlip`
  - `mixup.py` — 单一类别 PNG 混合 `CChessMixSinglePngCls`（从 PNG 资源在空位粘贴棋子 Piece Paste，使用扁平索引 Flat Index 定位单元格 Cell）
  - `perspective.py` — 随机透视变换 `RandomPerspective`（优先使用 `cv2`，回退使用 `PIL`）
  - `copy_half.py` — 缓存式半板复制 `CChessCachedCopyHalf`
  - `augment.py` — 颜色抖动 `ColorJitter`, 高斯模糊 `GaussianBlur`（函数式 API Functional API）, 随机擦除 `RandomErasing`
  - `base.py` — 组合 `Compose`, 调整大小 `Resize`, 转换为张量并归一化 `ToTensorNormalize`
- `convert_coreml.py` — 独立 CoreML 转换脚本 (Standalone CoreML Conversion Script)（通过 `torch.jit.trace` → `coremltools`）
- `datasets/demo/` — 示例目录 (Demo Directory)，包含 `.jpg` + `.txt`（FEN 格式 10行x9列）
- `datasets/single_cls2_png/` — 棋子 PNG 图片目录 (Pieces PNG Directory)（用于粘贴棋子增强 PiecePaste Augmentation）
- `colab/cchess_reg_training.ipynb` — Colab 训练笔记本 (Training Notebook)（Google Drive 持久化 Persistence + 检查点恢复 Checkpoint Resume）

## 数据增强 (Data Augmentation)

训练增强管线顺序 (`train_transform` Pipeline Order)（忠实复现旧版 `cchess_reg`，含显式 PIL↔numpy 转换屏障 Conversion Barrier）：
1. **PIL 块 (PIL block):** 中心裁剪 `CenterCrop` → 调整大小 `Resize(640, 576)` → [PIL→numpy]
2. **numpy 块 1 (numpy block 1):** 粘贴棋子 `PiecePaste` → 缓存式半板复制 `CachedCopyHalf` → 随机翻转 `RandomFlip` → 水平半板翻转 `HorizontalHalfFlip` → 垂直半板翻转 `VerticalHalfFlip`
3. [numpy→PIL] → 颜色抖动 `ColorJitter` → [PIL→numpy]
4. **numpy 块 2 (numpy block 2):** 高斯模糊 `GaussianBlur` → 随机擦除 `RandomErasing` ×2 → 随机透视变换 `RandomPerspective` → 转换为张量并归一化 `ToTensorNormalize`

所有翻转 (Flips) 只改变棋子的空间位置 (Spatial Position)，不改变类别 (Class)（`K` 永远是红王 Red King，`k` 永远是黑王 Black King）。

## 活跃技术 (Active Technologies)
- Python 3.11+ + PyTorch, coremltools, timm, torchvision, numpy, PIL, matplotlib, cv2 (003-模型验证 003-model-verification)
- 不适用 (N/A) (仅用于验证，无持久化存储 verification only, no persistent storage) (003-模型验证 003-model-verification)
- Python 3.11+ + 仅标准库 (stdlib only) (`pathlib`, `shutil`, `argparse`) (004-合并数据集文件 004-merge-dataset-files)
- 文件系统 (Filesystem) (将文件从嵌套目录复制到扁平目录 copy files from nested dirs to flat dir) (004-合并数据集文件 004-merge-dataset-files)

## 最近更改 (Recent Changes)
- 003-模型验证 (003-model-verification): 添加了 (Added) Python 3.11+ + PyTorch, coremltools, timm, torchvision, numpy, PIL, matplotlib, cv2
