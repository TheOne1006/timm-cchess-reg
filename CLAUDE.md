# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

中国象棋棋盘识别——全卷积网格预测 (Fully Convolutional Grid Prediction)。输入棋盘图像，输出 10x9x16 分类矩阵（90 个棋位，每位置 16 类）。目标部署平台为 iOS CoreML + Apple Neural Engine (ANE)。

使用 timm ConvNeXt Atto 替代上一代 SwinV2 + MMPretrain 方案（参考 `/Users/theone/theone/Programme/my-apps/cchess-apps/chinese-chess-recognition/cchess_reg`）。

## Commands

```bash
# 环境管理（使用 uv）
uv sync                              # 安装依赖
uv run python src/model.py           # 验证模型构建 + 前向传播
uv run python src/inference.py       # Mock 推理演示（需 cd src/）
uv run python convert_coreml.py      # CoreML 转换（macOS only）

# 训练
uv run python -m src.train --data_dir datasets/demo --epochs 100 --batch_size 8
uv run python -m src.train --data_dir /path/to/full_dataset --png_dir datasets/single_cls2_png --fp16

# 恢复训练
uv run python -m src.train --data_dir /path/to/dataset --resume_from outputs/checkpoint-XXX

# 可视化 transform 效果
uv run python -m src.visualize_transforms --image datasets/demo/xxx.jpg --png_dir datasets/single_cls2_png
```

## Architecture

**数据流：** `[B,3,640,576]` → ConvNeXt Atto backbone → `[B,320,20,18]` → stride-2 conv → `[B,128,10,9]` → dilated context module → 3-layer classifier (128→64→32→16) → `[B,16,10,9]`

**关键设计约束（CoreML/ANE）：**
- 所有通道数必须是 32 的倍数
- 只允许 Conv2d, BatchNorm2d, ReLU/GELU, Add, Concat
- 禁止动态形状、Adaptive Pooling、reshape(-1,...)
- 固定输入形状: `[1, 3, 640, 576]`（Height=640, Width=576）

**16 类标签：** `.`, `x`, K, A, B, N, R, C, P, k, a, b, n, r, c, p

**训练流程：** 使用 HuggingFace Trainer。模型 `forward()` 在提供 labels 时返回 `{"loss": ..., "logits": ...}`，无 labels 时返回 softmax 概率。DataLoader 通过 `CChessTrainer`（继承 Trainer）直接控制 DataLoader 创建，保留自定义 `collate_fn` 和 `drop_last`。支持 `--resume_from` 从 checkpoint 恢复训练。

## Key Files

- `src/model.py` — CChessNet（ContextModule + 下采样 + 3层1x1分类头），backbone 通道数从 `self.backbone.num_features` 动态获取，常量从 `dataset.py` 导入
- `src/dataset.py` — CChessDataset + FEN parser，定义所有常量（NUM_CLASSES=16, IMG_HEIGHT=640, IMG_WIDTH=576, CROP_WIDTH=400, CROP_HEIGHT=450 等）
- `src/train.py` — HF Trainer 训练入口，SubsetWithTransform 支持 train/val 分别叠加不同 transform
- `src/evaluate.py` — CChessEvaluator（class AP, position AP, mAP, full accuracy, errK, P/R/F1, piece_only_mAP）
- `src/visualize_transforms.py` — 可视化 transform 各阶段效果（调试增强管线）
- `src/transforms/` — 棋盘感知数据增强模块
  - `pipeline.py` — train_transform / val_transform 预定义管线（CenterCrop → Resize → 增强 → Normalize）
  - `flip.py` — CChessRandomFlip（水平/垂直/对角，只改变空间位置不改变类别）, CChessHalfFlip
  - `mixup.py` — CChessMixSinglePngCls（从 PNG 资源在空位粘贴棋子，使用 flat index 定位 cell）
  - `perspective.py` — RandomPerspective（cv2 优先，PIL 回退）
  - `copy_half.py` — CChessCachedCopyHalf（缓存式半板复制）
  - `augment.py` — ColorJitter, GaussianBlur（functional API）, RandomErasing
  - `base.py` — Compose, Resize, ToTensorNormalize
- `convert_coreml.py` — 独立 CoreML 转换脚本（torch.jit.trace → coremltools）
- `datasets/demo/` — 示例 .jpg + .txt（FEN 格式 10行x9列）
- `datasets/single_cls2_png/` — 棋子 PNG 图片（用于 PiecePaste 增强）
- `colab/cchess_reg_training.ipynb` — Colab 训练 notebook（Drive 持久化 + checkpoint resume）

## Data Augmentation

train_transform 管线顺序（忠实复现旧版 cchess_reg，含显式 PIL↔numpy 转换屏障）：
1. **PIL block:** CenterCrop → Resize(640, 576) → [PIL→numpy]
2. **numpy block 1:** PiecePaste → CachedCopyHalf → RandomFlip → HorizontalHalfFlip → VerticalHalfFlip
3. [numpy→PIL] → ColorJitter → [PIL→numpy]
4. **numpy block 2:** GaussianBlur → RandomErasing×2 → RandomPerspective → ToTensorNormalize

所有翻转只改变棋子的空间位置，不改变类别（K 永远是红王，k 永远是黑王）。

## Active Technologies
- Python 3.11+ + PyTorch, coremltools, timm, torchvision, numpy, PIL, matplotlib, cv2 (003-model-verification)
- N/A (verification only, no persistent storage) (003-model-verification)

## Recent Changes
- 003-model-verification: Added Python 3.11+ + PyTorch, coremltools, timm, torchvision, numpy, PIL, matplotlib, cv2
