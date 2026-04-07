# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

中国象棋棋盘识别项目——全卷积网格预测 (Fully Convolutional Grid Prediction)。输入棋盘图像，输出 10x9x16 分类矩阵（90 个棋位，每位置 16 类）。目标部署平台为 iOS CoreML + ANE。

参考项目: `/Users/theone/theone/Programme/my-apps/cchess-apps/chinese-chess-recognition/cchess_reg`（基于 SwinV2 + MMPretrain，本项目的上一代实现）。

## Commands

```bash
# 环境管理（使用 uv，保证纯净）
uv sync                              # 安装依赖
uv run python src/model.py           # 验证模型构建 + 前向传播
uv run python src/inference.py       # Mock 推理演示（需 cd src/）
uv run python convert_coreml.py      # CoreML 转换（macOS only）
```

## Architecture

**数据流：** `[B,3,640,576]` → ConvNeXt Atto backbone → `[B,320,20,18]` → stride-2 conv → `[B,128,10,9]` → dilated context module → 3-layer classifier (128→64→32→16) → softmax → `[B,10,9,16]`

**关键设计约束（CoreML/ANE）：**
- 所有通道数必须是 32 的倍数
- 只允许 Conv2d, BatchNorm2d, ReLU/GELU, Add, Concat
- 禁止动态形状、Adaptive Pooling、reshape(-1,...)
- 固定输入形状: `[1, 3, 640, 576]`（Height=640, Width=576）

**16 类标签：** `.`, `x`, K, A, B, N, R, C, P, k, a, b, n, r, c, p

## File Layout

- `src/model.py` — CChessNet 模型定义（ContextModule + CChessNet）
- `src/inference.py` — Mock 推理演示
- `convert_coreml.py` — 独立的 CoreML 转换脚本（不依赖 inference.py）
- `gemini_doc.md` — 架构设计参考文档
- `datasets/demo/` — 示例图片 (.jpg) + 标签 (.txt，FEN 格式 10 行 x 9 列)

## Training Plan

后续将使用 **HuggingFace Trainer** 进行训练（非 MMPretrain 或自定义训练循环）。添加训练代码时注意：
- 使用 HuggingFace `transformers` / `datasets` 生态
- 模型需要兼容 `Trainer` API（实现 `forward` 返回 loss 或封装为 HF 模型类）
