# MVP 设计：基于 ConvNeXt + timm 的中国象棋棋盘识别

**日期：** 2026-04-07
**状态：** 已批准
**范围：** MVP 演示 — 使用 mock 推理验证架构管线可行性

## 背景

参考项目 `cchess_reg` 使用 SwinV2 + MMPretrain 框架，CoreML/ANE 兼容性差。新方案（详见 `gemini_doc.md`）切换为纯 CNN 架构，针对 iOS CoreML 部署进行优化。

## 架构

### 数据流

```
输入 [1, 3, 640, 576]（高度=640，宽度=576）
  |
  v
ConvNeXt Atto 主干网络（timm: convnext_atto.d2_in1k，预训练，num_classes=0）
  |  forward_features()
  v
[1, 320, 20, 18]  （32倍下采样：640/32=20, 576/32=18）
  |
  v
Conv2d(320, 128, kernel=3, stride=2, padding=1)  -- 可学习的下采样
  |  BatchNorm2d(128) + ReLU
  v
[1, 128, 10, 9]  （20/2=10, 18/2=9） -- 完美匹配棋盘 10行 x 9列
  |
  v
上下文融合模块 ContextModule:
  分支A: Conv2d(128,128,k=3,p=1) + BN + ReLU  -- 局部上下文
  分支B: Conv2d(128,128,k=3,p=2,d=2) + BN + ReLU  -- 空洞卷积上下文
  输出 = 分支A + 分支B
  |
  v
[1, 128, 10, 9]
  |
  v
Conv2d(128, 64, kernel=1) + BN + ReLU   -- 128→64
  |
  v
Conv2d(64, 32, kernel=1) + BN + ReLU    -- 64→32
  |
  v
Conv2d(32, 16, kernel=1)                -- 32→16，分类头
  |
  v
[1, 16, 10, 9] -> permute -> [1, 10, 9, 16] -> softmax(dim=-1)
```

### 关键设计决策

1. **ConvNeXt Atto**：dims=(40, 80, 160, 320)，约 3.5M 参数量。足够小适合移动端部署，使用 ImageNet 预训练权重。
2. **通道数=128**（32 的整数倍）：满足 ANE 通道对齐要求，CoreML 部署时避免内存填充开销。
3. **步长卷积 (Stride-2 Conv2d)** 用于空间下采样：可学习，比池化更鲁棒，能更好地融合棋子字形特征。
4. **所有算子 CoreML 友好**：仅使用 Conv2d、BatchNorm2d、ReLU、Add。无动态形状，无自适应池化。
5. **16 个分类**：`.`、`x`、K、A、B、N、R、C、P、k、a、b、n、r、c、p（与参考项目一致）。

## 文件结构

```
timm-cchess-reg/
  datasets/demo/            # 已有的样本图片和标签
  src/
    __init__.py             # 空文件
    model.py                # CChessNet(nn.Module) - 约 60 行
    inference.py            # Mock 推理演示 - 约 30 行
  convert_coreml.py         # 独立的 CoreML 转换脚本 - 约 40 行
  pyproject.toml            # uv 项目配置
  gemini_doc.md             # 架构参考文档（已有）
```

## 组件详情

### model.py — CChessNet

一个 `nn.Module`，包含：

- **主干网络 (Backbone)**：`timm.create_model('convnext_atto.d2_in1k', pretrained=True, num_classes=0)`
  - 使用 `forward_features()` 获取空间特征图 `[B, 320, H/32, W/32]`
- **下采样层 (DownsampleLayer)**：`Conv2d(320, 128, 3, stride=2, padding=1)` + `BN` + `ReLU`
  - 将 `[B, 320, 20, 18]` 转换为 `[B, 128, 10, 9]`
- **上下文模块 (ContextModule)**：两个并联分支，输出相加
  - 分支A：标准 3x3 卷积（局部邻域信息）
  - 分支B：空洞 3x3 卷积，dilation=2（扩展感受野）
- **分类器 (Classifier)**：逐层通道降维的 1x1 卷积
  - `Conv2d(128, 64, 1)` + BN + ReLU
  - `Conv2d(64, 32, 1)` + BN + ReLU
  - `Conv2d(32, 16, 1)`（最终分类）
- **输出**：在类别维度上执行 softmax

### inference.py — Mock 演示

1. 实例化 `CChessNet`
2. 创建 mock 输入张量 `[1, 3, 640, 576]`（随机正态分布）
3. 执行前向推理
4. 打印输出 shape 和 argmax 棋盘布局（10x9 网格）

### convert_coreml.py — CoreML 转换

独立脚本，与主推理流程互不干扰：

1. 加载 `CChessNet` 并设置为 `eval()` 模式
2. 创建示例输入 `[1, 3, 640, 576]`
3. 使用 `torch.jit.trace()` 追踪模型
4. 通过 `coremltools.convert()` 转换，固定输入形状
5. 保存为 `.mlpackage`

## 环境配置

使用 `uv` 管理依赖，确保环境可复现：

```toml
[project]
name = "timm-cchess-reg"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.5",
    "timm>=1.0.26",
    "coremltools>=8.0",
    "Pillow>=10.0",
]
```

目标运行环境：Python 3.12，PyTorch (CPU)，timm 1.0.26+。

## MVP 范围外

- 真实图片加载和预处理
- 训练管线
- 数据增强
- 评估指标
- ONNX 导出（目标是 CoreML）
- 动态输入形状
