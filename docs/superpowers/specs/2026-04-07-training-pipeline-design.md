# Training Pipeline Design: Preprocessing, Training, and Evaluation

Date: 2026-04-07

## Overview

为 CChessNet（ConvNeXt Atto 全卷积网格预测模型）构建完整的训练管线，包括棋类专用数据增强、HuggingFace Trainer 训练、和自定义评估指标。目标平台：Google Colab（设备通用：CPU/CUDA/MPS）。

迁移自旧项目 `/Users/theone/theone/Programme/my-apps/cchess-apps/chinese-chess-recognition/cchess_reg`，从 MMPretrain 生态迁移到 HuggingFace Trainer 生态。

## Architecture

### 数据流

```
原始数据                              训练时
FEN .txt ──→ parse_fen() ──→ [10,9] int tensor
JPG image ──→ PIL ──→ transform pipeline ──→ [3,640,576] float tensor
                                                        ↓
                                              CChessDataset.__getitem__
                                              returns {pixel_values, labels}
                                                        ↓
                                              CChessNetForTraining.forward
                                              loss = CE(pred, label) over 90 positions
                                              logits = [B,10,9,16]
```

### 文件结构

```
src/
├── model.py              # (已有) CChessNet 模型
├── training/
│   ├── __init__.py
│   ├── dataset.py        # CChessDataset + FEN 解析
│   ├── transforms.py     # 所有棋类专用 transform
│   ├── model_wrapper.py  # CChessNetForTraining（compute_loss）
│   ├── metrics.py        # 5 种评估指标
│   └── train.py          # 训练入口（HF Trainer 配置）
```

不创建 HF PreTrainedModel 包装（过度设计）。不用 AutoImageProcessor（transform 是棋类专用的）。

## Components

### 1. CChessDataset (dataset.py)

加载图片 + FEN 标签文件。

**FEN 解析：**

```python
FEN_CHAR_TO_CLASS = {
    '.': 0, 'x': 1,
    'K': 2, 'A': 3, 'B': 4, 'N': 5, 'R': 6, 'C': 7, 'P': 8,
    'k': 9, 'a': 10, 'b': 11, 'n': 12, 'r': 13, 'c': 14, 'p': 15,
}

def parse_fen(lines: list[str]) -> np.ndarray:
    """10 行 FEN 字符串 → [10, 9] int array"""
    board = []
    for line in lines[:10]:
        row = [FEN_CHAR_TO_CLASS[ch] for ch in line.strip()]
        board.append(row)
    return np.array(board, dtype=np.int64)  # [10, 9]
```

**Dataset 接口：**

```python
class CChessDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        # 扫描 .jpg 文件，对应同目录 .txt 标签

    def __getitem__(self, idx):
        # 1. PIL 加载图片
        # 2. 读取 .txt → parse_fen() → np.array [10,9]
        # 3. 应用 transform pipeline → (image_tensor, labels)
        return {"pixel_values": image_tensor, "labels": labels_tensor}
```

**train/val 划分：** 101 张 demo 数据按 80/20 随机划分（约 80 train / 21 val），固定 random seed 确保可复现。

### 2. Transforms (transforms.py)

所有 transform 统一接口：`__call__(image: PIL.Image, labels: np.ndarray[10,9]) -> (PIL.Image, np.ndarray[10,9])`

**Test Pipeline：**

| 步骤 | Transform | 说明 |
|------|-----------|------|
| 1 | LoadImage + Resize(640,576) | 统一尺寸 |
| 2 | ToTensor + Normalize | ImageNet 标准化 |

**Train Pipeline（按顺序）：**

| 步骤 | Transform | 概率 | 说明 |
|------|-----------|------|------|
| 1 | Resize(640,576) | 1.0 | 统一尺寸 |
| 2 | CChessMixSinglePngCls | 0.7 | 在空位粘贴棋子 PNG（需在 ToTensor 前操作像素） |
| 3 | CChessRandomFlip | 各0.2 | 全棋盘翻转（水平/垂直/对角），label 同步 reshape(10,9) 翻转 |
| 4 | CChessHalfFlip (horizontal) | 0.5 | 左右半棋盘镜像 |
| 5 | CChessHalfFlip (vertical) | 0.5 | 上下半棋盘镜像 |
| 6 | CChessCachedCopyHalf | 0.3 | 从缓存复制半棋盘 |
| 7 | RandomPerspective | 0.7 | 透视变换，size_scale=(0.7,1.3) |
| 8 | ColorJitter | 1.0 | b=0.2, c=0.2, s=0.2, h=0.12 |
| 9 | GaussianBlur | 0.3 | σ∈(0.1,1.2) |
| 10 | RandomErasing | 0.5 | 小面积遮挡 |
| 11 | RandomErasing | 0.8 | 二次遮挡 |
| 12 | ToTensor + Normalize | 1.0 | ImageNet 标准化 |

**关键决策：**
- CChessMixSinglePngCls 在 ToTensor 前执行（需要 PIL Image 操作），`img_scale` 调整为 (640,576)
- 不用 CenterCrop — 直接 Resize 到目标尺寸
- 不用 RandAugment — 用可控的 ColorJitter + GaussianBlur 替代
- 翻转操作必须同步变换 label 的 10×9 网格

**各 Transform 实现要点：**

- **CChessRandomFlip**: label reshape 到 [10,9]，按方向翻转行/列，再 reshape 回来。三种方向：水平（列翻转）、垂直（行翻转）、对角（两者同时）
- **CChessHalfFlip**: 水平模式取列 0-3 镜像到列 5-8（或反向）；垂直模式取行 0-4 镜像到行 5-9（或反向）
- **CChessCachedCopyHalf**: 维护一个 FIFO 缓存（默认 100 个），随机选择上半/下半，把缓存样本对应半棋盘的 image 区域和 label 都复制过来
- **CChessMixSinglePngCls**: 从 `datasets/single_cls2_png/` 读取 14 类棋子 PNG，只粘贴到 label==0（空位）的位置，支持缩放(1.0-1.5)、旋转(-180°~180°)
- **RandomPerspective**: 随机扰动四角坐标做透视变换

### 3. CChessNetForTraining (model_wrapper.py)

```python
class CChessNetForTraining(nn.Module):
    def __init__(self, model: CChessNet):
        self.model = model

    def forward(self, pixel_values, labels=None):
        logits = self.model(pixel_values)  # [B, 10, 9, 16]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, 16),    # [B*90, 16]
                labels.view(-1),         # [B*90]
            )
        return {"loss": loss, "logits": logits}
```

Loss 计算：90 个位置的 cross-entropy 取均值。

### 4. Evaluation Metrics (metrics.py)

**5 种指标：**

| 指标 | 计算逻辑 | 输出 |
|------|----------|------|
| **per_class_ap** | 16 个类各自的 AP（90 个位置展平后按类计算） | 16 个值 |
| **per_position_ap** | 90 个位置各自的 AP | 90 个值 |
| **mAP_macro** | 16 类 AP 的均值 | 1 个值 |
| **board_accuracy** | 完全正确率 + 容差准确率 (errK=1,3,5) | 4 个值 |
| **filtered_mAP** | 排除空位和 other 后重新计算 mAP | 1 个值 |

**compute_metrics 签名：**

```python
def compute_metrics(eval_pred: EvalPrediction) -> dict:
    probs = softmax(predictions, axis=-1)   # [N, 10, 9, 16]
    preds = predictions.argmax(axis=-1)      # [N, 10, 9]
    labels = label_ids                       # [N, 10, 9]

    return {
        "mAP": ...,
        "board_accuracy": ...,
        "board_accuracy_err1": ...,
        "board_accuracy_err3": ...,
        "board_accuracy_err5": ...,
        "filtered_mAP": ...,
        # per_class_ap 和 per_position_ap 记日志
    }
```

**AP 计算：** 分类 AP，非目标检测 AP。将 `[N,10,9,16]` 展平为 `[N*90, 16]`，每个位置视为独立分类样本。对 class c：二值化，用 softmax 概率作置信度，计算 precision-recall 曲线下面积。

**board_accuracy 计算：**

```python
errors_per_board = (preds != labels).sum(axis=(1,2))  # [N]
board_acc      = (errors_per_board == 0).mean()
board_acc_err1 = (errors_per_board <= 1).mean()
board_acc_err3 = (errors_per_board <= 3).mean()
board_acc_err5 = (errors_per_board <= 5).mean()
```

Best model 指标：`board_accuracy_err1`（允许 1 个错误位置的整棋盘准确率）。

### 5. Training Entry (train.py)

```python
TrainingArguments(
    output_dir="./output",
    num_train_epochs=50,
    per_device_train_batch_size=8,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,              # Colab GPU
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="board_accuracy_err1",
)
```

设备通用：fp16 在 CUDA 上启用，CPU/MPS 上自动降级。

## Constraints

- 模型输入固定 `[1, 3, 640, 576]`，transform 输出必须匹配
- 所有通道数 32 的倍数（已有模型保证，transform 不影响）
- 设备通用：代码需在 CPU / CUDA / MPS 上运行
- Colab 友好：依赖用 pip install 可装

### 6. HuggingFace Hub 上传

训练完成后支持上传模型到 HF Hub。由于不是标准 `PreTrainedModel`，使用手动上传方案：

```python
# 训练完成后保存最佳模型
trainer.save_model("./best_model")

# 上传到 HF Hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./best_model",
    repo_id="your-username/cchess-reg",
    repo_type="model",
)
```

`TrainingArguments` 可配置 `push_to_hub=True`，但需配合自定义 `save_pretrained` 方法。

## Out of Scope

- 分布式训练
- Hyperparameter search
- RandAugment（用 ColorJitter + GaussianBlur 替代）
- CenterCrop（直接 Resize）

## Dependencies

已有：`torch`, `timm`, `pillow`, `coremltools`

新增：
- `transformers` (HF Trainer)
- `scikit-learn` (AP 计算)
- `scipy` (如有需要)
- `opencv-python` (透视变换)
