# Backbone Upgrade + FPN Neck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade model backbone from ConvNeXt Atto (3.4M) to ConvNeXt Nano (15M) with multi-scale FPN neck for improved accuracy.

**Architecture:** Replace single-stage backbone + stride-2 downsample with multi-stage backbone + dual-path FPN neck. Stage 2 features (40x36 spatial detail) and Stage 3 features (20x18 semantic) are each reduced to 128ch and downsampled to 10x9, then merged via element-wise addition. The rest of the model (Channel Attention, Context Module, Classifier) remains unchanged.

**Tech Stack:** Python 3.11+, PyTorch, timm 1.0.26, coremltools, HuggingFace Transformers (Trainer)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/model.py` | Modify | Add `FPNNeck` class (lines 44-87 insert). Modify `CChessNet.__init__` (lines 77-115) and `CChessNet.forward` (lines 117-138). Update default backbone and `__main__` block (lines 141-151). |
| `src/train.py` | Modify | Add `BackboneUnfreezeCallback` class. Override `create_optimizer` in `CChessTrainer` for discriminative LR. Add CLI args `--freeze_backbone_epochs`, `--backbone_lr_scale`. Update default `--backbone`. |
| `convert_coreml.py` | No changes | External interface unchanged: input [1,3,640,576], output [1,10,9,16]. |

---

### Task 1: Add FPNNeck class to model.py

**Files:**
- Modify: `src/model.py` — insert between line 43 (end of `ContextModule`) and line 46 (start of `ChannelAttention`)

- [ ] **Step 1: Add FPNNeck class**

Insert this class between `ContextModule` and `ChannelAttention` in `src/model.py` (after line 43):

```python
class FPNNeck(nn.Module):
    """多尺度特征金字塔 Neck，融合 Backbone 的 Stage 2 (空间细节) 和 Stage 3 (语义) 特征。

    Path A: 深层语义特征 (640ch, 20x18) -> 1x1 reduce -> stride-2 downsample -> 128ch, 10x9
    Path B: 浅层空间特征 (320ch, 40x36) -> 1x1 reduce -> stride-4 downsample -> 128ch, 10x9
    Merge: element-wise Add -> 128ch, 10x9
    """

    def __init__(self, deep_channels: int, shallow_channels: int, mid_channels: int = 128):
        super().__init__()
        # Path A: deep semantic features -> 20x18 -> 10x9
        self.reduce_deep = nn.Sequential(
            nn.Conv2d(deep_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.downsample_deep = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Path B: shallow spatial features -> 40x36 -> 10x9
        self.reduce_shallow = nn.Sequential(
            nn.Conv2d(shallow_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.downsample_shallow = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list) -> torch.Tensor:
        deep, shallow = features[1], features[0]  # stage3, stage2
        x_deep = self.downsample_deep(self.reduce_deep(deep))          # [B, 128, 10, 9]
        x_shallow = self.downsample_shallow(self.reduce_shallow(shallow))  # [B, 128, 10, 9]
        return x_deep + x_shallow
```

- [ ] **Step 2: Verify syntax by importing**

Run: `uv run python -c "from src.model import FPNNeck; print('FPNNeck imported OK')"`

Expected: `FPNNeck imported OK`

---

### Task 2: Modify CChessNet to use multi-stage backbone + FPNNeck

**Files:**
- Modify: `src/model.py` — `CChessNet.__init__` (lines 77-115), `CChessNet.forward` (lines 117-138), `__main__` block (lines 141-151)

- [ ] **Step 1: Update CChessNet.__init__ — change backbone creation and replace downsample with fpn_neck**

Replace the `CChessNet.__init__` method. The old code (lines 77-115):

```python
    def __init__(self, backbone_name: str = "convnext_atto.d2_in1k"):
        super().__init__()
        # Backbone：移除分类头，仅保留特征提取
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0
        )
        # backbone 输出通道数：从模型动态获取
        backbone_channels = self.backbone.num_features
        # 中间通道数：128（32 的倍数，满足 ANE 对齐要求）
        mid_channels = 128

        # 可学习的下采样：20x18 → 10x9
        self.downsample = nn.Sequential(
            nn.Conv2d(backbone_channels, mid_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 上下文融合
        self.register_buffer(
            "class_weight", torch.tensor(_CLASS_WEIGHT),
        )
        self.label_smoothing = _LABEL_SMOOTHING
        self.channel_attn = ChannelAttention(mid_channels)
        self.context = ContextModule(mid_channels)

        # 分类头：逐层通道变换 128→128→64→32→16
        self.classifier = nn.Sequential(
            nn.Conv2d(mid_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.NUM_CLASSES, 1),
        )
```

Replace with:

```python
    def __init__(self, backbone_name: str = "convnext_nano.in12k_ft_in1k"):
        super().__init__()
        # Backbone：多阶段特征提取 (Stage 2 + Stage 3)
        self.backbone = timm.create_model(
            backbone_name, pretrained=True,
            features_only=True, out_indices=(2, 3),
        )
        # 从 backbone feature_info 获取各阶段通道数
        ch = self.backbone.feature_info.channels()
        shallow_channels, deep_channels = ch[0], ch[1]
        # 中间通道数：128（32 的倍数，满足 ANE 对齐要求）
        mid_channels = 128

        # 多尺度 FPN Neck：融合 Stage 2 (空间细节) + Stage 3 (语义)
        self.fpn_neck = FPNNeck(deep_channels, shallow_channels, mid_channels)

        # 上下文融合
        self.register_buffer(
            "class_weight", torch.tensor(_CLASS_WEIGHT),
        )
        self.label_smoothing = _LABEL_SMOOTHING
        self.channel_attn = ChannelAttention(mid_channels)
        self.context = ContextModule(mid_channels)

        # 分类头：逐层通道变换 128→128→64→32→16
        self.classifier = nn.Sequential(
            nn.Conv2d(mid_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.NUM_CLASSES, 1),
        )
```

- [ ] **Step 2: Update CChessNet.forward — handle multi-stage features**

Replace the `CChessNet.forward` method. Old code (lines 117-138):

```python
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # x: [B, 3, 640, 576]
        features = self.backbone.forward_features(x)  # [B, 320, 20, 18]
        # backbone 输出可能非连续，MPS 后向需要 contiguous
        features = features.contiguous()
        x = self.downsample(features)  # [B, mid_channels, 10, 9]
        x = self.channel_attn(x)  # channel attention
        x = self.context(x)  # context module
        logits = self.classifier(x)  # [B, 16, 10, 9]

        if labels is not None:
            loss = F.cross_entropy(
                logits, labels, weight=self.class_weight,
                label_smoothing=self.label_smoothing,
            )
            # detach logits for evaluation metrics (no grad needed)
            logits_out = logits.detach().permute(0, 2, 3, 1).contiguous()
            return {"loss": loss, "logits": logits_out}

        # 推理模式：返回 softmax 概率
        logits = logits.permute(0, 2, 3, 1)  # [B, 10, 9, 16]
        return torch.softmax(logits, dim=-1)
```

Replace with:

```python
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # x: [B, 3, 640, 576]
        features = self.backbone(x)  # list: [stage2, stage3]
        x = self.fpn_neck(features)  # [B, 128, 10, 9]
        x = self.channel_attn(x)  # channel attention
        x = self.context(x)  # context module
        logits = self.classifier(x)  # [B, 16, 10, 9]

        if labels is not None:
            loss = F.cross_entropy(
                logits, labels, weight=self.class_weight,
                label_smoothing=self.label_smoothing,
            )
            # detach logits for evaluation metrics (no grad needed)
            logits_out = logits.detach().permute(0, 2, 3, 1).contiguous()
            return {"loss": loss, "logits": logits_out}

        # 推理模式：返回 softmax 概率
        logits = logits.permute(0, 2, 3, 1)  # [B, 10, 9, 16]
        return torch.softmax(logits, dim=-1)
```

- [ ] **Step 3: Verify model builds and forward pass produces correct shapes**

Run: `uv run python src/model.py`

Expected output:
```
Input shape:  torch.Size([1, 3, 640, 576])
Output shape: torch.Size([1, 10, 9, 16])
Output sum per position (should be ~1.0): 1.0000
模型验证通过！
```

Note: First run will download ~58MB pretrained weights from timm.

- [ ] **Step 4: Verify parameter count**

Run: `uv run python -c "
from src.model import CChessNet
m = CChessNet()
total = sum(p.numel() for p in m.parameters()) / 1e6
backbone = sum(p.numel() for p in m.backbone.parameters()) / 1e6
fpn = sum(p.numel() for p in m.fpn_neck.parameters()) / 1e6
rest = total - backbone - fpn
print(f'Total: {total:.2f}M')
print(f'  Backbone: {backbone:.2f}M')
print(f'  FPN Neck: {fpn:.2f}M')
print(f'  Rest (attn+context+classifier): {rest:.2f}M')
"`

Expected: Total ~15.8M, Backbone ~15.0M, FPN Neck ~0.25M, Rest ~0.5M

- [ ] **Step 5: Commit**

```bash
git add src/model.py
git commit -m "feat(model): upgrade to ConvNeXt Nano backbone with FPN neck

- Add FPNNeck class: dual-path multi-scale feature fusion
- Stage 2 (spatial detail) + Stage 3 (semantic) merged at 10x9
- Backbone: ConvNeXt Atto (3.4M) -> ConvNeXt Nano in12k (15M)
- Total params: ~15.8M (up from ~4.0M)
- CoreML export interface unchanged"
```

---

### Task 3: Verify CoreML export

**Files:**
- No file changes — just verification
- Uses: `convert_coreml.py`

- [ ] **Step 1: Run CoreML conversion**

Run: `uv run python convert_coreml.py`

Expected:
```
=== CoreML 转换 ===

加载 CChessNet...
追踪模型 (torch.jit.trace)...
转换为 CoreML...
保存到 ...CChessNet.mlpackage...
转换完成！
```

- [ ] **Step 2: Verify the exported model produces correct output**

Run: `uv run python -c "
import coremltools as ct
import numpy as np
mlmodel = ct.models.MLModel('CChessNet.mlpackage')
spec = mlmodel.get_spec()
inp = spec.description.input[0]
out = spec.description.output[0]
print(f'Input: {inp.name} shape={inp.type.multiArrayType.shape}')
print(f'Output: {out.name} shape={out.type.multiArrayType.shape}')

# Run prediction
dummy = np.random.rand(1, 3, 640, 576).astype(np.float32)
result = mlmodel.predict({'image': dummy})
pred = result['board_prediction']
print(f'Prediction shape: {pred.shape}')
print(f'Sum at position [0,0,0]: {pred[0,0,0].sum():.4f} (should be ~1.0)')
print('CoreML export verified OK')
"`

Expected: Output shape (1, 10, 9, 16), sum at each position ~1.0

---

### Task 4: Add progressive unfreezing and discriminative LR to train.py

**Files:**
- Modify: `src/train.py` — add callback class (insert before `train()`), modify `CChessTrainer`, modify `main()` argument parser and `train()` function

- [ ] **Step 1: Add BackboneUnfreezeCallback class**

First, update the existing import on line 63 from:

```python
from transformers import Trainer
```

to:

```python
from transformers import Trainer, TrainerCallback
```

Then insert the callback class between the `HFModelWrapper` class (ending at line 60) and the `from transformers import Trainer` line (line 63). Place it after line 60:

```python
class BackboneUnfreezeCallback(TrainerCallback):
    """在指定 step 解冻 backbone 参数。"""

    def __init__(self, unfreeze_step: int):
        self.unfreeze_step = unfreeze_step
        self._unfrozen = False

    def on_step_end(self, args, state, control, **kwargs):
        if self._unfrozen or state.global_step < self.unfreeze_step:
            return
        model = kwargs.get("model")
        if model is None:
            return
        cchess = model.cchess if hasattr(model, "cchess") else model
        for p in cchess.backbone.parameters():
            p.requires_grad = True
        self._unfrozen = True
        print(f"\n[BackboneUnfreeze] Unfreezing backbone at step {state.global_step}")
```

- [ ] **Step 2: Override create_optimizer in CChessTrainer for discriminative LR**

Modify the `CChessTrainer` class. Add `__init__` and `create_optimizer` methods:

Old `CChessTrainer` (lines 66-97):

```python
class CChessTrainer(Trainer):
    """自定义 Trainer：直接控制 DataLoader 创建，保留 collate_fn 和 drop_last。"""

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler
        nw = self.args.dataloader_num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=RandomSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=nw,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,
            prefetch_factor=2 if nw > 0 else None,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        from torch.utils.data import DataLoader, SequentialSampler
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        nw = self.args.dataloader_num_workers
        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=self.data_collator,
            num_workers=nw,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,
            prefetch_factor=2 if nw > 0 else None,
        )
```

Replace with:

```python
class CChessTrainer(Trainer):
    """自定义 Trainer：直接控制 DataLoader 创建，保留 collate_fn 和 drop_last，支持 discriminative LR。"""

    def __init__(self, *args, backbone_lr_scale: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone_lr_scale = backbone_lr_scale

    def create_optimizer(self):
        """使用 discriminative LR：backbone 用较低的 LR，新层用标准 LR。"""
        import torch
        cchess = self.model.cchess if hasattr(self.model, "cchess") else self.model
        backbone_params = [p for n, p in cchess.named_parameters() if "backbone" in n]
        new_params = [p for n, p in cchess.named_parameters() if "backbone" not in n]
        optimizer_groups = [
            {"params": backbone_params, "lr": self.args.learning_rate * self.backbone_lr_scale},
            {"params": new_params, "lr": self.args.learning_rate},
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=self.args.weight_decay,
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler
        nw = self.args.dataloader_num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=RandomSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=nw,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,
            prefetch_factor=2 if nw > 0 else None,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        from torch.utils.data import DataLoader, SequentialSampler
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        nw = self.args.dataloader_num_workers
        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=self.data_collator,
            num_workers=nw,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,
            prefetch_factor=2 if nw > 0 else None,
        )
```

- [ ] **Step 3: Add freeze logic and callbacks in train() function**

In the `train()` function, after creating the model (around line 131) and before creating the trainer, add the freeze and callback setup.

After the line `print(f"模型参数量: {param_count:.2f}M")` (line 132), insert:

```python
    # Freeze backbone if progressive unfreezing is enabled
    callbacks = []
    if args.freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        unfreeze_step = args.freeze_backbone_epochs * steps_per_epoch
        callbacks.append(BackboneUnfreezeCallback(unfreeze_step))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"Backbone frozen for {args.freeze_backbone_epochs} epochs ({unfreeze_step} steps)")
        print(f"Trainable params: {trainable:.2f}M (backbone excluded)")
```

Update the `CChessTrainer` instantiation to pass the new parameters:

Old (around line 171):
```python
    trainer = CChessTrainer(
        model=hf_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_cchess_metrics,
    )
```

Replace with:
```python
    trainer = CChessTrainer(
        model=hf_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_cchess_metrics,
        callbacks=callbacks if callbacks else None,
        backbone_lr_scale=args.backbone_lr_scale,
    )
```

- [ ] **Step 4: Update CLI arguments in main()**

In `main()`, update the default backbone and add new arguments.

Change the existing `--backbone` default (around line 197):
```python
    parser.add_argument("--backbone", type=str, default="convnext_nano.in12k_ft_in1k", help="timm backbone 名称")
```

Add after `--scheduler` argument (around line 207):
```python
    parser.add_argument("--freeze_backbone_epochs", type=int, default=15, help="冻结 backbone 的 epoch 数 (0=不冻结)")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1, help="backbone LR = base_lr * scale")
```

- [ ] **Step 5: Verify train.py syntax**

Run: `uv run python -c "from src.train import CChessTrainer, BackboneUnfreezeCallback; print('train.py imports OK')"`

Expected: `train.py imports OK`

- [ ] **Step 6: Commit**

```bash
git add src/train.py
git commit -m "feat(train): add progressive unfreezing and discriminative LR

- BackboneUnfreezeCallback: unfreeze backbone after N epochs
- CChessTrainer.create_optimizer: backbone LR scaled by backbone_lr_scale
- New CLI args: --freeze_backbone_epochs (default 15), --backbone_lr_scale (default 0.1)
- Default backbone updated to convnext_nano.in12k_ft_in1k"
```

---

### Task 5: Smoke test training pipeline

**Files:**
- No file changes — verification only
- Uses: `src/train.py`, `datasets/demo/`

- [ ] **Step 1: Verify training starts and runs for 2 epochs**

Run: `uv run python -m src.train --data_dir datasets/demo --epochs 2 --batch_size 2 --freeze_backbone_epochs 1 --num_workers 0 --log_interval 1 --report_to none`

Expected:
- Model loads with ~15.8M params
- Prints "Backbone frozen for 1 epochs"
- Prints "Trainable params: ~0.7M (backbone excluded)"
- Loss decreases over steps
- After epoch 1: "[BackboneUnfreeze] Unfreezing backbone at step N"
- Training completes epoch 2

- [ ] **Step 2: Verify full training (without freezing) also works**

Run: `uv run python -m src.train --data_dir datasets/demo --epochs 1 --batch_size 2 --freeze_backbone_epochs 0 --num_workers 0 --log_interval 1 --report_to none`

Expected: Training runs with all parameters unfrozen, loss decreases.

- [ ] **Step 3: Commit if any fixes were needed**

Only commit if you had to fix any issues discovered during testing.
