# Backbone Upgrade + FPN Neck Design

## Summary

Upgrade model backbone from ConvNeXt Atto (3.4M) to ConvNeXt V2 Nano (15M) and add a multi-scale Feature Pyramid Network (FPN) neck that fuses features from two backbone stages. This improves representational capacity (~4.4x more parameters) while maintaining CoreML/ANE export compatibility and the fixed [B, 16, 10, 9] output shape.

## Motivation

The current ConvNeXt Atto backbone (3.4M params) is the primary accuracy bottleneck. Its small capacity limits feature expressiveness for distinguishing similar chess pieces (e.g., red vs black variants). The reference implementation uses SwinV2-Nano which is incompatible with our 640x576 input due to window attention constraints. ConvNeXt V2 Nano provides comparable accuracy with pure convolution operations that are fully compatible with our input resolution and CoreML/ANE.

## Architecture

### Before (Current)

```
[B, 3, 640, 576]
  -> ConvNeXt Atto (3.4M)
  -> [B, 320, 20, 18]
  -> stride-2 conv (320 -> 128)
  -> [B, 128, 10, 9]
  -> Channel Attention -> Context Module -> Classifier
  -> [B, 16, 10, 9]
```

### After (New)

```
[B, 3, 640, 576]
  -> ConvNeXt V2 Nano (15M, features_only, out_indices=(2, 3))
  +-- Stage 2: [B, 320, 40, 36]  (spatial detail)
  +-- Stage 3: [B, 640, 20, 18]  (semantic features)

FPN Neck:
  Path A (deep):    Stage 3 -> 1x1 Conv(640->128) -> stride-2 Conv -> [B, 128, 10, 9]
  Path B (shallow): Stage 2 -> 1x1 Conv(320->128) -> stride-4 Conv -> [B, 128, 10, 9]
  Merge: element-wise Add -> [B, 128, 10, 9]

  -> Channel Attention -> Context Module -> Classifier (unchanged)
  -> [B, 16, 10, 9]
```

### Parameter Budget

| Component | Current | New |
|-----------|---------|-----|
| Backbone | 3.4M | 15.0M |
| FPN Neck | 0.12M (downsample only) | ~0.25M (dual-path) |
| Channel Attention | ~0.01M | ~0.01M (unchanged) |
| Context Module | ~0.47M | ~0.47M (unchanged) |
| Classifier | ~0.03M | ~0.03M (unchanged) |
| **Total** | **~4.0M** | **~15.8M** |

## FPN Neck Implementation

```python
class FPNNeck(nn.Module):
    """Multi-scale Feature Pyramid Network neck.

    Fuses Stage 2 (spatial detail, 40x36) and Stage 3 (semantic, 20x18)
    features from the backbone into a unified 128-channel 10x9 representation.
    """

    def __init__(self, deep_channels=640, shallow_channels=320, mid_channels=128):
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

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        deep, shallow = features[1], features[0]  # stage3, stage2
        x_deep = self.downsample_deep(self.reduce_deep(deep))      # [B, 128, 10, 9]
        x_shallow = self.downsample_shallow(self.reduce_shallow(shallow))  # [B, 128, 10, 9]
        return x_deep + x_shallow  # element-wise add
```

### CoreML/ANE Compatibility

All operations in the FPN neck are ANE-compatible:
- Conv2d with standard kernel sizes (1x1, 3x3, 4x4) and static strides
- BatchNorm2d
- ReLU
- Element-wise Add

No dynamic shapes, no adaptive pooling, no reshape(-1,...). Channel counts are all multiples of 32 (128).

## Training Strategy

### Backbone Selection

| Option | Pretrained Data | Params | ImageNet Top-1 |
|--------|----------------|--------|---------------|
| `convnextv2_nano.fcmae_ft_in1k` | ImageNet-1k + FCMAE | 15M | ~82.1% |
| `convnext_nano.in12k_ft_in1k` | ImageNet-12k -> 1k | 15M | ~82.0% |

Try `convnext_nano.in12k_ft_in1k` first (richer pretraining from 12.8M images), fall back to `convnextv2_nano.fcmae_ft_in1k` if needed.

### Progressive Unfreezing

**Phase 1** (freeze backbone, ~15 epochs):
- Freeze all backbone parameters
- Train only FPN Neck + Channel Attention + Context Module + Classifier
- LR = 1e-3 (higher, lets new layers learn quickly)

**Phase 2** (fine-tune all, remaining epochs):
- Unfreeze backbone
- Train all parameters
- Discriminative LR via parameter groups:
  - Backbone: LR = 1e-5 (lower, preserves pretrained features)
  - New layers: LR = 1e-4 (standard)
- Cosine scheduler with 10-epoch warmup

### CLI Changes

New arguments in `train.py`:
- `--freeze_backbone_epochs N` (default: 15) — freeze backbone for N epochs
- `--backbone_lr_scale FLOAT` (default: 0.1) — backbone LR = base_lr * scale

## Files to Modify

| File | Change |
|------|--------|
| `src/model.py` | Add `FPNNeck` class. Modify `CChessNet.__init__` to accept `features_only=True` backbone. Replace `self.downsample` with `self.fpn_neck`. Update `forward` to handle multi-stage features. |
| `src/train.py` | Add backbone freezing logic (callback or manual). Add discriminative LR parameter groups. New CLI args: `--freeze_backbone_epochs`, `--backbone_lr_scale`. |
| `convert_coreml.py` | No changes needed (model external interface unchanged: input [1,3,640,576], output [1,10,9,16]). |

## Verification Plan

1. Run `python src/model.py` to verify model builds and forward pass produces correct shapes
2. Run `python convert_coreml.py` to verify CoreML export succeeds
3. Train on demo dataset for 5 epochs to verify training loop works with FPN neck
4. Compare parameter count with current model
