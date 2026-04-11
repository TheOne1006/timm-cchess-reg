from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .dataset import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH

# 空位权重降低，避免空位主导 loss（对齐旧版 SwinV2 方案）
_CLASS_WEIGHT = [0.35] + [1.0] * (NUM_CLASSES - 1)
_LABEL_SMOOTHING = 0.08


class ContextModule(nn.Module):
    """三分支空洞卷积上下文融合模块。

    dilation=1 局部邻域 + dilation=2 近场上下文 + dilation=4 远场上下文。
    """

    def __init__(self, channels: int):
        super().__init__()
        # 分支A：标准 3x3 卷积，局部邻域
        self.branch_a = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # 分支B：空洞 3x3 卷积，dilation=2，近场上下文
        self.branch_b = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # 分支C：空洞 3x3 卷积，dilation=4，远场上下文
        self.branch_c = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch_a(x) + self.branch_b(x) + self.branch_c(x)


class ChannelAttention(nn.Module):
    """ANE 兼容的通道注意力模块（SE-style）。

    使用 mean reduction + 1x1 conv 实现，通道数保持 32 的倍数。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 32)
        mid = (mid + 31) // 32 * 32  # Round up to 32x for ANE
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=[2, 3], keepdim=True)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class CChessNet(nn.Module):
    """全卷积网格预测模型：输入棋盘图像，输出 10x9x16 分类矩阵。

    管线：ConvNeXt Atto → stride-2 conv → channel attention → context module → 1x1 classifier
    """

    # 常量从 dataset.py 导入，保持单一数据源
    NUM_CLASSES = NUM_CLASSES
    INPUT_HEIGHT = IMG_HEIGHT
    INPUT_WIDTH = IMG_WIDTH

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


if __name__ == "__main__":
    # 快速验证模型构建和前向传播
    model = CChessNet()
    model.eval()
    dummy = torch.randn(1, 3, CChessNet.INPUT_HEIGHT, CChessNet.INPUT_WIDTH)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output sum per position (should be ~1.0): {out[0, 0, 0].sum().item():.4f}")
    print("模型验证通过！")
