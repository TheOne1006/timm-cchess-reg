"""预定义 Pipeline: train_transform, val_transform。

顺序参考旧版 cchess_reg/configs/datasets/multi_label_dataset.py。

关键设计：所有增强在原始棋盘坐标系下完成（CenterCrop 后 400×450），
最后再 Resize 到模型输入尺寸 576×640。原因：
- 原图 450×500 有 ~25px padding，4 个锚点定义棋盘边界
- CenterCrop 移除 padding 得到纯净棋盘，锚点与图像四角对齐
- PiecePaste 等基于 cell 位置的操作必须在正确坐标系下执行
- 最后 Resize 仅影响模型输入尺寸，不影响增强正确性
"""

import os
from typing import Optional

from .augment import ColorJitter, GaussianBlur, RandomErasing
from .base import CenterCrop, Compose, Resize, ToTensorNormalize, PILToNumpy, NumpyToPIL
from .copy_half import CChessCachedCopyHalf
from .flip import CChessHalfFlip, CChessRandomFlip
from .mixup import CChessMixSinglePngCls
from .perspective import RandomPerspective
from .randaugment import RandAugment

from ..dataset import IMG_HEIGHT, IMG_WIDTH, CROP_WIDTH, CROP_HEIGHT


def train_transform(
    png_dir: Optional[str] = None,
    perspective_prob: float = 0.5,
    piece_paste_prob: float = 0.3,
    piece_max_cells: int = 15,
) -> Compose:
    """训练集 transform pipeline。

    与旧版 cchess_reg 流程一致：
    1. CenterCrop → 移除 padding，得到纯净 400×450 棋盘
    2. PiecePaste → 在原始坐标系粘贴棋子（cell 位置精确）
    3. Resize → 放大到模型输入 576×640
    4. 其余增强 → 在 576×640 上执行
    """
    transforms_list = [
        # --- Step 1: 裁剪到纯净棋盘 (400×450) ---
        CenterCrop(),

        # --- Step 2: 棋子粘贴（在原始坐标系，cell 位置精确）---
        PILToNumpy(),
    ]

    if png_dir and os.path.exists(png_dir):
        transforms_list.append(CChessMixSinglePngCls(
            png_dir=png_dir,
            img_scale=(CROP_WIDTH, CROP_HEIGHT),
            max_mix_cells=piece_max_cells,
            prob=piece_paste_prob,
        ))

    # --- Step 3: Resize 到模型输入 (576×640) ---
    transforms_list.append(NumpyToPIL())
    transforms_list.append(Resize(height=IMG_HEIGHT, width=IMG_WIDTH))

    # --- Step 4: 空间增强 (576×640) ---
    transforms_list.append(PILToNumpy())
    transforms_list.append(CChessCachedCopyHalf(cache_size=100, prob=0.3))
    transforms_list.append(CChessRandomFlip(
        prob=[0.2, 0.2, 0.2],
        direction=["horizontal", "vertical", "diagonal"],
    ))
    transforms_list.append(CChessHalfFlip(mode="horizontal", prob=0.5))
    transforms_list.append(CChessHalfFlip(mode="vertical", prob=0.5))

    # --- Step 5: 颜色增强 ---
    transforms_list.append(NumpyToPIL())
    transforms_list.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.12))

    # --- Step 6: 其余增强 ---
    transforms_list.append(PILToNumpy())
    transforms_list.append(RandAugment(num_policies=3, magnitude_level=5, total_level=10))
    transforms_list.append(GaussianBlur(kernel_size=5, sigma=(0.1, 1.2), prob=0.3))
    transforms_list.append(RandomErasing(prob=0.5, min_area_ratio=0.0025, max_area_ratio=0.005))
    transforms_list.append(RandomErasing(prob=0.8, min_area_ratio=0.0025, max_area_ratio=0.005))
    transforms_list.append(RandomPerspective(
        scale=(0.02, 0.06),
        size_scale=(0.9, 1.1),
        prob=perspective_prob,
    ))

    transforms_list.append(ToTensorNormalize())

    return Compose(transforms_list)


def val_transform() -> Compose:
    """验证/测试集 transform pipeline。"""
    return Compose([
        CenterCrop(),
        Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        ToTensorNormalize(),
    ])
