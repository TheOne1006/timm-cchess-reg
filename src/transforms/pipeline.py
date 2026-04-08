"""预定义 Pipeline: train_transform, val_transform。

顺序参考旧版 cchess_reg/configs/datasets/multi_label_dataset.py。
"""

import os
from typing import Optional

from .augment import ColorJitter, GaussianBlur, RandomErasing
from .base import CenterCrop, Compose, Resize, ToTensorNormalize, PILToNumpy, NumpyToPIL
from .copy_half import CChessCachedCopyHalf
from .flip import CChessHalfFlip, CChessRandomFlip
from .mixup import CChessMixSinglePngCls
from .perspective import RandomPerspective

from ..dataset import IMG_HEIGHT, IMG_WIDTH


def train_transform(
    png_dir: Optional[str] = None,
    perspective_prob: float = 0.7,
    piece_paste_prob: float = 0.3,
    piece_max_cells: int = 15,
) -> Compose:
    """训练集 transform pipeline。

    顺序与旧版完全一致。在 PIL/numpy 类型边界插入显式转换节点，
    避免每个 transform 内部重复 PIL↔numpy 转换。
    """
    transforms_list = [
        # --- PIL block: 空间变换 ---
        CenterCrop(),
        Resize(height=IMG_HEIGHT, width=IMG_WIDTH),

        # --- BARRIER: PIL → numpy ---
        PILToNumpy(),
    ]

    # --- numpy block 1: 棋子粘贴 + 半板复制 ---
    if png_dir and os.path.exists(png_dir):
        transforms_list.append(CChessMixSinglePngCls(
            png_dir=png_dir,
            img_scale=(IMG_WIDTH, IMG_HEIGHT),
            max_mix_cells=piece_max_cells,
            prob=piece_paste_prob,
        ))

    transforms_list.append(CChessCachedCopyHalf(cache_size=100, prob=0.3))

    # --- BARRIER: numpy → PIL (RandomFlip 使用 Image.transpose) ---
    transforms_list.append(NumpyToPIL())

    # --- PIL block: 随机翻转 ---
    transforms_list.append(CChessRandomFlip(
        prob=[0.2, 0.2, 0.2],
        direction=["horizontal", "vertical", "diagonal"],
    ))

    # --- BARRIER: PIL → numpy ---
    transforms_list.append(PILToNumpy())

    # --- numpy block 2: 半板镜像 ---
    transforms_list.append(CChessHalfFlip(mode="horizontal", prob=0.5))
    transforms_list.append(CChessHalfFlip(mode="vertical", prob=0.5))

    # --- BARRIER: numpy → PIL (GaussianBlur + ColorJitter 使用 torchvision PIL API) ---
    transforms_list.append(NumpyToPIL())

    # --- PIL block: 模糊 + 颜色 ---
    transforms_list.append(GaussianBlur(kernel_size=5, sigma=(0.1, 1.2), prob=0.3))
    transforms_list.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.12))

    # --- BARRIER: PIL → numpy ---
    transforms_list.append(PILToNumpy())

    # --- numpy block 3: 擦除 + 透视 ---
    transforms_list.append(RandomErasing(prob=0.5, min_area_ratio=0.0025, max_area_ratio=0.005))
    transforms_list.append(RandomErasing(prob=0.8, min_area_ratio=0.0025, max_area_ratio=0.005))
    transforms_list.append(RandomPerspective(
        scale=(0.05, 0.12),
        size_scale=(0.8, 1.2),
        prob=perspective_prob,
    ))

    # --- BARRIER: numpy → PIL (ToTensorNormalize 需要 PIL) ---
    transforms_list.append(NumpyToPIL())
    transforms_list.append(ToTensorNormalize())

    return Compose(transforms_list)


def val_transform() -> Compose:
    """验证/测试集 transform pipeline。"""
    return Compose([
        CenterCrop(),  # 裁掉原图 padding，与老项目 CenterCrop(400,450) 一致
        Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        ToTensorNormalize(),
    ])
