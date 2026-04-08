"""预定义 Pipeline: train_transform, val_transform。

顺序参考旧版 cchess_reg/configs/datasets/multi_label_dataset.py。
"""

import os
from typing import Optional

from .augment import ColorJitter, GaussianBlur, RandomErasing
from .base import CenterCrop, Compose, Resize, ToTensorNormalize
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

    顺序参考旧版:
    1.  Resize
    2.  CChessMixSinglePngCls  (棋子粘贴)
    3.  CChessCachedCopyHalf    (缓存式半板复制)
    4.  CChessHalfFlip          (水平半板镜像)
    5.  CChessHalfFlip          (垂直半板镜像)
    6.  CChessRandomFlip        (多方向随机翻转)
    7.  GaussianBlur
    8.  ColorJitter
    9.  RandomErasing           (×2)
    10. RandomPerspective
    11. ToTensorNormalize
    """
    transforms_list = [
        CenterCrop(),  # 裁掉原图 padding，与老项目 CenterCrop(400,450) 一致
        Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    ]

    # 棋子粘贴
    if png_dir and os.path.exists(png_dir):
        transforms_list.append(CChessMixSinglePngCls(
            png_dir=png_dir,
            img_scale=(IMG_WIDTH, IMG_HEIGHT),
            max_mix_cells=piece_max_cells,
            prob=piece_paste_prob,
        ))

    transforms_list.extend([
        # 缓存式半板复制
        CChessCachedCopyHalf(cache_size=100, prob=0.3),

        # 多方向随机翻转（至多执行一个方向，在半板镜像之前，避免被覆盖）
        CChessRandomFlip(
            prob=[0.2, 0.2, 0.2],
            direction=["horizontal", "vertical", "diagonal"],
        ),

        # 半板镜像（在随机翻转之后，作为独立的合成增强）
        CChessHalfFlip(mode="horizontal", prob=0.5),
        CChessHalfFlip(mode="vertical", prob=0.5),

        # 模糊
        GaussianBlur(kernel_size=5, sigma=(0.1, 1.2), prob=0.3),

        # 颜色抖动
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.12),

        # 随机擦除 (×2, 与旧版一致)
        RandomErasing(prob=0.5, min_area_ratio=0.0025, max_area_ratio=0.005),
        RandomErasing(prob=0.8, min_area_ratio=0.0025, max_area_ratio=0.005),

        # 透视变换（适度范围，避免网格位置与 label 严重偏移）
        RandomPerspective(
            scale=(0.05, 0.12),
            size_scale=(0.8, 1.2),
            prob=perspective_prob,
        ),

        # Tensor 化 + normalize
        ToTensorNormalize(),
    ])
    return Compose(transforms_list)


def val_transform() -> Compose:
    """验证/测试集 transform pipeline。"""
    return Compose([
        CenterCrop(),  # 裁掉原图 padding，与老项目 CenterCrop(400,450) 一致
        Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        ToTensorNormalize(),
    ])
