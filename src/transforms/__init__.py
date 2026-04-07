"""中国象棋棋盘感知的数据增强 Transform。"""

from .base import Compose, Resize, ToTensorNormalize, IMAGENET_MEAN, IMAGENET_STD
from .flip import CChessRandomFlip, CChessHalfFlip
from .copy_half import CChessCachedCopyHalf
from .perspective import RandomPerspective
from .mixup import CChessMixSinglePngCls
from .augment import ColorJitter, GaussianBlur, RandomErasing
from .pipeline import train_transform, val_transform

__all__ = [
    "Compose", "Resize", "ToTensorNormalize",
    "CChessRandomFlip", "CChessHalfFlip",
    "CChessCachedCopyHalf",
    "RandomPerspective",
    "CChessMixSinglePngCls",
    "ColorJitter", "GaussianBlur", "RandomErasing",
    "train_transform", "val_transform",
    "IMAGENET_MEAN", "IMAGENET_STD",
]
