"""棋盘感知翻转: CChessRandomFlip, CChessHalfFlip。

参考: cchess_reg/datasets/transforms/cchess_random_flip.py
      cchess_reg/datasets/transforms/cchess_half_flip.py

关键设计:
- 翻转只改变棋子的空间位置，不改变棋子类别（K 永远是红王，k 永远是黑王）
- 水平翻转: 翻转列顺序
- 垂直翻转: 翻转行顺序
- 对角翻转: 水平 + 垂直（180° 旋转）
- 半棋盘镜像: 一侧镜像到另一侧（合成数据增强）
"""

import random

import numpy as np
import torch
from PIL import Image
from torch import Tensor

VALID_DIRECTIONS = {"horizontal", "vertical", "diagonal"}


class CChessRandomFlip:
    """棋盘随机翻转（至多执行一个方向，避免多个翻转互相抵消）。

    忠实复现旧版 CChessRandomFlip：翻转只改变棋子空间位置，不改变类别。
    翻转后 label 中的棋子类 index 不变——K 仍然是红王，k 仍然是黑王。

    执行逻辑：按顺序检查每个方向的概率，第一个命中的方向被执行后立即返回。
    这保证至多一个方向生效，避免 horizontal+vertical+diagonal 三者同时触发
    导致图像和标签都回到原始状态（浪费样本）。

    Args:
        prob: 每个方向的执行概率列表
        direction: 方向列表 ['horizontal', 'vertical', 'diagonal']
    """

    def __init__(
        self,
        prob: tuple[float, ...] = (0.2, 0.2, 0.2),
        direction: tuple[str, ...] = ("horizontal", "vertical", "diagonal"),
    ):
        assert len(prob) == len(direction)
        for d in direction:
            assert d in VALID_DIRECTIONS, f"Invalid direction: {d!r}, must be one of {VALID_DIRECTIONS}"
        self.prob = list(prob)
        self.direction = list(direction)

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Image.Image, Tensor]:
        for p, d in zip(self.prob, self.direction):
            if random.random() < p:
                image, label = self._flip(image, label, d)
                break  # 只执行第一个命中的方向，避免多个翻转互相抵消
        return image, label

    @staticmethod
    def _flip(image: Image.Image, label: Tensor, direction: str):
        if direction == "horizontal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.flip(1)
        elif direction == "vertical":
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.flip(0)
        elif direction == "diagonal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.flip(1)
            label = label.flip(0)
        return image, label


class CChessHalfFlip:
    """半棋盘镜像增强。

    忠实复现旧版 CChessHalfFlip：
    - 水平模式：以 cell_w * 4 或 cell_w * 5 为界，左右半边互镜像
    - 垂直模式：以 h // 2 为界，上下半边互镜像

    翻转只改变棋子空间位置，不改变类别。
    这是合成数据增强，目的是增加多样性。

    Args:
        mode: 'horizontal' 或 'vertical'
        prob: 执行概率
    """

    def __init__(self, mode: str = "horizontal", prob: float = 0.5):
        assert mode in ("horizontal", "vertical")
        self.mode = mode
        self.prob = prob

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Image.Image, Tensor]:
        if random.random() >= self.prob:
            return image, label

        img = np.array(image)
        h, w = img.shape[:2]
        label_np = label.numpy().copy()

        if self.mode == "horizontal":
            cell_w = w // 9
            if random.random() < 0.5:
                # 左半边镜像到右半边
                mid_w = cell_w * 4
                flip_half_img = np.fliplr(img[:, :mid_w])
                source_mid_w = w - flip_half_img.shape[1]
                img[:, source_mid_w:] = flip_half_img
                label_np[:, 4:] = np.fliplr(label_np[:, :5])
            else:
                # 右半边镜像到左半边
                mid_w = cell_w * 5
                flip_half_img = np.fliplr(img[:, mid_w:])
                source_mid_w = flip_half_img.shape[1]
                img[:, :source_mid_w] = flip_half_img
                label_np[:, :5] = np.fliplr(label_np[:, 4:])
        else:  # vertical
            mid_h = h // 2
            if random.random() < 0.5:
                # 上半边镜像到下半边
                flip_half_img = np.flipud(img[:mid_h, :])
                source_mid_h = h - flip_half_img.shape[0]
                img[source_mid_h:, :] = flip_half_img
                label_np[5:, :] = np.flipud(label_np[:5, :])
            else:
                # 下半边镜像到上半边
                flip_half_img = np.flipud(img[mid_h:, :])
                source_mid_h = flip_half_img.shape[0]
                img[:source_mid_h, :] = flip_half_img
                label_np[:5, :] = np.flipud(label_np[5:, :])

        image = Image.fromarray(img)
        label = torch.from_numpy(label_np)
        return image, label
