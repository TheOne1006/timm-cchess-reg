"""缓存式半棋盘复制: CChessCachedCopyHalf。

参考: cchess_reg/datasets/transforms/cchess_cache_copy_half.py

逻辑:
- 维护一个样本缓存队列
- 每次先将当前样本存入缓存
- 从缓存中随机取一个样本，复制其上半或下半部分到当前图像
- 上半 = 前 45 个位置 (row 0-4)，下半 = 后 45 个位置 (row 5-9)
"""

import random

import numpy as np
import torch
from PIL import Image
from torch import Tensor

# 棋盘中线位置：10行×9列=90格，上半=前45格(row 0-4)，下半=后45格(row 5-9)
HALF_BOARD_SIZE = 10 * 9 // 2  # = 45


class CChessCachedCopyHalf:
    """缓存式半棋盘复制增强。

    Args:
        cache_size: 缓存队列最大长度
        prob: 执行概率
    """

    def __init__(self, cache_size: int = 100, prob: float = 0.3):
        self.cache_size = cache_size
        self.prob = prob
        self._cache: list[dict] = []

    def __call__(self, image, label: Tensor):
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image.copy()

        label_np = label.numpy().copy()

        # 当前样本存入缓存（使用原始数据的拷贝）
        cache_entry = {
            "img": img_np.copy(),
            "label": label_np.copy(),
        }

        # 尝试从缓存中取样本并复制半边
        if self._cache and random.random() < self.prob:
            cache_item = random.choice(self._cache)
            h = img_np.shape[0]
            half_h = h // 2

            if random.random() < 0.5:
                # 复制上半部分
                img_np[:half_h, :, :] = cache_item["img"][:half_h, :, :]
                label_flat = label_np.flatten()
                label_flat[:HALF_BOARD_SIZE] = cache_item["label"].flatten()[:HALF_BOARD_SIZE]
                label_np = label_flat.reshape(10, 9)
            else:
                # 复制下半部分
                img_np[half_h:, :, :] = cache_item["img"][half_h:, :, :]
                label_flat = label_np.flatten()
                label_flat[HALF_BOARD_SIZE:] = cache_item["label"].flatten()[HALF_BOARD_SIZE:]
                label_np = label_flat.reshape(10, 9)

            label = torch.from_numpy(label_np)

        # 维护缓存
        if len(self._cache) >= self.cache_size:
            self._cache.pop(random.randint(0, len(self._cache) - 1))
        self._cache.append(cache_entry)

        return img_np, label
