"""通用图像增强: ColorJitter, GaussianBlur, RandomErasing。"""

import math
import random

import numpy as np
from PIL import Image
from torch import Tensor


class ColorJitter:
    """颜色抖动。"""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.12,
        prob: float = 1.0,
    ):
        self.prob = prob
        from torchvision.transforms import ColorJitter as _CJ
        self.jitter = _CJ(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Image.Image, Tensor]:
        if random.random() < self.prob:
            image = self.jitter(image)
        return image, label


class GaussianBlur:
    """高斯模糊。"""

    def __init__(
        self,
        kernel_size: int = 5,
        sigma: tuple[float, float] = (0.1, 1.2),
        prob: float = 0.3,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Image.Image, Tensor]:
        if random.random() < self.prob:
            from torchvision.transforms.functional import gaussian_blur
            sigma = random.uniform(*self.sigma)
            image = gaussian_blur(image, kernel_size=self.kernel_size, sigma=sigma)
        return image, label


class RandomErasing:
    """随机擦除图像区域。label 不变。

    使用随机噪声填充（与旧版 mode='rand' 一致）。
    """

    def __init__(
        self,
        prob: float = 0.5,
        min_area_ratio: float = 0.0025,
        max_area_ratio: float = 0.005,
    ):
        self.prob = prob
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Image.Image, Tensor]:
        if random.random() >= self.prob:
            return image, label

        img = np.array(image)
        h, w = img.shape[:2]
        area = h * w
        erase_area = area * random.uniform(self.min_area_ratio, self.max_area_ratio)
        erase_h = int(math.sqrt(erase_area))
        erase_w = int(erase_area / max(erase_h, 1))

        y = random.randint(0, max(h - erase_h, 0))
        x = random.randint(0, max(w - erase_w, 0))
        img[y:y + erase_h, x:x + erase_w] = np.random.randint(
            0, 256, (erase_h, erase_w, 3), dtype=np.uint8,
        )

        return Image.fromarray(img), label
