"""RandAugment: 随机自动增强策略。

对齐旧版 cchess_reg 的 RandAugment 配置：
12 种策略, num_policies=3, magnitude_level=5, total_level=10。
所有操作基于 numpy/cv2 实现，适配当前 pipeline 的 numpy block。
"""

import random
from typing import Sequence

import cv2
import numpy as np
from torch import Tensor


# --- 各个增强策略的实现 ---

def _auto_contrast(img: np.ndarray, _magnitude: float) -> np.ndarray:
    """AutoContrast: 将像素值拉伸到全范围。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(gray, [1, 99])
    if hi <= lo:
        return img
    scale = 255.0 / (hi - lo)
    table = np.clip(((np.arange(256) - lo) * scale).astype(int), 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


def _equalize(img: np.ndarray, _magnitude: float) -> np.ndarray:
    """Equalize: 直方图均衡化。"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _posterize(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Posterize: 减少色阶数。magnitude 映射到 bits=[8,4]。"""
    bits = max(1, int(8 - magnitude * 4))
    divisor = 256 // (2 ** bits)
    result = (img // divisor) * divisor
    return result


def _solarize_add(img: np.ndarray, magnitude: float) -> np.ndarray:
    """SolarizeAdd: 给低于阈值的像素加上一个值。magnitude 映射到 add=[0,110]。"""
    add_val = int(magnitude * 110)
    result = img.astype(np.int16)
    result = np.where(result < 128, result + add_val, result)
    return np.clip(result, 0, 255).astype(np.uint8)


def _adjust_color(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Color: 调整饱和度。magnitude 映射到 factor=[0.5, 1.5]。"""
    factor = 0.5 + magnitude * 1.0  # [0.5, 1.5]
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls[:, :, 2] = np.clip(hls[:, :, 2].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def _adjust_contrast(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Contrast: 调整对比度。magnitude 映射到 factor=[0.5, 1.5]。"""
    factor = 0.5 + magnitude * 1.0
    mean = img.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    result = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return result


def _adjust_brightness(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Brightness: 调整亮度。magnitude 映射到 delta=[-63, 63]。"""
    delta = (magnitude - 0.5) * 126  # [-63, 63]
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def _adjust_sharpness(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Sharpness: 调整锐度。magnitude 映射到 factor=[0, 2]。"""
    factor = magnitude * 2.0
    if factor <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    result = np.clip(
        img.astype(np.float32) * (1 + factor) - blurred.astype(np.float32) * factor,
        0, 255,
    ).astype(np.uint8)
    return result


def _rotate(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Rotate: 随机旋转。magnitude 映射到 angle=[0, 15]度。"""
    angle = magnitude * 15.0
    if random.random() < 0.5:
        angle = -angle
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _shear_x(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Shear (horizontal): 水平剪切。magnitude 映射到 shear=[0, 0.1]。"""
    shear = magnitude * 0.1
    if random.random() < 0.5:
        shear = -shear
    h, w = img.shape[:2]
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _shear_y(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Shear (vertical): 垂直剪切。magnitude 映射到 shear=[0, 0.1]。"""
    shear = magnitude * 0.1
    if random.random() < 0.5:
        shear = -shear
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [shear, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _translate_x(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Translate (horizontal): 水平平移。magnitude 映射到 offset=[0, 0.1]*width。"""
    offset = int(magnitude * 0.1 * img.shape[1])
    if random.random() < 0.5:
        offset = -offset
    M = np.float32([[1, 0, offset], [0, 1, 0]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def _translate_y(img: np.ndarray, magnitude: float) -> np.ndarray:
    """Translate (vertical): 垂直平移。magnitude 映射到 offset=[0, 0.1]*height。"""
    offset = int(magnitude * 0.1 * img.shape[0])
    if random.random() < 0.5:
        offset = -offset
    M = np.float32([[1, 0, 0], [0, 1, offset]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


# 策略列表（与旧版一致）
POLICIES = [
    _auto_contrast,
    _equalize,
    _posterize,
    _solarize_add,
    _adjust_color,
    _adjust_contrast,
    _adjust_brightness,
    _adjust_sharpness,
    _rotate,
    _shear_x,
    _shear_y,
    _translate_x,
    _translate_y,
]


class RandAugment:
    """RandAugment: 随机选择 N 个增强策略并顺序执行。

    对齐旧版 cchess_reg 配置：
    - 13 种策略（AutoContrast, Equalize, Posterize, SolarizeAdd, Color, Contrast,
      Brightness, Sharpness, Rotate, ShearX/Y, TranslateX/Y）
    - num_policies=3, magnitude_level=5, total_level=10

    Args:
        num_policies: 每次随机选择的策略数量
        magnitude_level: 增强强度级别 (0~total_level)
        total_level: 总级别数
        prob: 执行概率
    """

    def __init__(
        self,
        num_policies: int = 3,
        magnitude_level: int = 5,
        total_level: int = 10,
        prob: float = 1.0,
    ):
        self.num_policies = num_policies
        self.magnitude = magnitude_level / total_level  # 归一化到 [0, 1]
        self.prob = prob

    def __call__(self, image: np.ndarray, label: Tensor) -> tuple[np.ndarray, Tensor]:
        if random.random() >= self.prob:
            return image, label

        policies = random.sample(POLICIES, self.num_policies)
        for policy in policies:
            image = policy(image, self.magnitude)

        return image, label
