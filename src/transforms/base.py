"""基础 Transform: Compose, CenterCrop, Resize, ToTensorNormalize。"""

import numpy as np
import torch
import torchvision.transforms.functional as F_tv
from PIL import Image
from torch import Tensor

from ..dataset import CROP_HEIGHT, CROP_WIDTH, IMG_HEIGHT, IMG_WIDTH

# ImageNet 标准化参数 (timm convnext 默认)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Compose:
    """组合多个 transform，依次执行。"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Tensor, Tensor]:
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class CenterCrop:
    """从图像中心裁剪到指定尺寸，与 mmcv.transforms.CenterCrop 行为一致。

    裁掉原图 padding 区域（老项目原图 450×500 含 50px padding），
    CenterCrop(400, 450) 后得到 400×450 的纯净棋盘区域。

    Args:
        crop_width: 裁剪后宽度
        crop_height: 裁剪后高度
    """

    def __init__(self, crop_width: int = CROP_WIDTH, crop_height: int = CROP_HEIGHT):
        self.crop_size = (crop_width, crop_height)

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Image.Image, Tensor]:
        w, h = image.size
        crop_w, crop_h = self.crop_size

        # 如果图像已经小于等于裁剪尺寸，不裁剪
        if w <= crop_w and h <= crop_h:
            return image, label

        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)

        image = image.crop((x1, y1, x1 + crop_w, y1 + crop_h))
        return image, label


class Resize:
    """等比缩放 PIL Image 并居中 padding 到固定尺寸，保持宽高比。label 不变。

    先按 min(target_w/src_w, target_h/src_h) 等比缩放，
    再居中 pad 不足的部分（黑色填充），返回精确 target_w × target_h。
    """

    def __init__(self, height: int = IMG_HEIGHT, width: int = IMG_WIDTH):
        self.target_w = width
        self.target_h = height

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Image.Image, Tensor]:
        src_w, src_h = image.size  # PIL: (w, h)

        # 等比缩放因子
        scale = min(self.target_w / src_w, self.target_h / src_h)
        new_w = round(src_w * scale)
        new_h = round(src_h * scale)

        image = image.resize((new_w, new_h), Image.BILINEAR)

        # 居中 padding
        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h
        if pad_w > 0 or pad_h > 0:
            left = pad_w // 2
            top = pad_h // 2
            # PIL pad: (left, top, right, bottom)
            image = F_tv.pad(image, [left, top, pad_w - left, pad_h - top], fill=0)

        return image, label


class ToTensorNormalize:
    """PIL Image 或 numpy array → Tensor + ImageNet normalize。"""

    def __call__(self, image, label: Tensor) -> tuple[Tensor, Tensor]:
        if isinstance(image, np.ndarray):
            # numpy HWC uint8 → CHW float32 [0,1]
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            tensor = F_tv.to_tensor(image)  # [3, H, W], float32, [0,1]
        tensor = F_tv.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, label


class PILToNumpy:
    """PIL Image → numpy array。用于在 pipeline 中显式控制类型边界。"""

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[np.ndarray, Tensor]:
        return np.array(image), label


class NumpyToPIL:
    """numpy array → PIL Image。用于在 pipeline 中显式控制类型边界。"""

    def __call__(self, image: np.ndarray, label: Tensor) -> tuple[Image.Image, Tensor]:
        return Image.fromarray(image), label
