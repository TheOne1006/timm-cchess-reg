"""基础 Transform: Compose, CenterCrop, Resize, ToTensorNormalize。"""

import numpy as np
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
    """Resize PIL Image 到固定尺寸。label 不变。"""

    def __init__(self, height: int = IMG_HEIGHT, width: int = IMG_WIDTH):
        self.size = (width, height)  # PIL resize 用 (w, h)

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Image.Image, Tensor]:
        image = image.resize(self.size, Image.BILINEAR)
        return image, label


class ToTensorNormalize:
    """PIL Image → Tensor + ImageNet normalize。"""

    def __call__(self, image: Image.Image, label: Tensor) -> tuple[Tensor, Tensor]:
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
