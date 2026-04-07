"""基础 Transform: Compose, Resize, ToTensorNormalize。"""

import torchvision.transforms.functional as F_tv
from PIL import Image
from torch import Tensor

from ..dataset import IMG_HEIGHT, IMG_WIDTH

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
