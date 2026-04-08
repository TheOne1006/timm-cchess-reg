"""透视变换: RandomPerspective。

参考: cchess_reg/datasets/transforms/perspective_transform.py

优先使用 cv2.getPerspectiveTransform（更稳定），回退到 PIL。
透视变换只扭曲图像，不改变 label（网格空间位置不变）。
"""

import random

import numpy as np
from PIL import Image
from torch import Tensor


class RandomPerspective:
    """随机透视变换，模拟拍摄角度变化。

    Args:
        scale: 透视扰动强度范围 (min, max)
        size_scale: 缩放比例范围 (min, max)
        prob: 执行概率
    """

    def __init__(
        self,
        scale: tuple[float, float] = (0.05, 0.18),
        size_scale: tuple[float, float] = (0.7, 1.3),
        prob: float = 0.7,
    ):
        self.scale_range = scale
        self.size_scale_range = size_scale
        self.prob = prob
        self._use_cv2 = self._check_cv2()

    @staticmethod
    def _check_cv2() -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except ImportError:
            return False

    def __call__(self, image, label: Tensor):
        if random.random() >= self.prob:
            return image, label

        if self._use_cv2:
            return self._warp_cv2(image, label)
        return self._warp_pil(image, label)

    def _generate_dst_corners(self, w: int, h: int, scale: float, size_scale: float):
        """生成透视变换的目标角点坐标。返回 (4, 2) numpy array。

        共用于 cv2 和 PIL 两种实现。
        """
        sw = w * size_scale
        sh = h * size_scale
        ox = (w - sw) / 2
        oy = (h - sh) / 2

        return np.array([
            [ox + random.uniform(-scale * w, scale * w),
             oy + random.uniform(-scale * h, scale * h)],
            [ox + sw + random.uniform(-scale * w, scale * w),
             oy + random.uniform(-scale * h, scale * h)],
            [ox + random.uniform(-scale * w, scale * w),
             oy + sh + random.uniform(-scale * h, scale * h)],
            [ox + sw + random.uniform(-scale * w, scale * w),
             oy + sh + random.uniform(-scale * h, scale * h)],
        ], dtype=np.float32)

    def _warp_cv2(self, image, label: Tensor) -> tuple[np.ndarray, Tensor]:
        import cv2

        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        h, w = img.shape[:2]

        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        size_scale = np.random.uniform(self.size_scale_range[0], self.size_scale_range[1])

        pts1 = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        pts2 = self._generate_dst_corners(w, h, scale, size_scale)

        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_warped = cv2.warpPerspective(img, M, (w, h))
        return img_warped, label

    def _warp_pil(self, image, label: Tensor) -> tuple[np.ndarray, Tensor]:
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        w, h = pil_img.size
        scale = random.uniform(*self.scale_range)
        size_scale = random.uniform(*self.size_scale_range)

        src_pts = [(0, 0), (w, 0), (0, h), (w, h)]
        dst_pts = self._generate_dst_corners(w, h, scale, size_scale)
        dst_pts = [tuple(pt) for pt in dst_pts]

        try:
            coeffs = self._find_perspective_coeffs(src_pts, dst_pts)
            pil_img = pil_img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        except np.linalg.LinAlgError:
            pass
        return np.array(pil_img), label

    @staticmethod
    def _find_perspective_coeffs(
        src: list[tuple[float, float]],
        dst: list[tuple[float, float]],
    ) -> list[float]:
        matrix = []
        for (xs, ys), (xd, yd) in zip(src, dst):
            matrix.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys])
            matrix.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys])
        A = np.array(matrix, dtype=np.float64)
        B = np.array([c for pt in dst for c in pt], dtype=np.float64)
        return np.linalg.solve(A, B).tolist()
