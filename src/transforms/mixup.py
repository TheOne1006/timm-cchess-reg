"""棋子粘贴增强: CChessMixSinglePngCls。

参考: cchess_reg/datasets/transforms/cchess_mix_single_png_cls.py

逻辑:
- 加载所有棋子 PNG (RGBA)，使用 alpha 通道作为 mask
- 只在 label==0（空位）的位置粘贴
- 支持随机缩放、旋转、翻转
- 使用 flat index 定位 cell: cell_index // 9 = row, cell_index % 9 = col
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch import Tensor

logger = logging.getLogger(__name__)

from ..dataset import IMG_HEIGHT, IMG_WIDTH


@dataclass
class PieceCacheItem:
    """棋子 PNG 缓存项（多尺度预缓存）。"""
    # scales: { (w, h): (img_rgb, mask) } — 预 resize 到的离散尺度
    scales: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    label: int  # class index (0-15)


# 预缓存尺度因子：覆盖 cell_scale [1.0, 1.5] 范围
_CACHE_SCALES = [1.0, 1.0625, 1.125, 1.1875, 1.25, 1.3125, 1.375, 1.4375, 1.5]


class CChessMixSinglePngCls:
    """在棋盘空位上随机粘贴棋子 PNG 图片。

    忠实复现旧版 CChessMixSinglePngCls：
    - 使用 flat index (0-89) 定位 cell
    - 只在 label==0（point 空位）粘贴
    - 粘贴数量: random.randint(max_mix_cells//2, max_mix_cells)
    - alpha mask 混合: new * mask + original * (1 - mask)
    - 越界自动裁剪

    旧版 prob 语义: prob=0.7 时，random < prob → return unchanged，
    即 prob 越大，不执行概率越大。实际执行率 = 1 - prob。
    为避免混淆，本版 prob 直接表示"执行概率"：
    - prob=0.3 → 30% 执行（与旧版 prob=0.7 的 30% 实际执行率一致）

    Args:
        png_dir: 棋子 PNG 根目录
        img_scale: 图像尺寸 (width, height)
        max_mix_cells: 最多粘贴多少个棋子
        cell_scale: 棋子缩放范围
        rotate_angle: 旋转角度范围
        prob: 执行概率（0-1）
    """

    CATE_TO_LABEL = {
        "point": 0, "other": 1,
        "red_king": 2, "red_advisor": 3, "red_bishop": 4,
        "red_knight": 5, "red_rook": 6, "red_cannon": 7, "red_pawn": 8,
        "black_king": 9, "black_advisor": 10, "black_bishop": 11,
        "black_knight": 12, "black_rook": 13, "black_cannon": 14, "black_pawn": 15,
    }

    def __init__(
        self,
        png_dir: str,
        img_scale: tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT),
        max_mix_cells: int = 15,
        cell_scale: tuple[float, float] = (1.0, 1.5),
        rotate_angle: tuple[float, float] = (-180, 180),
        prob: float = 0.3,
    ):
        self.img_scale = img_scale
        self.max_mix_cells = max_mix_cells
        self.cell_scale = cell_scale
        self.rotate_angle = rotate_angle
        self.prob = prob
        self.item_cell_width = img_scale[0] / 9
        self.item_cell_height = img_scale[1] / 10

        # 棋盘网格坐标 (10, 9, 2) — (x, y) 像素坐标
        self.cchess_table_10x9 = np.zeros((10, 9, 2), dtype=np.float32)
        for row in range(10):
            for col in range(9):
                self.cchess_table_10x9[row, col] = [
                    col * self.item_cell_width,
                    row * self.item_cell_height,
                ]

        self.cache_items: List[PieceCacheItem] = []
        self._load_png_resources(png_dir)

    def _load_png_resources(self, png_dir: str):
        """加载所有类别目录下的 PNG 图片，预 resize 到多个离散尺度。"""
        png_path = Path(png_dir)
        if not png_path.exists():
            return

        base_w = int(self.item_cell_width)
        base_h = int(self.item_cell_height)

        for cate_name, label_idx in self.CATE_TO_LABEL.items():
            if cate_name in ("point", "other"):
                continue
            cate_dir = png_path / cate_name
            if not cate_dir.exists():
                continue
            for f in cate_dir.glob("*.png"):
                img = Image.open(f).convert("RGBA")
                img_np = np.array(img)
                if img_np.shape[2] != 4:
                    continue
                raw_rgb = img_np[:, :, :3]
                raw_mask = img_np[:, :, 3]

                # 预 resize 到每个离散尺度
                scales = {}
                for s in _CACHE_SCALES:
                    sw, sh = int(base_w * s), int(base_h * s)
                    scales[(sw, sh)] = (
                        np.array(Image.fromarray(raw_rgb).resize((sw, sh), Image.BILINEAR)),
                        np.array(Image.fromarray(raw_mask).resize((sw, sh), Image.BILINEAR)),
                    )
                self.cache_items.append(PieceCacheItem(scales=scales, label=label_idx))

    def _get_cell_xy(self, cell_index: int) -> np.ndarray:
        """flat index (0-89) → (x, y) 像素坐标。"""
        row = cell_index // 9
        col = cell_index % 9
        return self.cchess_table_10x9[row, col]

    def _paste_cell_img(self, cell_index: int, img: np.ndarray, cache_item: PieceCacheItem):
        """在指定 cell 位置粘贴一个棋子。

        使用预缓存的多尺度图片，运行时无需 PIL resize。
        """
        x, y = self._get_cell_xy(cell_index)

        # 随机选尺度
        cell_scale = random.uniform(self.cell_scale[0], self.cell_scale[1])
        target_w = int(self.item_cell_width * cell_scale)
        target_h = int(self.item_cell_height * cell_scale)

        # 选最近的预缓存尺寸
        best_size = min(cache_item.scales.keys(),
                        key=lambda s: abs(s[0] - target_w) + abs(s[1] - target_h))
        cell_img, mask = cache_item.scales[best_size]
        w, h = best_size

        x, y = int(x), int(y)

        # 随机旋转
        if self.rotate_angle[0] != 0 or self.rotate_angle[1] != 0:
            angle = random.uniform(self.rotate_angle[0], self.rotate_angle[1])
            cell_img_pil = Image.fromarray(cell_img).rotate(angle, expand=False, fillcolor=(0, 0, 0))
            mask_pil = Image.fromarray(mask).rotate(angle, expand=False, fillcolor=0)
            cell_img = np.array(cell_img_pil)
            mask = np.array(mask_pil)

        # 随机上下翻转
        if random.random() > 0.5:
            cell_img = np.flipud(cell_img).copy()
            mask = np.flipud(mask).copy()

        # 随机左右翻转
        if random.random() > 0.5:
            cell_img = np.fliplr(cell_img).copy()
            mask = np.fliplr(mask).copy()

        # 越界处理：裁剪到实际可用区域
        origin_img_part = img[y:y + h, x:x + w]
        if origin_img_part.shape[0] != h or origin_img_part.shape[1] != w:
            actual_h = origin_img_part.shape[0]
            actual_w = origin_img_part.shape[1]
            mask = np.array(Image.fromarray(mask).resize((actual_w, actual_h), Image.BILINEAR))
            cell_img = np.array(Image.fromarray(cell_img).resize((actual_w, actual_h), Image.BILINEAR))

        # alpha mask 混合
        mask_binary = (mask > 122).astype(np.uint8)[..., np.newaxis]

        try:
            img[y:y + h, x:x + w] = cell_img * mask_binary + origin_img_part * (1 - mask_binary)
        except ValueError:
            logger.debug("Piece paste dimension mismatch at cell %d, skipping", cell_index)

    def __call__(self, image, label: Tensor):
        if random.random() >= self.prob:
            return image, label
        if not self.cache_items:
            return image, label

        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        label_np = label.numpy().copy()
        label_flat = label_np.flatten()

        # 找空位 (class index == 0, 即 point ".")
        empty_indices = [i for i, v in enumerate(label_flat) if v == 0]
        if not empty_indices:
            return img, torch.from_numpy(label_np)

        # 随机粘贴数量
        cover_num = random.randint(
            max(1, self.max_mix_cells // 2),
            self.max_mix_cells,
        )
        cover_num = min(cover_num, len(empty_indices))

        # 随机选择棋子和空位
        chosen_pieces = random.choices(self.cache_items, k=cover_num)
        chosen_positions = random.sample(empty_indices, cover_num)

        for cell_idx, piece in zip(chosen_positions, chosen_pieces):
            self._paste_cell_img(cell_idx, img, piece)
            label_flat[cell_idx] = piece.label

        label_np = label_flat.reshape(10, 9)
        label = torch.from_numpy(label_np)
        return img, label
