"""中国象棋棋盘识别数据集。

FEN 格式: 10 行 x 9 列，每行 9 个字符。
16 类: . x K A B N R C P k a b n r c p (index 0-15)
"""

from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

# FEN 字符 → class index 映射
FEN_CHAR_TO_IDX = {
    ".": 0, "x": 1,
    "K": 2, "A": 3, "B": 4, "N": 5, "R": 6, "C": 7, "P": 8,
    "k": 9, "a": 10, "b": 11, "n": 12, "r": 13, "c": 14, "p": 15,
}

IDX_TO_FEN_CHAR = {v: k for k, v in FEN_CHAR_TO_IDX.items()}

NUM_CLASSES = 16
BOARD_ROWS = 10
BOARD_COLS = 9
IMG_HEIGHT = 640
IMG_WIDTH = 576

# 原始数据集图像尺寸 (含 padding)，与老项目一致
RAW_IMG_WIDTH = 450
RAW_IMG_HEIGHT = 500

# CenterCrop 裁剪尺寸，移除 padding 区域 (与老项目 CenterCrop(400,450) 一致)
CROP_WIDTH = 400
CROP_HEIGHT = 450


def parse_fen_label(text: str) -> torch.Tensor:
    """解析 FEN 文本为 (10, 9) int tensor。

    Args:
        text: 10 行文本，每行 9 个字符（可能有尾部换行）

    Returns:
        (10, 9) int tensor，值为 0-15
    """
    lines = text.strip().split("\n")
    assert len(lines) == BOARD_ROWS, f"Expected {BOARD_ROWS} lines, got {len(lines)}"
    board = []
    for line in lines:
        line = line.strip()
        assert len(line) == BOARD_COLS, f"Expected {BOARD_COLS} chars, got '{line}'"
        row = [FEN_CHAR_TO_IDX[ch] for ch in line]
        board.append(row)
    return torch.tensor(board, dtype=torch.long)


class CChessDataset(Dataset):
    """中国象棋棋盘数据集。

    目录结构:
        root/
            xxx.jpg  — 棋盘图片
            xxx.txt  — FEN 格式标签 (10 行 x 9 列)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.samples = self._scan_samples()

    def _scan_samples(self) -> list[tuple[Path, Path]]:
        """扫描目录，配对 jpg-txt 文件。"""
        samples = []
        for img_path in sorted(self.root.glob("*.jpg")):
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                samples.append((img_path, label_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.samples[idx]

        # 加载图像 (HWC, RGB, uint8)
        image = Image.open(img_path).convert("RGB")
        # 解析标签
        label_text = label_path.read_text(encoding="utf-8")
        label = parse_fen_label(label_text)  # (10, 9)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label
