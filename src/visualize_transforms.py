"""可视化 transform pipeline 各阶段效果（不含 ImageNet normalize）。

用法 (Colab):
    !python -m src.visualize_transforms --image datasets/demo/xxx.jpg --png_dir datasets/single_cls2_png
    !python -m src.visualize_transforms --data_dir datasets/demo --png_dir datasets/single_cls2_png
"""

import argparse
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互式后端，支持 Colab / headless 环境
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .dataset import (
    CChessDataset, IDX_TO_FEN_CHAR,
    BOARD_ROWS, BOARD_COLS,
    parse_fen_label,
)
from .transforms.pipeline import train_transform
from .transforms.base import ToTensorNormalize


def draw_board_overlay(image: Image.Image, label: torch.Tensor) -> Image.Image:
    """在图像上叠加棋盘网格线和棋子标注。"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cell_w = w / BOARD_COLS
    cell_h = h / BOARD_ROWS

    # 尝试加载较大字体 (自适应单元格大小，最小 16)
    font_size = max(int(min(cell_w, cell_h) * 0.5), 16)
    try:
        # Pillow >= 10.1.0 支持 load_default 直接传 size
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        # Fallback 尝试系统字体
        font_paths = [
            "Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        font = ImageFont.load_default()
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except OSError:
                continue

    # 网格线
    for row in range(BOARD_ROWS + 1):
        y = int(row * cell_h)
        draw.line([(0, y), (w, y)], fill="red", width=1)
    for col in range(BOARD_COLS + 1):
        x = int(col * cell_w)
        draw.line([(x, 0), (x, h)], fill="red", width=1)

    # 棋子标注
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            cls_idx = label[row, col].item()
            char = IDX_TO_FEN_CHAR[cls_idx]
            if char in ('.', 'x'):
                continue
            
            cx = int(col * cell_w + cell_w / 2)
            cy = int(row * cell_h + cell_h / 2)
            
            # 计算文字包围盒，使其完全居中
            try:
                bbox = draw.textbbox((0, 0), char, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = 10, 10
                
            draw.text((cx - text_w / 2, cy - text_h / 2 - 4), char, fill="yellow", font=font)

    return img


def capture_stages(transform, image: Image.Image, label: torch.Tensor):
    """逐阶段执行 transform，捕获中间结果（跳过 ToTensorNormalize）。"""
    stages = []
    current_img = image
    current_label = label

    for i, t in enumerate(transform.transforms):
        name = type(t).__name__
        if isinstance(t, ToTensorNormalize):
            stages.append((f"{i}: {name} (skipped)", current_img, current_label))
            continue
        current_img, current_label = t(current_img, current_label)
        stages.append((f"{i}: {name}", current_img, current_label))

    return stages


def visualize(args):
    # 加载图像和标签
    if args.image:
        image = Image.open(args.image).convert("RGB")
        label_path = Path(args.image).with_suffix(".txt")
        label = parse_fen_label(label_path.read_text(encoding="utf-8"))
    else:
        dataset = CChessDataset(root=args.data_dir, transform=None)
        idx = random.randint(0, len(dataset) - 1)
        image, label = dataset[idx]
        print(f"随机选取第 {idx} 张图片")

    print(f"图像尺寸: {image.size}, 标签形状: {label.shape}")

    # 构建 transform pipeline
    transform = train_transform(
        png_dir=args.png_dir,
        perspective_prob=args.perspective_prob,
        piece_paste_prob=args.piece_paste_prob,
        piece_max_cells=args.piece_max_cells,
    )

    # 捕获各阶段
    stages = capture_stages(transform, image, label)

    # 加上原图
    all_stages = [("0: Original", image, label)] + stages

    # 绘制网格
    n = len(all_stages)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, (name, img, lbl) in enumerate(all_stages):
        overlay = draw_board_overlay(img, lbl)
        axes_flat[i].imshow(np.array(overlay))
        axes_flat[i].set_title(name, fontsize=9)
        axes_flat[i].axis("off")

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"可视化已保存到 {args.output}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="可视化 transform pipeline 各阶段效果")

    # 数据源（二选一）
    parser.add_argument("--image", type=str, help="单张图片路径（自动匹配 .txt 标签）")
    parser.add_argument("--data_dir", type=str, help="数据集目录（随机选一张）")

    # Transform 参数
    parser.add_argument("--png_dir", type=str, default=None)
    parser.add_argument("--perspective_prob", type=float, default=0.7)
    parser.add_argument("--piece_paste_prob", type=float, default=0.7)
    parser.add_argument("--piece_max_cells", type=int, default=15)

    # 输出
    parser.add_argument("--output", type=str, default="transform_vis.png")

    args = parser.parse_args()

    if not args.image and not args.data_dir:
        parser.error("需要 --image 或 --data_dir")

    visualize(args)


if __name__ == "__main__":
    main()
