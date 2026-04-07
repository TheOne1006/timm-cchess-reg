"""将 CChessNet 转换为 CoreML 模型 (.mlpackage)。

独立脚本，不依赖 src/inference.py，可单独运行。
转换完成后可用 Xcode 打开 .mlpackage 查看模型结构。
"""

import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

# 将 src/ 加入 import 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))
from model import CChessNet

INPUT_HEIGHT = CChessNet.INPUT_HEIGHT
INPUT_WIDTH = CChessNet.INPUT_WIDTH
OUTPUT_PATH = Path(__file__).parent / "CChessNet.mlpackage"


def convert_to_coreml() -> None:
    print("=== CoreML 转换 ===\n")

    # 1. 加载模型
    print("加载 CChessNet...")
    model = CChessNet()
    model.eval()

    # 2. 创建示例输入并追踪模型
    print("追踪模型 (torch.jit.trace)...")
    example_input = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    traced_model = torch.jit.trace(model, example_input)

    # 3. 转换为 CoreML
    print("转换为 CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="image",
                shape=(1, 3, INPUT_HEIGHT, INPUT_WIDTH),
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="board_prediction", dtype=np.float32),
        ],
        convert_to="mlprogram",  # ML Program 格式，支持 ANE
    )

    # 4. 保存
    print(f"保存到 {OUTPUT_PATH}...")
    mlmodel.save(str(OUTPUT_PATH))
    print(f"\n转换完成！文件: {OUTPUT_PATH}")
    print("可用 Xcode 打开查看模型结构。")


if __name__ == "__main__":
    convert_to_coreml()
