"""Mock 推理演示：验证 CChessNet 管线连通性。"""

import torch
from src.model import CChessNet

# 16 类标签映射
CLASS_NAMES = [
    ".", "x",  # 空白、其他
    "K", "A", "B", "N", "R", "C", "P",  # 红方：帅仕相马车炮兵
    "k", "a", "b", "n", "r", "c", "p",  # 黑方：将士象马车炮卒
]


def print_board(predictions: torch.Tensor) -> None:
    """打印 10x9 棋盘预测结果。

    Args:
        predictions: [1, 10, 9, 16] softmax 输出
    """
    pred_classes = predictions[0].argmax(dim=-1)  # [10, 9]
    pred_scores = predictions[0].max(dim=-1).values  # [10, 9]

    print("\n棋盘预测结果 (argmax)：")
    print("  " + " ".join(str(i) for i in range(9)))
    for row in range(10):
        pieces = [CLASS_NAMES[pred_classes[row, col].item()] for col in range(9)]
        score = pred_scores[row].mean().item()
        print(f"{row} " + " ".join(pieces) + f"  (avg_conf: {score:.3f})")


def main():
    print("=== CChessNet Mock 推理演示 ===\n")

    # 1. 构建模型
    print("加载模型...")
    model = CChessNet()
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 2. 生成 mock 输入
    print(f"\n生成 mock 输入: [1, 3, {CChessNet.INPUT_HEIGHT}, {CChessNet.INPUT_WIDTH}]")
    dummy_input = torch.randn(1, 3, CChessNet.INPUT_HEIGHT, CChessNet.INPUT_WIDTH)

    # 3. 前向推理
    print("执行前向推理...")
    with torch.no_grad():
        output = model(dummy_input)

    # 4. 验证输出
    print(f"\n输出 shape: {output.shape}")
    assert output.shape == (1, 10, 9, 16), f"输出 shape 错误: {output.shape}"
    print(f"softmax 验证 (sum≈1.0): {output[0, 0, 0].sum().item():.6f}")

    # 5. 打印棋盘
    print_board(output)

    print("\n=== 推理演示完成 ===")


if __name__ == "__main__":
    main()
