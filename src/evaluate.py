"""中国象棋棋盘识别评估指标。

多层次评估:
1. 位置级 AP — 90 个位置各自识别准确度
2. 类别级 AP — 16 个类别各自的 AP
3. mAP — 全局平均精度
4. P/R/F1 — 精确率/召回率/F1
5. 全盘准确率 — 整盘完全正确的比例 + errK 容错准确率
"""

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from .dataset import IDX_TO_FEN_CHAR, NUM_CLASSES

# 类别名称 (index → name)
CLASS_NAMES = [IDX_TO_FEN_CHAR[i] for i in range(NUM_CLASSES)]

# 棋子类别 (排除空位 . 和 x)
PIECE_LABELS = list(range(2, NUM_CLASSES))


def _average_precision(pred: Tensor, target: Tensor) -> float:
    """计算单个类别的 AP。

    Args:
        pred: (N,) 预测分数
        target: (N,) 0/1 真值

    Returns:
        AP 值
    """
    eps = 1e-8

    # 移除无效 target
    valid = target > -1
    pred = pred[valid]
    target = target[valid]

    # 按预测分数降序排序
    sorted_indices = torch.argsort(pred, descending=True)
    sorted_target = target[sorted_indices]

    # 累计 TP
    is_tp = sorted_target == 1
    tps = torch.cumsum(is_tp.float(), 0)
    total_pos = tps[-1].item()

    # 累计预测数量
    pred_count = torch.arange(1, len(sorted_target) + 1, dtype=torch.float32)

    # 只保留 TP 位置的 precision
    precision = torch.zeros_like(tps)
    precision[is_tp] = tps[is_tp] / pred_count[is_tp]

    ap = precision.sum().item() / max(total_pos, eps)
    return ap


class CChessEvaluator:
    """棋盘识别评估器，累积预测结果后统一计算指标。

    用法:
        evaluator = CChessEvaluator()
        for batch in dataloader:
            logits = model(images)
            evaluator.add_batch(logits, labels)
        metrics = evaluator.compute()
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.all_preds: List[Tensor] = []   # 预测类别
        self.all_logits: List[Tensor] = []  # 原始 logits (for AP)
        self.all_labels: List[Tensor] = []  # 真值类别

    def reset(self):
        self.all_preds.clear()
        self.all_logits.clear()
        self.all_labels.clear()

    def add_batch(self, logits: Tensor, labels: Tensor):
        """添加一个 batch 的预测结果。

        Args:
            logits: (B, 10, 9, 16) 模型原始输出（softmax 前）
            labels: (B, 10, 9) 真值 class indices
        """
        preds = logits.argmax(dim=-1)  # (B, 10, 9)
        self.all_preds.append(preds.cpu())
        self.all_logits.append(logits.cpu())
        self.all_labels.append(labels.cpu())

    def compute(self) -> dict:
        """计算所有指标。"""
        preds = torch.cat(self.all_preds, dim=0)      # (N, 10, 9)
        logits = torch.cat(self.all_logits, dim=0)     # (N, 10, 9, 16)
        labels = torch.cat(self.all_labels, dim=0)     # (N, 10, 9)

        probs = F.softmax(logits, dim=-1)  # (N, 10, 9, 16)

        metrics = {}
        metrics.update(self._compute_class_ap(probs, labels))
        metrics.update(self._compute_position_ap(probs, labels))
        metrics.update(self._compute_full_accuracy(preds, labels))
        metrics.update(self._compute_prf1(preds, labels))
        metrics.update(self._compute_piece_only_metrics(probs, labels))

        return metrics

    def _compute_class_ap(self, probs: Tensor, labels: Tensor) -> dict:
        """计算每个类别的 AP 和 mAP。

        probs: (N, 10, 9, 16), labels: (N, 10, 9)
        """
        # 将所有位置展平
        probs_flat = probs.reshape(-1, self.num_classes)   # (N*90, 16)
        labels_flat = labels.reshape(-1)                     # (N*90,)
        target_onehot = F.one_hot(labels_flat, self.num_classes).float()  # (N*90, 16)

        metrics = {}
        aps = []
        for c in range(self.num_classes):
            ap = _average_precision(probs_flat[:, c], target_onehot[:, c])
            aps.append(ap)
            metrics[f"class_AP_{CLASS_NAMES[c]}"] = ap * 100.0

        metrics["mAP"] = sum(aps) / len(aps) * 100.0
        return metrics

    def _compute_position_ap(self, probs: Tensor, labels: Tensor) -> dict:
        """计算每个棋盘位置的 AP。

        probs: (N, 10, 9, 16), labels: (N, 10, 9)
        """
        metrics = {}
        aps = []

        for row in range(10):
            for col in range(9):
                pos_probs = probs[:, row, col, :]     # (N, 16)
                pos_labels = labels[:, row, col]       # (N,)
                target_onehot = F.one_hot(pos_labels, self.num_classes).float()

                pos_aps = []
                for c in range(self.num_classes):
                    ap = _average_precision(pos_probs[:, c], target_onehot[:, c])
                    pos_aps.append(ap)

                pos_map = sum(pos_aps) / len(pos_aps) * 100.0
                tag = f"{chr(65 + row)}{col}"  # A0, A1, ..., J8
                metrics[f"pos_AP_{tag}"] = pos_map
                aps.append(pos_map)

        metrics["position_mAP"] = sum(aps) / len(aps)
        return metrics

    def _compute_full_accuracy(self, preds: Tensor, labels: Tensor) -> dict:
        """计算全盘准确率和 errK 容错准确率。"""
        # 每个样本的错误数
        errors = (preds != labels).sum(dim=(1, 2)).float()  # (N,)
        n = len(errors)

        metrics = {
            "full_accuracy": (errors == 0).float().mean().item() * 100.0,
        }

        for k in (1, 3, 5):
            metrics[f"err{k}_accuracy"] = (errors <= k).float().mean().item() * 100.0

        return metrics

    def _compute_prf1(self, preds: Tensor, labels: Tensor) -> dict:
        """计算精确率、召回率、F1（位置级别）。"""
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        # 每个类别的 P/R/F1
        precisions, recalls, f1s = [], [], []
        for c in range(self.num_classes):
            pred_c = (preds_flat == c)
            label_c = (labels_flat == c)
            tp = (pred_c & label_c).sum().float()
            fp = (pred_c & ~label_c).sum().float()
            fn = (~pred_c & label_c).sum().float()

            p = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
            r = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        # macro 平均
        n = len(precisions)
        metrics = {
            "macro_precision": sum(precisions) / n * 100.0,
            "macro_recall": sum(recalls) / n * 100.0,
            "macro_f1": sum(f1s) / n * 100.0,
        }

        # micro 平均（全局 TP/FP/FN）
        tp_all = (preds_flat == labels_flat).sum().float()
        total = preds_flat.numel()
        metrics["micro_accuracy"] = (tp_all / total).item() * 100.0

        return metrics

    def _compute_piece_only_metrics(self, probs: Tensor, labels: Tensor) -> dict:
        """仅评估棋子类别（排除空位 . 和 x）的指标。"""
        # 将所有位置展平
        probs_flat = probs.reshape(-1, self.num_classes)
        labels_flat = labels.reshape(-1)

        # 过滤: 只保留标签为棋子的位置
        piece_mask = labels_flat >= 2
        if piece_mask.sum() == 0:
            return {"piece_only_mAP": 0.0}

        probs_pieces = probs_flat[piece_mask][:, PIECE_LABELS]      # (M, 14)
        labels_pieces = labels_flat[piece_mask] - 2                   # remap to 0-13
        target_onehot = F.one_hot(labels_pieces, len(PIECE_LABELS)).float()

        aps = []
        for c in range(len(PIECE_LABELS)):
            ap = _average_precision(probs_pieces[:, c], target_onehot[:, c])
            aps.append(ap)

        return {"piece_only_mAP": sum(aps) / len(aps) * 100.0}


def compute_cchess_metrics(eval_preds) -> dict:
    """HF Trainer compute_metrics 回调函数。

    Args:
        eval_preds: transformers.EvalPrediction 对象
            - predictions: (N, 10, 9, 16) logits
            - label_ids: (N, 10, 9) labels

    Returns:
        指标字典
    """
    import numpy as np

    logits = torch.from_numpy(np.asarray(eval_preds.predictions))
    labels = torch.from_numpy(np.asarray(eval_preds.label_ids))

    # 移除可能的 padding batch
    valid_mask = labels >= 0
    if valid_mask.all(dim=(1, 2)).any():
        # 取有效样本
        valid = valid_mask.all(dim=(1, 2))
        logits = logits[valid]
        labels = labels[valid]

    evaluator = CChessEvaluator()
    evaluator.add_batch(logits, labels)
    return evaluator.compute()
