"""中国象棋棋盘识别训练入口。

使用 HuggingFace Trainer API 进行训练。

用法:
    uv run python -m src.train --data_dir datasets/demo --epochs 100 --batch_size 8
    uv run python -m src.train --data_dir /path/to/full_dataset --png_dir datasets/single_cls2_png
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import random_split

from .dataset import CChessDataset
from .evaluate import compute_cchess_metrics
from .model import CChessNet
from .transforms import train_transform, val_transform


def collate_fn(batch):
    """自定义 collate：将 list[(image, label)] 堆叠为 batch tensor。"""
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return {"pixel_values": images, "labels": labels}


class SubsetWithTransform(torch.utils.data.Subset):
    """支持自定义 transform 的 Subset。

    通过 __getitem__ 调用父级 dataset 的完整 __getitem__（含原始 transform），
    然后叠加子集级别的 transform。
    """

    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.sub_transform = transform

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        if self.sub_transform is not None:
            image, label = self.sub_transform(image, label)
        return image, label

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


class HFModelWrapper(torch.nn.Module):
    """包装 CChessNet 使其兼容 HF Trainer。"""

    def __init__(self, cchess_model: CChessNet):
        super().__init__()
        self.cchess = cchess_model

    def forward(self, pixel_values, labels=None):
        return self.cchess(pixel_values, labels=labels)


from transformers import Trainer


class CChessTrainer(Trainer):
    """自定义 Trainer：直接控制 DataLoader 创建，保留 collate_fn 和 drop_last。"""

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler
        nw = self.args.dataloader_num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=RandomSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=nw,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,
            prefetch_factor=2 if nw > 0 else None,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        from torch.utils.data import DataLoader, SequentialSampler
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        nw = self.args.dataloader_num_workers
        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=self.data_collator,
            num_workers=nw,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,
            prefetch_factor=2 if nw > 0 else None,
        )


def train(args):
    torch.backends.cudnn.benchmark = True

    # 数据集
    transform_train = train_transform(
        png_dir=args.png_dir,
        perspective_prob=args.perspective_prob,
        piece_paste_prob=args.piece_paste_prob,
        piece_max_cells=args.piece_max_cells,
    )
    transform_val = val_transform()

    full_dataset = CChessDataset(root=args.data_dir, transform=None)

    # 划分训练/验证集
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    train_dataset = SubsetWithTransform(full_dataset, train_indices, transform_train)
    val_dataset = SubsetWithTransform(full_dataset, val_indices, transform_val)

    print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    # 模型
    model = CChessNet(backbone_name=args.backbone)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {param_count:.2f}M")

    # HF Trainer
    from transformers import TrainingArguments

    hf_model = HFModelWrapper(model)

    steps_per_epoch = len(train_dataset) // args.batch_size
    eval_steps = steps_per_epoch * 2  # ~每 2 epoch 评估一次

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=int(args.warmup_ratio * args.epochs * len(train_dataset) / args.batch_size),
        lr_scheduler_type=args.scheduler,
        logging_steps=args.log_interval,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="full_accuracy",
        greater_is_better=True,
        fp16=args.fp16,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=args.report_to,
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
    )

    trainer = CChessTrainer(
        model=hf_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_cchess_metrics,
    )

    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best_model"))


def main():
    parser = argparse.ArgumentParser(description="中国象棋棋盘识别训练")

    # 数据
    parser.add_argument("--data_dir", type=str, required=True, help="数据集根目录")
    parser.add_argument("--png_dir", type=str, default=None, help="棋子 PNG 目录 (for PiecePaste)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 模型
    parser.add_argument("--backbone", type=str, default="convnext_atto.d2_in1k", help="timm backbone 名称")

    # 训练
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup 比例")
    parser.add_argument("--scheduler", type=str, default="cosine", help="学习率调度器")
    parser.add_argument("--fp16", action="store_true", help="启用 FP16 混合精度")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # 增强
    parser.add_argument("--perspective_prob", type=float, default=0.7, help="透视变换概率")
    parser.add_argument("--piece_paste_prob", type=float, default=0.7, help="棋子粘贴概率")
    parser.add_argument("--piece_max_cells", type=int, default=15, help="最大粘贴棋子数")

    # 恢复
    parser.add_argument("--resume_from", type=str, default=None, help="恢复训练的 checkpoint 路径")

    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔 (steps)")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="日志上报 (none/tensorboard/wandb)")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
