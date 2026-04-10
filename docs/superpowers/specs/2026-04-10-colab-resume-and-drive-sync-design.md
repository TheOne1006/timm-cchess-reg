# Colab Training Resume & Google Drive Sync

## Problem

Colab free tier has limited compute — training cannot complete in one session. Need ability to interrupt and resume training, plus persist outputs to Google Drive so they survive session resets.

## Design

### 1. `train.py`: `--resume_from` parameter

Add `--resume_from` CLI argument. When provided, pass to HF Trainer's `resume_from_checkpoint`. No auto-detection logic — user specifies the exact checkpoint path.

```python
parser.add_argument("--resume_from", type=str, default=None,
                    help="恢复训练的 checkpoint 路径")

# In train():
if args.resume_from:
    trainer.train(resume_from_checkpoint=args.resume_from)
else:
    trainer.train()
```

HF Trainer checkpoints already contain: model weights, optimizer state, scheduler state, RNG state, epoch, global_step. No custom save logic needed.

### 2. Notebook: `resume_from` parameter cell

Cell 5 (training config) adds a form field:

```python
resume_from = ""  # @param {type:"string"} 留空=从头训练
args.resume_from = resume_from if resume_from else None
```

User fills in the checkpoint path (e.g. `outputs/checkpoint-500`) or leaves empty for fresh training.

### 3. Notebook: Sync to Drive cell

New cell. Syncs `outputs/` to Drive as-is. No filtering, no "latest" detection — user decides when to run it.

```python
import shutil
drive_output = "/content/drive/MyDrive/cchess_outputs"
shutil.copytree("outputs", drive_output, dirs_exist_ok=True)
print(f"已同步到 {drive_output}")
```

### 4. Notebook: Restore from Drive cell

New cell. Copies Drive outputs back to local. User decides when to run it.

```python
import shutil
drive_output = "/content/drive/MyDrive/cchess_outputs"
if os.path.exists(drive_output):
    shutil.copytree(drive_output, "outputs", dirs_exist_ok=True)
    print("已从 Drive 恢复 outputs/")
else:
    print("Drive 上无 outputs/，跳过恢复")
```

## User Control

- User manually specifies `resume_from` checkpoint path
- User manually runs sync/restore cells when needed
- No auto-detection, no "latest checkpoint" logic

## Files Changed

- `src/train.py` — add `--resume_from` arg, ~5 lines
- `colab/cchess_reg_training.ipynb` — add `resume_from` param, sync cell, restore cell
