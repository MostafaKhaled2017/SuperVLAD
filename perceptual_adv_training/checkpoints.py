import os
from os.path import exists, join
from pathlib import Path


def maybe_copy_resume_checkpoint(args) -> None:
    if args.resume is None:
        return

    initial_checkpoint_path = join(args.save_dir, "initial_model.pth")
    if not exists(initial_checkpoint_path):
        Path(initial_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copyfile(args.resume, initial_checkpoint_path)


def apply_lr_schedule(optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def maybe_remove_old_checkpoint(args, checkpoint_epoch: int) -> None:
    if args.keep_every <= 1:
        return
    if checkpoint_epoch % args.keep_every == 0:
        return
    checkpoint_name = join(args.save_dir, f"checkpoint_epoch_{checkpoint_epoch:04d}.pth")
    if exists(checkpoint_name):
        os.remove(checkpoint_name)
