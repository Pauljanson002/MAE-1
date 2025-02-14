import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import os


class CosineScheduler:
    def __init__(self, optimizer, args):
        self.args = args
        self.optimizer = optimizer

    def step(self, epoch, current_task=0):
        return adjust_learning_rate(self.optimizer, epoch, self.args)

class InfiniteScheduler:
    def __init__(self, optimizer, args):
        self.args = args
        self.optimizer = optimizer

    def step(self, iter_num, total_iters, epoch, current_task=0, decay_style="cosine_cooldown_infinite"):
        return adjust_learning_rate_2(
            self.optimizer,
            iter_num,
            total_iters,
            epoch,
            self.args,
            current_task,
            decay_style,
        )

def adjust_learning_rate_2(
    optimizer,
    iter_num,
    total_iters,
    epoch,
    args,
    current_task=0,
    decay_style="cosine_cooldown_infinite",
):
    lr = infinite_lr(
        current_iter=iter_num,
        total_iters=total_iters,
        current_epoch=epoch,
        total_epochs=args.total_epoch,
        current_task=current_task,
        warmup_ratio=args.warmup_ratio,
        cooldown_ratio=args.cooldown_ratio,
        constant_ratio=args.constant_ratio,
        start_lr=args.lr,
        constant_lr=args.lr * args.constant_lr_ratio,
        min_lr=args.lr * args.min_lr_ratio,
        decay_style=decay_style,
        timescale=10,
    )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def infinite_lr(
    current_iter: int,
    total_iters: int,
    current_epoch: int,
    total_epochs: int,
    current_task: int,
    warmup_ratio: float = 0.05,
    cooldown_ratio: float = 0.15,
    constant_ratio: float = 0.80,
    start_lr: float = 1e-3,
    constant_lr: float = 1e-4,
    min_lr: float = 1e-5,
    decay_style: str = "cosine_cooldown_infinite",
    timescale: float = 10,
) -> Tuple[float, int]:

    # Calculate total iterations across all epochs
    total_iterations = total_iters * total_epochs
    actual_current_iter = current_epoch * total_iters + current_iter
    # Calculate iterations for each phase
    if current_task == 0:
        warmup_iters = int(total_iterations * warmup_ratio)
        cooldown_iters = int(total_iterations * cooldown_ratio)
        constant_iters = int(total_iterations * constant_ratio)
        decay_iters = total_iterations - constant_iters
    else:
        warmup_iters = 0
        cooldown_iters = 0
        constant_iters = int(total_iterations * constant_ratio)
        decay_iters = total_iterations - constant_iters

    # Calculate learning rate
    if current_task == 0:
        if actual_current_iter <= warmup_iters:
            # Warmup phase (only for the first task)
            lr = (actual_current_iter / warmup_iters) * start_lr
        elif actual_current_iter <= cooldown_iters:
            # Cooldown phase
            cooldown_progress = (actual_current_iter - warmup_iters) / (
                cooldown_iters - warmup_iters
            )
            if decay_style == "constant_infinite":
                lr = start_lr - ((start_lr - constant_lr) * cooldown_progress)
            elif decay_style == "inverse_sqrt_infinite":

                def inv_f(t):
                    return (1 / math.sqrt(1 + (timescale * t))) - 1

                lr = start_lr + (
                    (constant_lr - start_lr) / inv_f(1) * inv_f(cooldown_progress)
                )
            elif decay_style == "cosine_cooldown_infinite":
                lr = constant_lr + (
                    (start_lr - constant_lr)
                    / 2.0
                    * (math.cos(math.pi * cooldown_progress) + 1)
                )
            else:
                raise NotImplementedError(f"Decay style {decay_style} not implemented")
        elif actual_current_iter <= constant_iters:
            # Constant phase
            lr = constant_lr
        else:
            # Decay phase
            decay_iter = actual_current_iter - constant_iters
            exp_factor = -math.log(min_lr / constant_lr) / (
                total_iterations - constant_iters
            )
            lr = constant_lr * math.exp(-1 * exp_factor * decay_iter)
    else:
        # For tasks after the first one: only constant and decay phases
        if actual_current_iter <= constant_iters:
            lr = constant_lr
        else:
            decay_iter = (
                actual_current_iter - warmup_iters - cooldown_iters - constant_iters
            )
            exp_factor = math.log(constant_lr / min_lr) / decay_iters
            lr = constant_lr * math.exp(-exp_factor * decay_iter)

    return lr


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epoch:
        lr = args.lr * epoch / args.warmup_epoch
    else:
        lr = args.lr * args.min_lr_ratio + (
            args.lr - args.lr * args.min_lr_ratio
        ) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epoch)
                / (args.total_epoch - args.warmup_epoch)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
