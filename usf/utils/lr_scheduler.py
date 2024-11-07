#!/usr/bin/env python3
#
# Created on Mon Nov 03 2025 17:05:04
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_rate: float = 0.0,
) -> float:
    """Compute the LR multiplier for a cosine-with-warmup schedule.

    During the first ``num_warmup_steps`` the multiplier ramps linearly from 0 to 1.
    After warmup, it follows a cosine decay from 1 to ``min_lr_rate``.

    Args:
        current_step (int): The current training step (0-indexed).
        num_warmup_steps (int): Steps for the linear warmup phase.
        num_training_steps (int): Total number of training steps.
        num_cycles (float): Number of cosine half-cycles (0.5 = single decay to min).
        min_lr_rate (float, optional): Minimum LR as a fraction of the peak LR.
            Defaults to 0.0.

    Returns:
        float: Learning-rate multiplier in [0, 1].
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    warmup_steps_rate: float | None = None,
    min_lr_rate: float | None = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        num_training_steps (int): The total number of training steps.
        num_cycles (float, optional): The number of waves in the cosine schedule. Defaults to 0.5.
        last_epoch (int, optional): The index of the last epoch when resuming training. Defaults to -1.
        warmup_steps_rate (float | None, optional): The rate as opposed to total steps to warmup lr until max. Defaults to None.
        min_lr_rate (float | None, optional): The minimum learning rate as a ratio of the initial learning rate. Defaults to None.

    Returns:
        LambdaLR: Scheduler with the cosine-with-warmup lambda.
    """
    warmup_steps_rate = warmup_steps_rate or 0.0
    min_lr_rate = min_lr_rate or 0.0

    assert (
        0.0 <= warmup_steps_rate <= 1.0
    ), f"warmup_steps_rate must be in [0.0, 1.0], got {warmup_steps_rate}"
    assert (
        0.0 <= min_lr_rate <= 1.0
    ), f"min_lr_rate must be in [0.0, 1.0], got {min_lr_rate}"

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=int(warmup_steps_rate * num_training_steps),
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
