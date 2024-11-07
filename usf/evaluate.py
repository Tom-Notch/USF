#!/usr/bin/env python3
#
# Created on Sat Feb 08 2025 13:03:45
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
"""Evaluate (validate) a USF model on the test/validation split.

CLI entry point installed as ``evaluate``. Uses Hydra for configuration::

    evaluate task=semantic_segmentation
    evaluate task=object_detection

Automatically forces strategy to ``"auto"`` (single-GPU) for evaluation.
"""

import os
import os.path as osp

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

import usf  # for its __init__.py override, e.g. pretty print


@hydra.main(
    config_path=osp.join(os.environ["USF_PROJECT_DIRECTORY"], "config"),
    config_name="default.yaml",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Hydra entry point: instantiate trainer, data module, and model, then validate.

    Args:
        config (DictConfig): Hydra-resolved configuration.
    """
    print(OmegaConf.to_container(config, resolve=True))

    config.task.trainer.strategy = (
        "auto"  # default strategy for evaluation on a single GPU
    )
    trainer: Trainer = instantiate(config.task.trainer)
    data_module: LightningDataModule = instantiate(config.task.data_module)
    model: LightningModule = instantiate(config.task.model)

    trainer.validate(
        model,
        data_module,
        ckpt_path=OmegaConf.select(config, "task.checkpoint", default=None),
    )


if __name__ == "__main__":
    main()
