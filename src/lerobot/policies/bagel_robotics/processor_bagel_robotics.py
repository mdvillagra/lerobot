# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from lerobot.processor import PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_bagel_robotics import BagelRoboticsConfig


def make_bagel_robotics_pre_post_processors(
    config: BagelRoboticsConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Use identity processing. Bagel applies its checkpoint statistics inside the policy."""
    return make_default_pre_post_processors(config)
