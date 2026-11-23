# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from lerobot.processor import (
    PolicyProcessorPipeline,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_bagel_robotics import BagelRoboticsConfig


def make_bagel_robotics_pre_post_processors(
    config: BagelRoboticsConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Keep RGB in [0,1] and physical state/actions; Bagel applies checkpoint normalization."""
    steps = make_default_policy_processor_steps(config)
    return make_policy_processor_pipelines(
        [steps.rename_observations, steps.add_batch_dim, steps.to_device], [steps.to_cpu]
    )
