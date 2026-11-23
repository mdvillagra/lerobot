# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("bagel_robotics")
@dataclass
class BagelRoboticsConfig(PreTrainedConfig):
    """Bagel cp1000: head/left-wrist/right-wrist RGB and absolute dual-arm EE16 poses."""

    bundle_path: str | None = None
    wan_model_dir: str | None = None
    download_assets: bool = True
    local_files_only: bool = False
    camera_keys: tuple[str, str, str] = (
        "observation.images.head_camera",
        "observation.images.left_camera",
        "observation.images.right_camera",
    )
    chunk_size: int = 32
    n_action_steps: int = 16
    num_inference_steps: int = 10
    sigma_shift: float = 1.0
    inference_seed: int = 1
    dtype: torch.dtype | None = torch.bfloat16
    freeze_action_expert: bool = True
    use_gradient_checkpointing: bool = True
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 0.01
    push_to_hub: bool = False
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: dict.fromkeys(("VISUAL", "STATE", "ACTION"), NormalizationMode.IDENTITY)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.chunk_size, self.n_action_steps, self.n_obs_steps) != (32, 16, 1):
            raise ValueError("Bagel-Robotics-1 requires chunk_size=32, n_action_steps=16, n_obs_steps=1")
        if self.num_inference_steps < 1 or self.sigma_shift <= 0 or self.inference_seed < 0:
            raise ValueError("Require positive inference steps/shift and a nonnegative inference_seed")
        if self.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Bagel-Robotics-1 dtype must be bfloat16, float16, or float32")
        if len(self.camera_keys) != 3 or len(set(self.camera_keys)) != 3:
            raise ValueError("camera_keys must contain head, left wrist, right wrist, in that order")
        if any(not key.startswith("observation.images.") for key in self.camera_keys):
            raise ValueError("camera_keys must use the observation.images.<camera> convention")
        if any(
            self.normalization_mapping.get(key) != NormalizationMode.IDENTITY
            for key in ("VISUAL", "STATE", "ACTION")
        ):
            raise ValueError("Bagel normalizes internally; external normalization must be IDENTITY")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def image_observation_delta_indices(self) -> list[int]:
        # Dataset factory prepends the episode anchor, yielding 17 RGB frames.
        return [*range(-175, 1, 25), *range(4, 33, 4)]

    @property
    def image_observation_include_episode_start(self) -> bool:
        return True

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if set(self.image_features) != set(self.camera_keys):
            raise ValueError("Bagel requires exactly the three configured RGB camera features")
        if any(len(f.shape) != 3 or f.shape[0] != 3 for f in self.image_features.values()):
            raise ValueError("Bagel camera features must be RGB [3,H,W]")
        if self.robot_state_feature != PolicyFeature(FeatureType.STATE, (16,)):
            raise ValueError(f"Bagel requires {OBS_STATE} with shape (16,) in absolute dual-arm EE16 order")
        if self.action_feature != PolicyFeature(FeatureType.ACTION, (16,)):
            raise ValueError(f"Bagel requires {ACTION} with shape (16,) in absolute dual-arm EE16 order")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None
