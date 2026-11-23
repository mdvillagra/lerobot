# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LeRobot training/inference adapter for the optional, pinned Bagel backbone."""

from collections import deque
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _bagel_robotics_available, _hydra_available, require_package

from .assets_bagel_robotics import resolve_bagel_assets
from .configuration_bagel_robotics import BagelRoboticsConfig

if TYPE_CHECKING or (_bagel_robotics_available and _hydra_available):
    from bagel_robotics.inference.data import DEFAULT_PROMPT
    from bagel_robotics.inference.policy import RoboDojoVPP2Policy
    from bagel_robotics.inference.temporal_ensemble import TemporalEnsemble


class BagelRoboticsPolicy(PreTrainedPolicy):
    config_class = BagelRoboticsConfig
    name = "bagel_robotics"
    config: BagelRoboticsConfig
    _max_shard_size = "5GB"
    _fsdp_wrap_modules = ["DiTBlock"]

    def __init__(self, config: BagelRoboticsConfig, **kwargs: Any) -> None:
        super().__init__(config)
        config.validate_features()
        dataset_meta = kwargs.get("dataset_meta")
        if dataset_meta is not None and dataset_meta.fps != 25:
            raise ValueError("Bagel's history stride and video/action alignment require a 25-Hz dataset")
        require_package("bagel-robotics", extra="bagel_robotics", import_name="bagel_robotics")
        require_package("hydra-core", extra="bagel_robotics", import_name="hydra")
        require_package("transformers", extra="bagel_robotics")
        bundle, wan = resolve_bagel_assets(config)
        precision = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[config.dtype]
        self._runtime = RoboDojoVPP2Policy(
            checkpoint_path=bundle / "action.pt",
            video_checkpoint_path=bundle / "video.pt",
            dataset_stats_path=bundle / "dataset_stats.json",
            wan_model_dir=wan,
            device=config.device,
            mixed_precision=precision,
            num_inference_steps=config.num_inference_steps,
            sigma_shift=config.sigma_shift,
            seed=config.inference_seed,
            replan_steps=config.n_action_steps,
        )
        self.model = self._runtime.model
        self.model.requires_grad_(False)
        self.model.video_expert.requires_grad_(True)
        if not config.freeze_action_expert:
            self.model.action_expert.requires_grad_(True)
            self.model.proprio_encoder.requires_grad_(True)
        for expert in (self.model.video_expert, self.model.action_expert):
            expert.use_gradient_checkpointing = config.use_gradient_checkpointing
        self.model.mot.mot_checkpoint_mixed_attn = config.use_gradient_checkpointing
        for field in ("state", "action"):
            key = self._runtime.processor.shape_meta[field][0]["key"]
            norm = self._runtime.processor.normalizer.normalizers[field][key]
            for name in ("scale", "offset"):
                self.register_buffer(f"_{field}_{name}", getattr(norm, name).float().clone())
        self.reset()

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        # Recursive nn.Module.train() would re-enable dropout in frozen encoders.
        for module in (self.model.vae, self.model.text_encoder, self.model.clip_encoder):
            if module is not None:
                module.eval()
        if self.config.freeze_action_expert:
            self.model.action_expert.eval()
        if mode:
            self.reset()
        return self

    def get_optim_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self) -> None:
        self._runtime.reset()
        self._ensemble = TemporalEnsemble()
        self._action_queue: deque[Tensor] = deque()
        self._instruction: str | None = None

    def drop_queued_actions(self) -> None:
        self.reset()

    def _sync_runtime(self) -> None:
        # LeRobot may move a policy after construction (e.g. evaluation device).
        reference = next(self.model.video_expert.parameters())
        self.model.device = reference.device
        self.model.torch_dtype = reference.dtype
        self.model._restore_auxiliary_component_dtypes()

    @staticmethod
    def _tasks(batch: dict[str, Any], batch_size: int) -> list[str]:
        tasks = batch.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        if not isinstance(tasks, (list, tuple)) or len(tasks) != batch_size:
            raise ValueError("Bagel requires one task instruction string per batch element")
        if any(not isinstance(task, str) or not task.strip() for task in tasks):
            raise ValueError("Bagel task instructions must be nonempty strings")
        return list(tasks)

    def _observe(self, batch: dict[str, Any]) -> tuple[dict[str, Any], str]:
        self.eval()
        self._sync_runtime()
        state = batch[OBS_STATE]
        if state.shape != (1, 16) or not torch.isfinite(state).all():
            raise ValueError("Bagel synchronous inference requires finite state [1,16]; use one environment")
        instruction = self._tasks(batch, 1)[0]
        if self._instruction is not None and instruction != self._instruction:
            self.reset()
        self._instruction = instruction
        images = {}
        for key, native in zip(
            self.config.camera_keys, ("head_camera", "left_camera", "right_camera"), strict=True
        ):
            image = batch[key]
            if image.ndim != 4 or image.shape[:2] != (1, 3):
                raise ValueError(f"{key} must be the current RGB image [1,3,H,W]")
            images[native] = {"rgb": self._rgb_uint8(image)[0].permute(1, 2, 0).cpu().numpy()}
        observation = {
            "observation": images,
            "joint_action": {"vector": state[0].detach().float().cpu().numpy()},
        }
        # Record every control tick, including ticks playing queued actions.
        self._runtime.observe(observation)
        return observation, instruction

    @staticmethod
    def _rgb_uint8(image: Tensor) -> Tensor:
        if image.dtype == torch.uint8:
            return image
        if not image.is_floating_point() or not ((image >= 0) & (image <= 1)).all():
            raise ValueError("Bagel RGB must be uint8 [0,255] or floating point [0,1]")
        # Native adapter truncates float images when converting to simulator RGB.
        return (image.detach() * 255).to(torch.uint8)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Observe a tick and predict raw physical actions; use select_action for playback."""
        chunk = self._runtime._infer_action_chunk(*self._observe(batch))
        if chunk.shape != (32, 16) or not np.isfinite(chunk).all():
            raise ValueError("Bagel backbone must return a finite [32,16] action chunk")
        return torch.tensor(chunk, device=self.model.device, dtype=torch.float32).unsqueeze(0)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        observation, instruction = self._observe(batch)
        if not self._action_queue:
            chunk = self._runtime._infer_action_chunk(observation, instruction)
            actions = torch.from_numpy(self._ensemble.apply(chunk)).to(self.model.device)
            self._action_queue.extend(actions.unsqueeze(1))
        return self._action_queue.popleft()

    def _training_video(self, batch: dict[str, Any]) -> Tensor:
        views = []
        batch_size = batch[ACTION].shape[0]
        for key, size in zip(self.config.camera_keys, ((480, 640), (240, 240), (240, 240)), strict=True):
            image = batch[key]
            if image.ndim != 5 or image.shape[:3] != (batch_size, 17, 3):
                raise ValueError(f"{key} must contain [B,17,3,H,W]: episode anchor, history8, future8")
            image = self._rgb_uint8(image).flatten(0, 1).float().div(255)
            views.append(resize(image, list(size), InterpolationMode.BILINEAR, antialias=True))
        layout = torch.cat((views[0], torch.cat(views[1:], dim=2)), dim=3)
        video = resize(layout, [240, 416], InterpolationMode.BILINEAR, antialias=True)
        return video.mul(2).sub(1).unflatten(0, (batch_size, 17)).transpose(1, 2).contiguous()

    def forward(self, batch: dict[str, Any], reduction: str = "mean") -> tuple[Tensor, dict[str, float]]:
        if reduction != "mean":
            raise ValueError("Bagel's native training loss supports reduction='mean' only")
        self._sync_runtime()
        action, state = batch[ACTION], batch[OBS_STATE]
        if action.ndim != 3 or action.shape[1:] != (32, 16) or state.shape != (action.shape[0], 16):
            raise ValueError("Bagel training requires action [B,32,16] and current state [B,16]")
        if not torch.isfinite(action).all() or not torch.isfinite(state).all():
            raise ValueError("Bagel state/actions must be finite")
        video = self._training_video(batch).to(self.model.device)
        tasks = self._tasks(batch, action.shape[0])
        context, mask = self.model.encode_prompt([DEFAULT_PROMPT.format(task=task) for task in tasks])
        image_pad = torch.zeros((action.shape[0], 17), dtype=torch.bool, device=video.device)
        for key in self.config.camera_keys:
            if f"{key}_is_pad" in batch:
                image_pad |= batch[f"{key}_is_pad"].to(video.device)
        # Clamped history (including the episode anchor) is valid conditioning.
        image_pad[:, :9] = False
        sample = {
            "video": video,
            "condition_video": video[:, :, :9].clone(),
            "video_num_frames": 17,
            "context": context,
            "context_mask": mask,
            "proprio": (state.to(self._state_scale) * self._state_scale + self._state_offset)
            .clamp(-5, 5)
            .unsqueeze(1),
            "action": (action.to(self._action_scale) * self._action_scale + self._action_offset).clamp(-5, 5),
            "action_is_pad": batch.get("action_is_pad"),
            "image_is_pad": image_pad,
        }
        return self.model.training_loss(sample)
