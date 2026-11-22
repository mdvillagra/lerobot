# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests: real native history, ensemble and reduced MoT training math.

Only checkpoint IO and the expensive VAE/CLIP/text encoders are replaced. The
training loss, video/action experts, and cross-expert gradients are upstream code.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

pytest.importorskip("bagel_robotics")
pytest.importorskip("transformers")
pytest.importorskip("hydra")

from bagel_robotics.inference.datasets.lerobot.utils.normalizer import LinearNormalizer
from bagel_robotics.inference.policy import RoboDojoVPP2Policy, compose_training_condition
from bagel_robotics.models.backbone.wan21_14b.action_dit import ActionDiT
from bagel_robotics.models.backbone.wan21_14b.mot import MoT
from bagel_robotics.models.backbone.wan21_14b.vpp2 import VPP2
from bagel_robotics.models.backbone.wan21_14b.wan_video_dit import WanVideoDiT

import lerobot.policies.bagel_robotics.modeling_bagel_robotics as modeling
from lerobot.policies import make_pre_post_processors
from tests.policies.bagel_robotics.test_configuration_bagel_robotics import config


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4

    def encode(self, videos, **kwargs):
        return [
            functional.adaptive_avg_pool3d(video[:2], ((video.shape[1] - 1) // 4 + 1, 4, 4))
            for video in videos
        ]


class TinyCLIP(nn.Module):
    def forward(self, videos):
        return videos[0].new_zeros((len(videos), 257, 1280))


class TinyBackbone(VPP2):
    def __init__(self):
        video = WanVideoDiT(
            hidden_dim=16,
            ffn_dim=32,
            freq_dim=8,
            text_dim=8,
            in_dim=8,
            out_dim=2,
            num_heads=2,
            attn_head_dim=8,
            num_layers=1,
        )
        action = ActionDiT(
            hidden_dim=16,
            action_dim=16,
            ffn_dim=32,
            text_dim=8,
            freq_dim=8,
            eps=1e-6,
            num_heads=2,
            attn_head_dim=8,
            num_layers=1,
        )
        super().__init__(
            video,
            action,
            MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False),
            TinyVAE(),
            text_encoder=nn.Linear(1, 8),
            clip_encoder=TinyCLIP(),
            text_dim=8,
            proprio_dim=16,
            torch_dtype=torch.float32,
        )
        self.inference_calls = []

    def encode_prompt(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        return torch.zeros(len(prompts), 2, 8), torch.ones(len(prompts), 2, dtype=torch.bool)

    def infer_action(self, **kwargs):
        # Sampling itself needs the real giant assets; retain all native policy
        # normalization/history/seeding and only replace the model prediction.
        self.inference_calls.append(kwargs)
        action = torch.full((1, 32, 16), float(kwargs["seed"]))
        action[:, :, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        action[:, :, 11:15] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        return {"action": action}

    def training_loss(self, sample, **kwargs):
        self.last_sample = sample
        return super().training_loss(sample, **kwargs)


class OfflineRuntime(RoboDojoVPP2Policy):
    def __init__(self, **kwargs):
        self.model = TinyBackbone()
        shape_meta = {field: [{"key": "default", "shape": 16}] for field in ("action", "state")}
        stats = {
            field: {"default": {"global_mean": torch.zeros(16), "global_std": torch.ones(16)}}
            for field in ("action", "state")
        }
        norm = LinearNormalizer(shape_meta, False, "z-score", None, stats)
        self.processor = SimpleNamespace(
            shape_meta=shape_meta, normalizer=norm, action_state_transform=lambda x: x
        )
        self.output_action_dim = self.action_horizon = 16
        self.action_horizon = 32
        self.num_inference_steps = kwargs["num_inference_steps"]
        self.sigma_shift = kwargs["sigma_shift"]
        self.base_seed = kwargs["seed"]
        self.video_seed_offset = 1000003
        self.text_cfg_scale, self.negative_prompt, self.rand_device = 1.0, "", "cpu"
        self.tiled = False
        self.video_size, self.visual_layout, self.per_camera_video_size = (240, 416), "robodojo_tshape", None
        self.num_video_frames, self.history_condition_frames, self.history_stride = 17, 8, 25
        self.history_buffer_size = 177
        self.visual_planning_output_root = None
        self.visual_planning_full_steps = 10
        for name in self._PER_ENV_STATE:
            setattr(self, name, {})
        self._inference_index = 0


@pytest.fixture
def policy(monkeypatch, tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(modeling, "RoboDojoVPP2Policy", OfflineRuntime)
    monkeypatch.setattr(modeling, "resolve_bagel_assets", lambda cfg: (tmp_path, tmp_path))
    try:
        yield modeling.BagelRoboticsPolicy(config(dtype=torch.float32))
    finally:
        torch.set_num_threads(previous_threads)


def batch(cfg, temporal=False):
    dims = (1, 17) if temporal else (1,)
    result = {key: torch.rand(*dims, 3, 12, 16) for key in cfg.camera_keys}
    result.update({"observation.state": torch.zeros(1, 16), "task": ["lift the board"]})
    if temporal:
        result["action"] = torch.rand(1, 32, 16)
        result["action_is_pad"] = torch.zeros(1, 32, dtype=torch.bool)
        for key in cfg.camera_keys:
            result[f"{key}_is_pad"] = torch.zeros(1, 17, dtype=torch.bool)
    return result


def test_training_camera_composition_matches_native(policy):
    data = batch(policy.config, temporal=True)
    result = policy._training_video(data)
    for frame in (0, 8, 16):
        images = [data[key][0, frame].permute(1, 2, 0).numpy() for key in policy.config.camera_keys]
        expected = compose_training_condition(*images)
        torch.testing.assert_close(result[:, :, frame], expected, rtol=0, atol=0)


def test_native_history_records_queued_ticks_and_seeds(policy):
    data = batch(policy.config)
    for _tick in range(177):
        action = policy.select_action(data)
        assert action.shape == (1, 16)
    calls = policy.model.inference_calls
    assert len(calls) == 12
    assert [call["seed"] for call in calls] == list(range(1, 13))
    assert calls[-1]["condition_video"].shape == (1, 3, 9, 240, 416)
    _, info = policy._runtime._build_history_condition(policy._runtime._latest_image_by_env[0], 0)
    assert info["current_step"] == 176
    assert info["selected_steps"] == [0, 1, 26, 51, 76, 101, 126, 151, 176]


def test_temporal_blending_and_reset(policy):
    data = batch(policy.config)
    first = policy.select_action(data)
    for _ in range(15):
        policy.select_action(data)
    second = policy.select_action(data)
    assert first[0, 0].item() == pytest.approx(1.0)
    assert second[0, 0].item() == pytest.approx(1.9)
    torch.testing.assert_close(second[:, 3:7].norm(dim=-1), torch.ones(1))
    policy.reset()
    torch.testing.assert_close(policy.select_action(data), first)
    assert policy._runtime._history_step_by_env == {0: 0}


def test_instruction_change_resets_queue_history_and_seed(policy):
    data = batch(policy.config)
    policy.select_action(data)
    data["task"] = ["lower the board"]
    policy.select_action(data)
    assert len(policy.model.inference_calls) == 2
    assert policy.model.inference_calls[-1]["seed"] == 1
    assert policy._runtime._history_step_by_env == {0: 0}
    policy.drop_queued_actions()
    assert not policy._action_queue
    assert not policy._runtime._history_by_env


def test_direct_chunk_and_remote_contract(policy):
    chunk = policy.predict_action_chunk(batch(policy.config))
    assert chunk.shape == (1, 32, 16)
    assert not policy._action_queue
    with pytest.raises(ValueError, match="temporal history"):
        policy.chunk_inference_spec()


def test_real_native_euler_sampling_is_finite_and_reproducible(policy):
    # Run the actual cached-video/Euler sampler with the reduced experts. Only
    # encoders and checkpoint loading remain replaced in this test.
    policy.model.infer_action = VPP2.infer_action.__get__(policy.model)
    data = batch(policy.config)
    first = policy.predict_action_chunk(data)
    assert first.shape == (1, 32, 16)
    assert torch.isfinite(first).all()
    policy.reset()
    torch.testing.assert_close(policy.predict_action_chunk(data), first, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["batch", "task", "rgb", "state", "temporal"])
def test_invalid_inference_inputs(policy, invalid):
    data = batch(policy.config)
    if invalid == "batch":
        data["observation.state"] = torch.zeros(2, 16)
    elif invalid == "task":
        data["task"] = [""]
    elif invalid == "rgb":
        data[policy.config.camera_keys[0]].fill_(2)
    elif invalid == "state":
        data["observation.state"].fill_(float("nan"))
    else:
        data[policy.config.camera_keys[0]] = torch.zeros(1, 17, 3, 12, 16)
    with pytest.raises(ValueError):
        policy.select_action(data)


def test_native_training_backward_frozen_modules_and_padding(policy):
    data = batch(policy.config, temporal=True)
    data["action_is_pad"][:, -2:] = True
    for key in policy.config.camera_keys:
        data[f"{key}_is_pad"][:, :8] = True
        data[f"{key}_is_pad"][:, -1] = True
    policy.train()
    # Isolate action supervision: a video-only loss could hide a detached
    # cross-expert gradient path when the action expert is frozen.
    policy.model.loss_lambda_video = 0
    frozen = {name: p.clone() for name, p in policy.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4)
    loss, metrics = policy(data)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(isinstance(value, float) for value in metrics.values())
    sample = policy.model.last_sample
    assert sample["proprio"].shape == (1, 1, 16)
    assert not sample["image_is_pad"][:, :9].any()
    assert sample["image_is_pad"][:, -1].all()
    torch.testing.assert_close(sample["condition_video"], sample["video"][:, :, :9])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.model.video_expert.parameters())
    assert all(p.grad is None for p in policy.model.action_expert.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    optimizer.step()
    assert all(torch.equal(frozen[name], p) for name, p in policy.named_parameters() if name in frozen)
    assert not policy.model.text_encoder.training
    assert not policy.model.action_expert.training
    with pytest.raises(ValueError, match="mean"):
        policy(data, reduction="none")


def test_unfreezing_action_expert(policy, monkeypatch, tmp_path):
    cfg = config(dtype=torch.float32, freeze_action_expert=False)
    unfrozen = modeling.BagelRoboticsPolicy(cfg)
    assert all(p.requires_grad for p in unfrozen.model.action_expert.parameters())
    assert all(p.requires_grad for p in unfrozen.model.proprio_encoder.parameters())
    assert not any(p.requires_grad for p in unfrozen.model.text_encoder.parameters())


def test_training_rejects_non_native_frame_rate_before_loading(policy):
    with pytest.raises(ValueError, match="25-Hz"):
        modeling.BagelRoboticsPolicy(config(dtype=torch.float32), dataset_meta=SimpleNamespace(fps=30))


def test_checkpoint_and_processors_roundtrip(policy, tmp_path):
    data = batch(policy.config)
    policy.eval()
    expected = policy.predict_action_chunk(data)
    policy.save_pretrained(tmp_path)
    pre, post = make_pre_post_processors(policy.config)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded = modeling.BagelRoboticsPolicy.from_pretrained(tmp_path)
    torch.testing.assert_close(loaded.predict_action_chunk(data), expected)
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor)
    assert not loaded.training
