# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks with native history/ensemble/MoT math and lightweight encoder/IO substitutes."""

from types import SimpleNamespace
from unittest.mock import Mock

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
        dimensions = {
            "hidden_dim": 16,
            "ffn_dim": 32,
            "freq_dim": 8,
            "text_dim": 8,
            "num_heads": 2,
            "attn_head_dim": 8,
            "num_layers": 1,
        }
        video = WanVideoDiT(in_dim=8, out_dim=2, **dimensions)
        action = ActionDiT(action_dim=16, eps=1e-6, **dimensions)
        super().__init__(
            video,
            action,
            MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False),
            TinyVAE(),
            text_encoder=nn.Linear(1, 8),
            clip_encoder=TinyCLIP(),
            text_dim=8,
            proprio_dim=16,
        )

    def encode_prompt(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        return torch.zeros(len(prompts), 2, 8), torch.ones(len(prompts), 2, dtype=torch.bool)


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
        self.output_action_dim = 16
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
    return result


@pytest.fixture
def replans(policy, monkeypatch):
    sampler = Mock(wraps=policy.model.infer_action)
    monkeypatch.setattr(policy.model, "infer_action", sampler)
    return sampler


def test_history_blending_and_replan_seeds(policy, replans):
    data = batch(policy.config)
    for tick in range(177):
        action = policy.select_action(data)
        assert action.shape == (1, 16)
        if tick == 0:
            first = policy._ensemble.previous
            torch.testing.assert_close(action[0], torch.from_numpy(first[0]))
        elif tick == 16:
            current = policy._ensemble.previous
            assert action[0, 0].item() == pytest.approx(0.9 * current[0, 0] + 0.1 * first[16, 0])
            for quat in (action[:, 3:7], action[:, 11:15]):
                torch.testing.assert_close(quat.norm(dim=-1), torch.ones(1))
    assert [call.kwargs["seed"] for call in replans.call_args_list] == list(range(1, 13))
    assert replans.call_args.kwargs["condition_video"].shape == (1, 3, 9, 240, 416)
    _, info = policy._runtime._build_history_condition(policy._runtime._latest_image_by_env[0], 0)
    assert info["current_step"] == 176
    assert info["selected_steps"] == [0, 1, 26, 51, 76, 101, 126, 151, 176]


@pytest.mark.parametrize("boundary", ["reset", "drop_queued_actions", "instruction"])
def test_episode_boundaries_clear_history_queue_and_seed(policy, boundary, replans):
    data = batch(policy.config)
    first = policy.select_action(data)
    if boundary == "instruction":
        data["task"] = ["lower the board"]
    else:
        getattr(policy, boundary)()
        assert not policy._action_queue and not policy._runtime._history_by_env
    torch.testing.assert_close(policy.select_action(data), first)
    assert replans.call_args.kwargs["seed"] == 1
    assert policy._runtime._history_step_by_env == {0: 0}


def test_native_euler_sampling_and_checkpoint_roundtrip(policy, tmp_path):
    data = batch(policy.config)
    first = policy.predict_action_chunk(data)
    assert first.shape == (1, 32, 16)
    assert torch.isfinite(first).all()
    assert not policy._action_queue
    with pytest.raises(ValueError, match="temporal history"):
        policy.chunk_inference_spec()
    policy.reset()
    torch.testing.assert_close(policy.predict_action_chunk(data), first, rtol=0, atol=0)
    policy.save_pretrained(tmp_path)
    loaded = modeling.BagelRoboticsPolicy.from_pretrained(tmp_path)
    assert not loaded.training
    torch.testing.assert_close(loaded.state_dict(), policy.state_dict())
    torch.testing.assert_close(loaded.predict_action_chunk(data), first, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("observation.state", torch.zeros(2, 16)),
        ("task", [""]),
        ("observation.images.head_camera", torch.full((1, 3, 12, 16), 2.0)),
        ("observation.images.head_camera", torch.full((1, 3, 12, 16), torch.nan)),
        ("observation.state", torch.full((1, 16), torch.nan)),
        ("observation.images.head_camera", torch.zeros(1, 17, 3, 12, 16)),
    ],
)
def test_invalid_inference_inputs(policy, key, value):
    data = batch(policy.config)
    data[key] = value
    with pytest.raises(ValueError):
        policy.select_action(data)


@pytest.mark.parametrize("method", ["select_action", "predict_action_chunk"])
def test_nonfinite_predictions_are_rejected(policy, monkeypatch, method):
    monkeypatch.setattr(
        policy._runtime, "_infer_action_chunk", lambda *args: torch.full((32, 16), torch.nan).numpy()
    )
    with pytest.raises(ValueError, match="finite"):
        getattr(policy, method)(batch(policy.config))


def test_native_training_backward_frozen_modules_and_padding(policy, monkeypatch):
    data = batch(policy.config, temporal=True)
    data["action_is_pad"] = torch.arange(32).unsqueeze(0) >= 30
    for key in policy.config.camera_keys:
        data[f"{key}_is_pad"] = torch.tensor([[True] * 9 + [False] * 7 + [True]])
    policy.train()
    # Isolate action supervision to check gradients through the frozen action expert.
    policy.model.loss_lambda_video = 0
    frozen = {name: p.clone() for name, p in policy.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4)
    native_loss = Mock(wraps=policy.model.training_loss)
    monkeypatch.setattr(policy.model, "training_loss", native_loss)
    loss, metrics = policy(data)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(isinstance(value, float) for value in metrics.values())
    sample = native_loss.call_args.args[0]
    for frame in (0, 8, 16):
        images = [data[key][0, frame].permute(1, 2, 0).numpy() for key in policy.config.camera_keys]
        torch.testing.assert_close(
            sample["video"][:, :, frame], compose_training_condition(*images), rtol=0, atol=0
        )
    torch.testing.assert_close(sample["action_is_pad"], data["action_is_pad"])
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


def test_unfreezing_action_expert(policy):
    cfg = config(dtype=torch.float32, freeze_action_expert=False)
    unfrozen = modeling.BagelRoboticsPolicy(cfg)
    assert all(p.requires_grad for p in unfrozen.model.action_expert.parameters())
    assert all(p.requires_grad for p in unfrozen.model.proprio_encoder.parameters())
    assert not any(p.requires_grad for p in unfrozen.model.text_encoder.parameters())


def test_training_rejects_non_native_frame_rate_before_loading(policy):
    with pytest.raises(ValueError, match="25-Hz"):
        modeling.BagelRoboticsPolicy(config(dtype=torch.float32), dataset_meta=SimpleNamespace(fps=30))
