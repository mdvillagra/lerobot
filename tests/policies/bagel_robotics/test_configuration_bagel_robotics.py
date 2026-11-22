# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import (
    BagelRoboticsConfig,
    get_policy_class,
    make_policy_config,
    make_pre_post_processors,
)


def config(**kwargs):
    cfg = BagelRoboticsConfig(device="cpu", **kwargs)
    cfg.input_features = {
        **{key: PolicyFeature(FeatureType.VISUAL, (3, 12, 16)) for key in cfg.camera_keys},
        "observation.state": PolicyFeature(FeatureType.STATE, (16,)),
    }
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (16,))}
    return cfg


def test_config_factory_and_roundtrip(tmp_path):
    cfg = config()
    assert isinstance(make_policy_config("bagel_robotics", device="cpu"), BagelRoboticsConfig)
    assert get_policy_class("bagel_robotics").config_class is BagelRoboticsConfig
    assert cfg.push_to_hub is False
    cfg.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    cfg.pretrained_path = tmp_path
    assert loaded == cfg
    loaded.validate_features()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 16},
        {"n_action_steps": 32},
        {"n_obs_steps": 2},
        {"camera_keys": ("observation.images.a",) * 3},
        {"num_inference_steps": 0},
        {"sigma_shift": 0},
        {"inference_seed": -1},
        {"dtype": None},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_features_reject_joint_actions_and_missing_camera():
    cfg = config()
    cfg.validate_features()
    cfg.output_features["action"] = PolicyFeature(FeatureType.ACTION, (6,))
    with pytest.raises(ValueError, match="EE16"):
        cfg.validate_features()
    cfg = config()
    del cfg.input_features[cfg.camera_keys[-1]]
    with pytest.raises(ValueError, match="three configured"):
        cfg.validate_features()


def test_processors_preserve_physical_coordinates_and_rgb(tmp_path):
    cfg = config()
    pre, post = make_pre_post_processors(cfg, dataset_stats={"action": {"mean": torch.full((16,), 999.0)}})
    batch = {
        **{key: torch.full((3, 12, 16), 0.25) for key in cfg.camera_keys},
        "observation.state": torch.arange(16).float(),
        "task": "pick up the board",
    }
    result = pre(batch)
    torch.testing.assert_close(result["observation.state"], batch["observation.state"].unsqueeze(0))
    assert result[cfg.camera_keys[0]].shape == (1, 3, 12, 16)
    action = torch.randn(1, 32, 16)
    torch.testing.assert_close(post(action), action)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    pre2, post2 = make_pre_post_processors(cfg, pretrained_path=tmp_path)
    torch.testing.assert_close(pre2(batch)["observation.state"], result["observation.state"])
    torch.testing.assert_close(post2(action), action)


def test_import_does_not_load_optional_backbone():
    code = "from lerobot.policies import BagelRoboticsConfig; import sys; assert 'bagel_robotics.inference.policy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
