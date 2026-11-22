# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets")

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.utils import delta_window
from lerobot.policies import ACTConfig, BagelRoboticsConfig

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("frame", [0, 1, 176, 999])
def test_bagel_anchor_and_history_match_native_indices(streaming, frame):
    cfg = BagelRoboticsConfig(device="cpu")
    meta = SimpleNamespace(
        total_frames=5000,
        fps=25,
        features={
            "observation.images.front": {},
            "observation.state": {},
            "action": {},
        },
    )
    deltas = resolve_delta_timestamps(cfg, meta, {"observation.images.front": cfg.camera_keys[0]})
    indices = [round(x * meta.fps) for x in deltas["observation.images.front"]]
    start = 0 if streaming else 2000
    rows, pad = delta_window(start + frame, indices, start, start + 1000)
    expected = [
        0,
        *[max(0, frame + o) for o in range(-175, 1, 25)],
        *[min(999, frame + o) for o in range(4, 33, 4)],
    ]
    assert rows == [start + i for i in expected]
    assert len(rows) == 17
    assert bool(pad[0])  # The policy treats clamped history as valid conditioning.
    assert "observation.state" not in deltas
    assert len(deltas["action"]) == 32


def test_other_policies_keep_their_observation_windows():
    cfg = ACTConfig(device="cpu")
    meta = SimpleNamespace(fps=25, features={"observation.images.front": {}, "observation.state": {}})
    deltas = resolve_delta_timestamps(cfg, meta)
    assert deltas is None
    rows, pad = delta_window(2005, [0], 2000, 3000)
    assert rows == [2005]
    torch.testing.assert_close(pad, torch.tensor([False]))
