# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
)
from lerobot.datasets.utils import delta_window
from lerobot.policies import ACTConfig, BagelRoboticsConfig
from tests.fixtures.constants import DUMMY_MOTOR_FEATURES

pytestmark = pytest.mark.dataset


@pytest.fixture(scope="module")
def valid_delta_timestamps_factory():
    def _create_valid_delta_timestamps(
        fps: int = 30, keys: list = DUMMY_MOTOR_FEATURES, min_max_range: tuple[int, int] = (-10, 10)
    ) -> dict:
        delta_timestamps = {key: [i * (1 / fps) for i in range(*min_max_range)] for key in keys}
        return delta_timestamps

    return _create_valid_delta_timestamps


@pytest.fixture(scope="module")
def invalid_delta_timestamps_factory(valid_delta_timestamps_factory):
    def _create_invalid_delta_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4, keys: list = DUMMY_MOTOR_FEATURES
    ) -> dict:
        delta_timestamps = valid_delta_timestamps_factory(fps, keys)
        # Modify a single timestamp just outside tolerance
        for key in keys:
            delta_timestamps[key][3] += tolerance_s * 1.1
        return delta_timestamps

    return _create_invalid_delta_timestamps


@pytest.fixture(scope="module")
def slightly_off_delta_timestamps_factory(valid_delta_timestamps_factory):
    def _create_slightly_off_delta_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4, keys: list = DUMMY_MOTOR_FEATURES
    ) -> dict:
        delta_timestamps = valid_delta_timestamps_factory(fps, keys)
        # Modify a single timestamp just inside tolerance
        for key in delta_timestamps:
            delta_timestamps[key][3] += tolerance_s * 0.9
            delta_timestamps[key][-3] += tolerance_s * 0.9
        return delta_timestamps

    return _create_slightly_off_delta_timestamps


@pytest.fixture(scope="module")
def delta_indices_factory():
    def _delta_indices(keys: list = DUMMY_MOTOR_FEATURES, min_max_range: tuple[int, int] = (-10, 10)) -> dict:
        return {key: list(range(*min_max_range)) for key in keys}

    return _delta_indices


def test_check_delta_timestamps_valid(valid_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    valid_delta_timestamps = valid_delta_timestamps_factory(fps)
    result = check_delta_timestamps(
        delta_timestamps=valid_delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_delta_timestamps_slightly_off(slightly_off_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    slightly_off_delta_timestamps = slightly_off_delta_timestamps_factory(fps, tolerance_s)
    result = check_delta_timestamps(
        delta_timestamps=slightly_off_delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_delta_timestamps_invalid(invalid_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    invalid_delta_timestamps = invalid_delta_timestamps_factory(fps, tolerance_s)
    with pytest.raises(ValueError):
        check_delta_timestamps(
            delta_timestamps=invalid_delta_timestamps,
            fps=fps,
            tolerance_s=tolerance_s,
        )


def test_check_delta_timestamps_invalid_no_exception(invalid_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    invalid_delta_timestamps = invalid_delta_timestamps_factory(fps, tolerance_s)
    result = check_delta_timestamps(
        delta_timestamps=invalid_delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
        raise_value_error=False,
    )
    assert result is False


def test_check_delta_timestamps_empty():
    delta_timestamps = {}
    fps = 30
    tolerance_s = 1e-4
    result = check_delta_timestamps(
        delta_timestamps=delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_delta_indices(valid_delta_timestamps_factory, delta_indices_factory):
    fps = 50
    min_max_range = (-100, 100)
    delta_timestamps = valid_delta_timestamps_factory(fps, min_max_range=min_max_range)
    expected_delta_indices = delta_indices_factory(min_max_range=min_max_range)
    actual_delta_indices = get_delta_indices(delta_timestamps, fps)
    assert expected_delta_indices == actual_delta_indices


@pytest.mark.parametrize("start", [0, 2000], ids=["episode-relative", "dataset-absolute"])
@pytest.mark.parametrize("frame", [0, 1, 176, 999])
def test_bagel_episode_anchor_and_relative_frames(start, frame):
    cfg = BagelRoboticsConfig(device="cpu")
    camera = "observation.images.front"
    meta = SimpleNamespace(
        fps=25, total_frames=5000, features=dict.fromkeys((camera, "observation.state", "action"))
    )
    deltas = resolve_delta_timestamps(cfg, meta, {camera: cfg.camera_keys[0]})
    indices = get_delta_indices(deltas, meta.fps)
    assert set(indices) == {camera, "action"}
    assert indices["action"] == list(range(32))
    rows, pad = delta_window(start + frame, indices[camera], start, start + 1000)
    history = [start + max(0, frame + offset) for offset in range(-175, 1, 25)]
    future = [start + min(999, frame + offset) for offset in range(4, 33, 4)]
    assert rows == [start, *history, *future]
    assert pad[0]


def test_act_has_no_image_or_state_window():
    meta = SimpleNamespace(fps=25, features={"observation.images.front": {}, "observation.state": {}})
    assert resolve_delta_timestamps(ACTConfig(device="cpu"), meta) is None
