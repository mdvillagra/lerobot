# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("bagel_robotics")

from lerobot.policies import BagelRoboticsConfig
from lerobot.policies.bagel_robotics import assets_bagel_robotics as assets


def test_explicit_local_assets_are_verified_without_network(monkeypatch, tmp_path):
    calls = []

    def unexpected_download(*args, **kwargs):
        pytest.fail("Explicit local assets must not download anything")

    monkeypatch.setattr(assets, "snapshot_download", unexpected_download)
    monkeypatch.setattr(assets, "verify_bundle", lambda path: calls.append(("bundle", path)))
    monkeypatch.setattr(assets, "verify_wan", lambda path: calls.append(("wan", path)))
    cfg = BagelRoboticsConfig(
        device="cpu",
        bundle_path=str(tmp_path / "cp1000"),
        wan_model_dir=str(tmp_path / "wan"),
        download_assets=False,
    )
    bundle, wan = assets.resolve_bagel_assets(cfg)
    assert calls == [("bundle", bundle), ("wan", wan)]


def test_offline_cache_requests_only_the_pinned_release(monkeypatch, tmp_path):
    requests = []

    def cached(repo_id, **kwargs):
        requests.append((repo_id, kwargs))
        return str(tmp_path / str(len(requests)))

    monkeypatch.setattr(assets, "snapshot_download", cached)
    monkeypatch.setattr(assets, "verify_bundle", lambda path: None)
    monkeypatch.setattr(assets, "verify_wan", lambda path: None)
    bundle, wan = assets.resolve_bagel_assets(BagelRoboticsConfig(device="cpu", local_files_only=True))
    assert bundle == tmp_path / "1"
    assert wan == tmp_path / "2/checkpoints/Wan2.1-I2V-14B-480P"
    assert requests[0][0] == "bageldotcom/Bagel-Robotics-1"
    assert len(requests[0][1]["revision"]) == 40
    assert set(requests[0][1]["allow_patterns"]) == set(assets.MANIFEST["files"])
    assert all(request[1]["local_files_only"] for request in requests)
    assert not any("diffusion_pytorch_model" in name for name in requests[1][1]["allow_patterns"])


def test_missing_local_paths_or_corrupt_bundle_stop_loading(monkeypatch, tmp_path):
    cfg = BagelRoboticsConfig(device="cpu", download_assets=False)
    with pytest.raises(ValueError, match="requires bundle_path"):
        assets.resolve_bagel_assets(cfg)

    def corrupt(path):
        raise ValueError("Checkpoint SHA-256 mismatch: action.pt")

    monkeypatch.setattr(assets, "verify_bundle", corrupt)
    cfg.bundle_path = cfg.wan_model_dir = str(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        assets.resolve_bagel_assets(cfg)
