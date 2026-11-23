# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import DEFAULT, patch

import pytest

pytest.importorskip("bagel_robotics")

from lerobot.policies import BagelRoboticsConfig
from lerobot.policies.bagel_robotics import assets_bagel_robotics as assets


@pytest.mark.parametrize("local", [True, False])
def test_assets_use_local_paths_or_pinned_offline_cache(tmp_path, local):
    bundle, wan = tmp_path / "bundle", tmp_path / "wan/checkpoints/Wan2.1-I2V-14B-480P"
    paths = {"bundle_path": str(bundle), "wan_model_dir": str(wan)} if local else {}
    cfg = BagelRoboticsConfig(device="cpu", local_files_only=True, download_assets=not local, **paths)
    with patch.multiple(assets, snapshot_download=DEFAULT, verify_bundle=DEFAULT, verify_wan=DEFAULT) as io:
        download = io["snapshot_download"]
        download.side_effect = [str(bundle), str(tmp_path / "wan")]
        assert assets.resolve_bagel_assets(cfg) == (bundle, wan)
        io["verify_bundle"].assert_called_once_with(bundle)
        io["verify_wan"].assert_called_once_with(wan)
        if local:
            download.assert_not_called()
        else:
            release, encoders = download.call_args_list
            assert release.args == ("bageldotcom/Bagel-Robotics-1",)
            assert len(release.kwargs["revision"]) == 40
            assert release.kwargs["local_dir"].name == release.kwargs["revision"]
            assert set(release.kwargs["allow_patterns"]) == set(assets.MANIFEST["files"])
            assert all(call.kwargs["local_files_only"] for call in (release, encoders))
            assert not any("diffusion_pytorch_model" in name for name in encoders.kwargs["allow_patterns"])


def test_missing_paths_and_corrupt_assets_stop_loading(tmp_path):
    cfg = BagelRoboticsConfig(device="cpu", download_assets=False)
    with pytest.raises(ValueError, match="requires bundle_path"):
        assets.resolve_bagel_assets(cfg)
    cfg.bundle_path = cfg.wan_model_dir = str(tmp_path)
    with (
        patch.object(assets, "verify_bundle", side_effect=ValueError("SHA-256 mismatch")),
        pytest.raises(ValueError, match="SHA-256 mismatch"),
    ):
        assets.resolve_bagel_assets(cfg)
