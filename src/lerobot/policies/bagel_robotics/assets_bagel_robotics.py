# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import _bagel_robotics_available, require_package

from .configuration_bagel_robotics import BagelRoboticsConfig

if TYPE_CHECKING or _bagel_robotics_available:
    from bagel_robotics.inference.assets import MANIFEST, verify_bundle, verify_wan


def resolve_bagel_assets(config: BagelRoboticsConfig) -> tuple[Path, Path]:
    """Resolve local assets or download the pinned release, then verify its hashes."""
    require_package("bagel-robotics", extra="bagel_robotics", import_name="bagel_robotics")
    if not config.download_assets and (not config.bundle_path or not config.wan_model_dir):
        raise ValueError("download_assets=False requires bundle_path and wan_model_dir")
    bundle = config.bundle_path or snapshot_download(
        MANIFEST["repo_id"],
        revision=MANIFEST["revision"],
        # The native loader needs architecture metadata beside the resolved video.pt.
        local_dir=HF_LEROBOT_HOME / "bagel_robotics" / MANIFEST["revision"],
        allow_patterns=list(MANIFEST["files"]),
        local_files_only=config.local_files_only,
    )
    wan = config.wan_model_dir or str(
        Path(
            snapshot_download(
                MANIFEST["wan"]["repo"],
                revision=MANIFEST["wan"]["revision"],
                allow_patterns=[entry["rfilename"] for entry in MANIFEST["wan"]["files"]],
                local_files_only=config.local_files_only,
            )
        )
        / "checkpoints/Wan2.1-I2V-14B-480P"
    )
    bundle_path, wan_path = (Path(path).expanduser().resolve() for path in (bundle, wan))
    verify_bundle(bundle_path)
    verify_wan(wan_path)
    return bundle_path, wan_path
