# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download

from lerobot.utils.import_utils import _bagel_robotics_available, require_package

from .configuration_bagel_robotics import BagelRoboticsConfig

if TYPE_CHECKING or _bagel_robotics_available:
    from bagel_robotics.inference.assets import MANIFEST, verify_bundle, verify_wan


def resolve_bagel_assets(config: BagelRoboticsConfig) -> tuple[Path, Path]:
    """Resolve and verify the exact public cp1000 bundle and auxiliary assets.

    Reuse the immutable manifest shipped with the pinned backbone package. Only
    its six release files and separately pinned encoders/tokenizer are fetched.
    """
    require_package("bagel-robotics", extra="bagel_robotics", import_name="bagel_robotics")
    if not config.download_assets and (not config.bundle_path or not config.wan_model_dir):
        raise ValueError("download_assets=False requires bundle_path and wan_model_dir")
    bundle = (
        Path(config.bundle_path).expanduser().resolve()
        if config.bundle_path
        else Path(
            snapshot_download(
                MANIFEST["repo_id"],
                revision=MANIFEST["revision"],
                allow_patterns=list(MANIFEST["files"]),
                local_files_only=config.local_files_only,
            )
        )
    )
    prefix = "checkpoints/Wan2.1-I2V-14B-480P"
    wan = (
        Path(config.wan_model_dir).expanduser().resolve()
        if config.wan_model_dir
        else Path(
            snapshot_download(
                MANIFEST["wan"]["repo"],
                revision=MANIFEST["wan"]["revision"],
                allow_patterns=[entry["rfilename"] for entry in MANIFEST["wan"]["files"]],
                local_files_only=config.local_files_only,
            )
        )
        / prefix
    )
    verify_bundle(bundle)
    verify_wan(wan)
    return bundle, wan
