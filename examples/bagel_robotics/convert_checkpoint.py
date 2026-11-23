#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export Bagel cp1000 locally in LeRobot format, using feature shapes from a matching dataset."""

import argparse
import json
from pathlib import Path

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import BagelRoboticsConfig, make_policy, make_pre_post_processors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bundle-path")
    parser.add_argument("--wan-model-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--rename-map", type=json.loads, default={})
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be absent or an empty directory")
    config = BagelRoboticsConfig(
        device=args.device,
        bundle_path=args.bundle_path,
        wan_model_dir=args.wan_model_dir,
        local_files_only=args.local_files_only,
    )
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    policy = make_policy(config, ds_meta=metadata, rename_map=args.rename_map)
    preprocessor, postprocessor = make_pre_post_processors(config)
    preprocessor.steps[0].rename_map = args.rename_map
    policy.save_pretrained(args.output)
    for processor in (preprocessor, postprocessor):
        processor.save_pretrained(args.output)
    print(f"Saved Bagel-Robotics-1 locally to {args.output.resolve()}")


if __name__ == "__main__":
    main()
