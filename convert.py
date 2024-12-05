#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
#

import os
import sys
import logging
from argparse import ArgumentParser
import shutil


def run_command(command, error_message):
    exit_code = os.system(command)
    if exit_code != 0:
        logging.error("%s with code %d. Exiting.", error_message, exit_code)
        sys.exit(exit_code)


def main():
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    args = parser.parse_args()

    colmap_command = "colmap"

    use_gpu = 1 if not args.no_gpu else 0

    if not args.skip_matching:
        os.makedirs(f"{args.source_path}/distorted/sparse", exist_ok=True)

        run_command(
            f"{colmap_command} feature_extractor --database_path {args.source_path}/distorted/database.db "
            f"--image_path {args.source_path}/input --ImageReader.single_camera 1 "
            f"--ImageReader.camera_model {args.camera} --SiftExtraction.use_gpu {use_gpu}",
            "Feature extraction failed",
        )

        run_command(
            f"{colmap_command} exhaustive_matcher --database_path {args.source_path}/distorted/database.db "
            f"--SiftMatching.use_gpu {use_gpu}",
            "Feature matching failed",
        )

        run_command(
            f"{colmap_command} mapper --database_path {args.source_path}/distorted/database.db "
            f"--image_path {args.source_path}/input --output_path {args.source_path}/distorted/sparse "
            f"--Mapper.ba_global_function_tolerance=0.000001",
            "Mapper failed",
        )

    run_command(
        f"{colmap_command} image_undistorter --image_path {args.source_path}/input "
        f"--input_path {args.source_path}/distorted/sparse/0 --output_path {args.source_path} "
        f"--output_type COLMAP",
        "Image undistortion failed",
    )

    os.makedirs(f"{args.source_path}/sparse/0", exist_ok=True)
    for file in os.listdir(f"{args.source_path}/sparse"):
        if file != "0":
            shutil.move(
                os.path.join(args.source_path, "sparse", file),
                os.path.join(args.source_path, "sparse", "0", file),
            )

    print("Done.")


if __name__ == "__main__":
    main()
