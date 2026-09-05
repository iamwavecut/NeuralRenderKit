#!/usr/bin/env python3
"""Run a validated external neural-rendering golden bundle through MLX-DLSS."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import compare_neural_rendering_golden_bundle as golden


def build_arguments(
    bundle: pathlib.Path,
    model: pathlib.Path,
    output: pathlib.Path,
    executable: pathlib.Path,
) -> list[str]:
    height, width, motion_convention, frames, jitter_deltas = (
        golden._validated_frames(bundle)
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    arguments = [
        str(executable),
        "run-sequence",
        str(model),
        "--input-format",
        "rgb-temporal-reference",
    ]
    if motion_convention == "pixel-current-to-previous":
        if manifest["motionWidth"] != width or manifest["motionHeight"] != height:
            raise ValueError(
                "candidate runner requires full-rect motionWidth/Height equal to "
                "the processing size"
            )
        arguments += [
            "--motion-format",
            "pixel",
            "--motion-scale-x",
            str(float(manifest["motionScaleX"])),
            "--motion-scale-y",
            str(float(manifest["motionScaleY"])),
        ]
    arguments += [
        "--height",
        str(height),
        "--width",
        str(width),
        "--depth-inverted",
        str(manifest["depthInverted"]).lower(),
        "--output-dir",
        str(output),
    ]
    for frame, jitter_delta in zip(frames, jitter_deltas, strict=True):
        arguments += [
            "--input",
            str(frame["color"]),
            "--motion",
            str(frame["motion"]),
            "--depth",
            str(frame["depth"]),
        ]
        if motion_convention == "pixel-current-to-previous":
            arguments += [
                "--jitter-delta-x",
                str(jitter_delta[0]),
                "--jitter-delta-y",
                str(jitter_delta[1]),
            ]
    return arguments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a private NVIDIA golden bundle through MLX-DLSS."
    )
    parser.add_argument("bundle", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--executable",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / ".build/release/mlxdlss",
    )
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    try:
        arguments = build_arguments(
            options.bundle,
            options.model,
            options.output,
            options.executable,
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if options.dry_run:
        print(json.dumps({"arguments": arguments}))
        return 0
    return subprocess.run(arguments, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
