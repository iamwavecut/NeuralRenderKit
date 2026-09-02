#!/usr/bin/env python3
"""Create a validated manifest for an in-place private NVIDIA raw capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re

import compare_neural_rendering_golden_bundle as golden


_CHANNELS = {"color": 3, "motion": 2, "depth": 1, "output": 3}
_OUTPUT_NAME = re.compile(r"output-([0-9]{6})[.]f32")


def _boolean(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _jitter(value: str) -> list[float]:
    fields = value.split(",")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("jitter delta must be X,Y")
    try:
        result = [float(fields[0]), float(fields[1])]
    except ValueError as error:
        raise argparse.ArgumentTypeError("jitter delta must be X,Y") from error
    if not all(math.isfinite(field) for field in result):
        raise argparse.ArgumentTypeError("jitter delta must be finite")
    return result


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_files(
    bundle: pathlib.Path,
    height: int,
    width: int,
) -> list[dict[str, dict[str, str]]]:
    indices = sorted(
        int(match.group(1))
        for path in bundle.iterdir()
        if path.is_file() and (match := _OUTPUT_NAME.fullmatch(path.name))
    )
    if not indices:
        raise ValueError("capture has no output-NNNNNN.f32 files")
    if indices != list(range(len(indices))):
        raise ValueError("capture frame indices must be contiguous from 000000")

    frames = []
    for index in indices:
        frame = {}
        for role, channels in _CHANNELS.items():
            path = bundle / f"{role}-{index:06d}.f32"
            if path.is_symlink():
                raise ValueError(f"capture file must not be a symlink: {path.name}")
            if not path.is_file():
                raise FileNotFoundError(f"missing capture file: {path.name}")
            expected = height * width * channels * 4
            if path.stat().st_size != expected:
                raise ValueError(
                    f"{path.name} byte count must be {expected}, "
                    f"actual {path.stat().st_size}"
                )
            frame[role] = {"file": path.name, "sha256": _digest(path)}
        frames.append(frame)
    return frames


def create_manifest(options: argparse.Namespace) -> dict[str, object]:
    bundle = options.bundle.resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"bundle is not a directory: {bundle}")
    manifest_path = bundle / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    frames = _capture_files(bundle, options.height, options.width)
    manifest: dict[str, object] = {
        "schemaVersion": 1,
        "height": options.height,
        "width": options.width,
        "motionConvention": options.motion_convention,
        "depthInverted": options.depth_inverted,
        "frames": frames,
    }
    if options.motion_convention == "pixel-current-to-previous":
        for name in ("motion_scale_x", "motion_scale_y"):
            value = getattr(options, name)
            if value is None or not math.isfinite(value):
                raise ValueError(f"--{name.replace('_', '-')} must be finite")
        if options.motion_width is None or options.motion_height is None:
            raise ValueError("pixel motion requires --motion-width and --motion-height")
        jitter = options.jitter_delta or [[0.0, 0.0]]
        if len(jitter) not in (1, len(frames)):
            raise ValueError("provide one --jitter-delta or one per frame")
        if len(jitter) == 1:
            jitter = jitter * len(frames)
        manifest.update(
            {
                "motionScaleX": options.motion_scale_x,
                "motionScaleY": options.motion_scale_y,
                "motionWidth": options.motion_width,
                "motionHeight": options.motion_height,
            }
        )
        for frame, delta in zip(frames, jitter, strict=True):
            frame["jitterDeltaPixels"] = delta
    elif any(
        value is not None
        for value in (
            options.motion_scale_x,
            options.motion_scale_y,
            options.motion_width,
            options.motion_height,
        )
    ) or options.jitter_delta:
        raise ValueError("pixel motion metadata requires pixel-current-to-previous")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hash and package an in-place private DLSS NR raw capture."
    )
    parser.add_argument("bundle", type=pathlib.Path)
    parser.add_argument("--height", type=_positive, required=True)
    parser.add_argument("--width", type=_positive, required=True)
    parser.add_argument(
        "--motion-convention",
        choices=sorted(golden.MOTION_CONVENTIONS),
        required=True,
    )
    parser.add_argument("--depth-inverted", type=_boolean, required=True)
    parser.add_argument("--motion-scale-x", type=float)
    parser.add_argument("--motion-scale-y", type=float)
    parser.add_argument("--motion-width", type=_positive)
    parser.add_argument("--motion-height", type=_positive)
    parser.add_argument("--jitter-delta", action="append", type=_jitter)
    options = parser.parse_args()
    try:
        manifest = create_manifest(options)
        path = options.bundle.resolve() / "manifest.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        try:
            golden._validated_frames(options.bundle.resolve())
        except Exception:
            path.unlink(missing_ok=True)
            raise
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"frameCount": len(manifest["frames"]), "manifest": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
