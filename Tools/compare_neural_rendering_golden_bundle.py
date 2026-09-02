#!/usr/bin/env python3
"""Validate and compare an external neural-rendering golden bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re

import compare_f32


MOTION_CONVENTIONS = {
    "normalized-history-uv-offset",
    "pixel-current-to-previous",
}
_ROLES = {"color": 3, "motion": 2, "depth": 1, "output": 3}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: pathlib.Path, role: str, record: object) -> pathlib.Path:
    if not isinstance(record, dict):
        raise ValueError(f"{role} record must be an object")
    name = record.get("file")
    if not isinstance(name, str):
        raise ValueError(f"{role} file must be a string")
    relative = pathlib.PurePath(name)
    if relative.is_absolute() or len(relative.parts) != 1 or name in {".", ".."}:
        raise ValueError(f"unsafe {role} file: {name}")
    digest = record.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{role} sha256 must be 64 lowercase hexadecimal digits")
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"missing {role} file: {path}")
    actual = _sha256(path)
    if actual != digest:
        raise ValueError(
            f"SHA-256 mismatch for {role} file {name}: expected {digest}, actual {actual}"
        )
    return path


def _validated_frames(
    bundle: pathlib.Path,
) -> tuple[
    int,
    int,
    str,
    list[dict[str, pathlib.Path]],
    list[list[float]],
]:
    bundle = pathlib.Path(bundle)
    manifest_path = bundle / "manifest.json"
    if not bundle.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"bundle must contain manifest.json: {bundle}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 1:
        raise ValueError("schemaVersion must be 1")
    height = manifest.get("height")
    width = manifest.get("width")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError("height must be a positive integer")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be a positive integer")
    motion_convention = manifest.get("motionConvention")
    if motion_convention not in MOTION_CONVENTIONS:
        raise ValueError(
            "motionConvention must be normalized-history-uv-offset "
            "or pixel-current-to-previous"
        )
    if motion_convention == "pixel-current-to-previous":
        for key in ("motionScaleX", "motionScaleY"):
            value = manifest.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
        for key in ("motionWidth", "motionHeight"):
            value = manifest.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
    if not isinstance(manifest.get("depthInverted"), bool):
        raise ValueError("depthInverted must be a boolean")
    records = manifest.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError("frames must be a non-empty array")

    frames: list[dict[str, pathlib.Path]] = []
    jitter_deltas: list[list[float]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"frame {index} must be an object")
        jitter_delta = record.get("jitterDeltaPixels")
        if motion_convention == "normalized-history-uv-offset":
            if jitter_delta is not None:
                raise ValueError(
                    f"frame {index} jitterDeltaPixels requires pixel motion"
                )
            jitter_deltas.append([0.0, 0.0])
        else:
            if jitter_delta is None:
                jitter_delta = [0.0, 0.0]
            if (
                not isinstance(jitter_delta, list)
                or len(jitter_delta) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in jitter_delta
                )
            ):
                raise ValueError(
                    f"frame {index} jitterDeltaPixels must be two finite numbers"
                )
            jitter_deltas.append([float(jitter_delta[0]), float(jitter_delta[1])])
        frame: dict[str, pathlib.Path] = {}
        for role, channels in _ROLES.items():
            path = _safe_file(bundle, role, record.get(role))
            expected_bytes = height * width * channels * 4
            if path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"{role} byte count for frame {index}: expected "
                    f"{expected_bytes}, actual {path.stat().st_size}"
                )
            frame[role] = path
        frames.append(frame)
    return height, width, motion_convention, frames, jitter_deltas


def compare_bundle(
    bundle: pathlib.Path,
    candidate_directory: pathlib.Path,
) -> dict[str, object]:
    bundle = pathlib.Path(bundle)
    candidate_directory = pathlib.Path(candidate_directory)
    height, width, motion_convention, frames, jitter_deltas = _validated_frames(bundle)
    if not candidate_directory.is_dir():
        raise FileNotFoundError(
            f"candidate output directory does not exist: {candidate_directory}"
        )

    frame_results: list[dict[str, object]] = []
    finite_pairs = 0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    maximum_absolute_error = 0.0
    all_finite = True
    non_finite_reference = 0
    non_finite_candidate = 0
    for index, frame in enumerate(frames):
        candidate = candidate_directory / f"{index:06d}.f32"
        result = compare_f32.compare(frame["output"], candidate)
        count = int(result["finitePairCount"])
        finite_pairs += count
        absolute_error_sum += float(result["meanAbsoluteError"] or 0) * count
        squared_error_sum += float(result["meanSquaredError"] or 0) * count
        maximum_absolute_error = max(
            maximum_absolute_error,
            float(result["maximumAbsoluteError"] or 0),
        )
        all_finite = all_finite and bool(result["allFinite"])
        non_finite_reference += int(result["nonFiniteReferenceCount"])
        non_finite_candidate += int(result["nonFiniteCandidateCount"])
        frame_results.append({"index": index, **result})

    mean_absolute_error = absolute_error_sum / finite_pairs if finite_pairs else None
    mean_squared_error = squared_error_sum / finite_pairs if finite_pairs else None
    result = {
        "schemaVersion": 1,
        "height": height,
        "width": width,
        "motionConvention": motion_convention,
        "frameCount": len(frames),
        "allFinite": all_finite,
        "finitePairCount": finite_pairs,
        "nonFiniteReferenceCount": non_finite_reference,
        "nonFiniteCandidateCount": non_finite_candidate,
        "maximumAbsoluteError": maximum_absolute_error if finite_pairs else None,
        "meanAbsoluteError": mean_absolute_error,
        "meanSquaredError": mean_squared_error,
        "rootMeanSquaredError": (
            math.sqrt(mean_squared_error) if mean_squared_error is not None else None
        ),
        "unitRangePeakSignalToNoiseRatioDecibels": (
            -10 * math.log10(mean_squared_error)
            if mean_squared_error is not None and mean_squared_error > 0
            else None
        ),
        "frames": frame_results,
    }
    if motion_convention == "pixel-current-to-previous":
        result["jitterDeltaPixelsPerFrame"] = jitter_deltas
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare private external golden outputs with numbered NRK outputs."
    )
    parser.add_argument("bundle", type=pathlib.Path)
    parser.add_argument("candidate_directory", type=pathlib.Path)
    parser.add_argument("--atol", type=float)
    arguments = parser.parse_args()
    if arguments.atol is not None and arguments.atol < 0:
        parser.error("atol must be non-negative")
    result = compare_bundle(arguments.bundle, arguments.candidate_directory)
    print(json.dumps(result, sort_keys=True))
    if not result["allFinite"]:
        return 1
    if (
        arguments.atol is not None
        and result["maximumAbsoluteError"] is not None
        and result["maximumAbsoluteError"] > arguments.atol
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
