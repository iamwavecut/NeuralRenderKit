#!/usr/bin/env python3
"""Compare two headerless little-endian float32 tensors without ML dependencies."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct


def compare(reference: pathlib.Path, candidate: pathlib.Path) -> dict[str, object]:
    reference = pathlib.Path(reference)
    candidate = pathlib.Path(candidate)
    reference_bytes = reference.stat().st_size
    candidate_bytes = candidate.stat().st_size
    if reference_bytes != candidate_bytes:
        raise ValueError(
            f"byte counts differ: reference {reference_bytes}, candidate {candidate_bytes}"
        )
    if reference_bytes == 0 or reference_bytes % 4 != 0:
        raise ValueError("inputs must contain a non-empty whole number of float32 values")

    element_count = 0
    finite_pair_count = 0
    non_finite_reference_count = 0
    non_finite_candidate_count = 0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    maximum_absolute_error = 0.0
    all_finite = True
    chunk_bytes = 1024 * 1024
    with reference.open("rb") as reference_file, candidate.open("rb") as candidate_file:
        while reference_chunk := reference_file.read(chunk_bytes):
            candidate_chunk = candidate_file.read(len(reference_chunk))
            for (reference_value,), (candidate_value,) in zip(
                struct.iter_unpack("<f", reference_chunk),
                struct.iter_unpack("<f", candidate_chunk),
                strict=True,
            ):
                element_count += 1
                reference_is_finite = math.isfinite(reference_value)
                candidate_is_finite = math.isfinite(candidate_value)
                if not reference_is_finite:
                    non_finite_reference_count += 1
                if not candidate_is_finite:
                    non_finite_candidate_count += 1
                if not reference_is_finite or not candidate_is_finite:
                    all_finite = False
                    continue
                finite_pair_count += 1
                error = abs(reference_value - candidate_value)
                absolute_error_sum += error
                squared_error_sum += error * error
                maximum_absolute_error = max(maximum_absolute_error, error)

    mean_squared_error = (
        squared_error_sum / finite_pair_count if finite_pair_count else None
    )
    return {
        "allFinite": all_finite,
        "elementCount": element_count,
        "finitePairCount": finite_pair_count,
        "nonFiniteReferenceCount": non_finite_reference_count,
        "nonFiniteCandidateCount": non_finite_candidate_count,
        "maximumAbsoluteError": (
            maximum_absolute_error if finite_pair_count else None
        ),
        "meanAbsoluteError": (
            absolute_error_sum / finite_pair_count if finite_pair_count else None
        ),
        "meanSquaredError": mean_squared_error,
        "rootMeanSquaredError": (
            math.sqrt(mean_squared_error) if mean_squared_error is not None else None
        ),
        "unitRangePeakSignalToNoiseRatioDecibels": (
            -10 * math.log10(mean_squared_error)
            if mean_squared_error is not None and mean_squared_error > 0
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw little-endian float32 tensors.")
    parser.add_argument("reference", type=pathlib.Path)
    parser.add_argument("candidate", type=pathlib.Path)
    parser.add_argument("--atol", type=float)
    arguments = parser.parse_args()
    if arguments.atol is not None and arguments.atol < 0:
        parser.error("atol must be non-negative")

    result = compare(arguments.reference, arguments.candidate)
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
