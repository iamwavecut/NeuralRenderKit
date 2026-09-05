#!/usr/bin/env python3
"""Wrap external logical neural-rendering weights in an MLX-DLSS model package."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import pathlib
import shutil
import tempfile

from safetensors.numpy import safe_open


SOURCE_FORMATS = {f"dlssnr-logical-v{version}" for version in range(8, 19)}
ARCHITECTURE = "mlxdlss.neural-rendering-transformer.v1"
IDENTIFIER = "org.mlxdlss.external.neural-rendering-transformer"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_specs(source: pathlib.Path) -> list[dict[str, object]]:
    dtype_names = {"F16": "float16", "F32": "float32"}
    specs: list[dict[str, object]] = []
    with safe_open(source, framework="numpy") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") not in SOURCE_FORMATS:
            raise ValueError("source format must be dlssnr-logical-v8 through v18")
        if metadata.get("fully_logical") != "true":
            raise ValueError("source must declare fully_logical=true")
        for name in sorted(handle.keys()):
            view = handle.get_slice(name)
            dtype = view.get_dtype()
            if dtype not in dtype_names:
                raise ValueError(f"unsupported tensor dtype for {name}: {dtype}")
            specs.append(
                {
                    "name": name,
                    "dataType": dtype_names[dtype],
                    "shape": list(view.get_shape()),
                }
            )
    return specs


def _link_or_copy(source: pathlib.Path, destination: pathlib.Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
    shutil.copyfile(source, destination)
    return "copy"


def package(source: pathlib.Path, destination: pathlib.Path) -> dict[str, object]:
    source = pathlib.Path(source)
    destination = pathlib.Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    if not source.is_file():
        raise FileNotFoundError(f"source is not a file: {source}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"destination parent is not a directory: {destination.parent}"
        )

    specs = _tensor_specs(source)
    digest = _sha256(source)
    manifest = {
        "schemaVersion": 1,
        "identifier": IDENTIFIER,
        "architecture": ARCHITECTURE,
        "inputs": [
            {
                "name": "color",
                "dataType": "float32",
                "layout": "nhwc",
                "shape": [1, "height", "width", 16],
            }
        ],
        "outputs": [
            {
                "name": "color",
                "dataType": "float32",
                "layout": "nhwc",
                "shape": [1, "height", "width", 4],
            }
        ],
        "state": {"kind": "stateless"},
        "weights": {
            "file": "weights.safetensors",
            "sha256": digest,
            "tensors": specs,
        },
    }

    staging = pathlib.Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        storage = _link_or_copy(source, staging / "weights.safetensors")
        manifest_path = staging / "manifest.json"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return {
        "schemaVersion": 1,
        "architecture": ARCHITECTURE,
        "destination": str(destination),
        "tensorCount": len(specs),
        "weightsSHA256": digest,
        "storage": storage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package external logical neural-rendering weights for MLX-DLSS."
    )
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    arguments = parser.parse_args()
    print(json.dumps(package(arguments.source, arguments.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
