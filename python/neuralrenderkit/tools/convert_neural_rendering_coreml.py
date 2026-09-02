#!/usr/bin/env python3
"""Convert external logical neural-rendering weights to fixed-shape Core ML.

The recovered graph is implemented locally in ``neural_rendering_reference``.
The generated package contains user-supplied weights and must remain external to
the NeuralRenderKit source repository.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import tempfile

from neuralrenderkit import model as neural_rendering_reference
import torch
from safetensors import safe_open
from torch import nn

IDENTIFIER = "org.neuralrenderkit.external.neural-rendering-transformer.coreml"
ARCHITECTURE = "nrk.neural-rendering-transformer.v1"


class NCHWHeadWrapper(nn.Module):
    """Expose the NHWC reference graph through Core ML's NCHW tensor boundary."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        nhwc = value.permute(0, 2, 3, 1)
        return self.model(nhwc).permute(0, 3, 1, 2)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert_weights(
    weights_path: pathlib.Path,
    destination: pathlib.Path,
    *,
    precision: str,
    height: int = 128,
    width: int = 128,
) -> None:
    import coremltools as ct
    import numpy as np

    weights_path = pathlib.Path(weights_path)
    destination = pathlib.Path(destination)
    if min(height, width) < 128 or height % 64 or width % 64:
        raise ValueError("height and width must be at least 128 and multiples of 64")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"destination parent is not a directory: {destination.parent}"
        )
    if precision not in {"float16", "float32"}:
        raise ValueError("precision must be float16 or float32")

    with safe_open(str(weights_path), framework="pt", device="cpu") as source:
        source_format = (source.metadata() or {}).get("format", "unknown")

    model = NCHWHeadWrapper(neural_rendering_reference.load_model(weights_path)).eval()
    example = torch.zeros((1, 16, height, width), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=True)

    if precision == "float32":
        compute_precision = ct.precision.FLOAT32
    else:
        compute_precision = ct.transform.FP16ComputePrecision(
            op_selector=lambda operation: (
                operation.op_type not in {"reduce_mean", "reduce_sum", "softmax"}
            )
        )
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=compute_precision,
        # Loading here triggers an unnecessary on-device AOT compile of the
        # 147k-op graph. Save the portable package first; runtime probes compile
        # it explicitly on the target Mac.
        skip_model_load=True,
        inputs=[
            ct.TensorType(
                name="color",
                shape=example.shape,
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="restored", dtype=np.float32)],
    )
    converted.author = "NeuralRenderKit contributors"
    converted.short_description = (
        f"Recovered neural-rendering head ({height}x{width}, {precision})"
    )
    converted.user_defined_metadata["com.neuralrenderkit.identifier"] = IDENTIFIER
    converted.user_defined_metadata["com.neuralrenderkit.architecture"] = ARCHITECTURE
    converted.user_defined_metadata["com.neuralrenderkit.compute_precision"] = precision
    converted.user_defined_metadata["com.neuralrenderkit.io_type"] = "multiarray"
    converted.user_defined_metadata["com.neuralrenderkit.input_layout"] = "nchw"
    converted.user_defined_metadata["com.neuralrenderkit.checkpoint_sha256"] = _sha256(
        weights_path
    )
    converted.user_defined_metadata["com.neuralrenderkit.source_format"] = source_format

    staging_root = pathlib.Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    staged_package = staging_root / "model.mlpackage"
    try:
        converted.save(staged_package)
        os.rename(staged_package, destination)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert external logical neural-rendering weights to a fixed "
            "Core ML package."
        )
    )
    parser.add_argument("weights", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    arguments = parser.parse_args()
    convert_weights(
        arguments.weights,
        arguments.destination,
        precision=arguments.precision,
        height=arguments.height,
        width=arguments.width,
    )
    print(f"wrote {arguments.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
