"""``nrk-torch``: command-line front end mirroring the Swift ``nrk run`` first-frame path."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from .features import PROFILES
from .pipeline import PRECISIONS, NeuralRenderingPipeline


def read_image(path: pathlib.Path, width: int | None, height: int | None) -> np.ndarray:
    if path.suffix.lower() == ".f32":
        if not (width and height):
            raise SystemExit("--width and --height are required for raw .f32 input")
        data = np.fromfile(path, dtype="<f4")
        if data.size != width * height * 3:
            raise SystemExit(f"{path} holds {data.size} floats, expected {width * height * 3}")
        return data.reshape(height, width, 3)
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)


def write_image(path: pathlib.Path, image: np.ndarray) -> None:
    if path.suffix.lower() == ".f32":
        image.astype("<f4").tofile(path)
        return
    from PIL import Image

    Image.fromarray((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nrk-torch", description="PyTorch inference for the recovered neural-rendering transformer")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="enhance one image")
    run.add_argument("--weights", required=True, type=pathlib.Path, help="logical safetensors (from nrk-weights)")
    run.add_argument("--input", required=True, type=pathlib.Path, help="PNG/JPEG, or raw .f32 RGB with --width/--height")
    run.add_argument("--output", required=True, type=pathlib.Path, help="PNG/JPEG or raw .f32 (must not exist)")
    run.add_argument("--width", type=int); run.add_argument("--height", type=int)
    run.add_argument("--profile", default="standard", choices=tuple(PROFILES))
    run.add_argument("--processing-scale", type=float, default=1.0)
    run.add_argument("--detail-strength", type=float, default=1.0)
    run.add_argument("--colour-strength", type=float, default=1.0)
    run.add_argument("--detail-radius", type=float, default=4.0)
    run.add_argument("--intensity", type=float, default=1.0)
    run.add_argument("--noise-frame-index", type=int, default=0)
    run.add_argument("--control-mask", type=pathlib.Path, help="RGB mask image: red blend, green tone, blue structure")
    run.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N or mps")
    run.add_argument("--precision", default="reference", choices=PRECISIONS)
    run.add_argument("--summary", action="store_true", help="print a JSON summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"destination exists: {args.output}")
    image = read_image(args.input, args.width, args.height)
    control_mask = None
    if args.control_mask is not None:
        control_mask = read_image(args.control_mask, image.shape[1], image.shape[0])
    started = time.perf_counter()
    pipeline = NeuralRenderingPipeline.from_safetensors(args.weights, device=args.device, precision=args.precision)
    load_seconds = time.perf_counter() - started
    result = pipeline.enhance(
        image,
        profile=args.profile,
        processing_scale=args.processing_scale,
        detail_strength=args.detail_strength,
        colour_strength=args.colour_strength,
        detail_radius=args.detail_radius,
        intensity=args.intensity,
        frame_index=args.noise_frame_index,
        control_mask=control_mask,
    )
    write_image(args.output, result.image)
    summary = {
        "device": str(pipeline.device), "precision": args.precision, "shape": list(result.image.shape),
        "networkExtent": list(result.network_extent), "loadSeconds": round(load_seconds, 3),
        **{f"{key}Seconds": round(value, 3) for key, value in result.timings.items()},
    }
    if args.summary:
        print(json.dumps(summary, indent=2))
    else:
        print(f"wrote {args.output} ({summary['shape'][1]}x{summary['shape'][0]}, network {summary['networkExtent'][1]}x{summary['networkExtent'][0]}, {summary['device']}, network {summary['networkSeconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
