"""``nrk-weights``: bring-your-own-DLL tooling.

    nrk-weights extract  nvngx_dlssnr.dll packed.safetensors     # packed intermediate (the DLL's WEIGHTS_HT resource)
    nrk-weights decode   packed.safetensors logical.safetensors  # logical tensors for PyTorch (and every converter)
    nrk-weights mlx      logical.safetensors Model.nrkmodel      # NeuralRenderKit MLX package (Swift nrk)
    nrk-weights coreml   logical.safetensors Model.mlpackage --width W --height H   # fixed-shape Core ML (needs coremltools)
    nrk-weights all      nvngx_dlssnr.dll OUTPUT_DIR [--coreml WxH ...]             # everything in one go
    nrk-weights inspect  packed.safetensors                     # list what an (unknown) DLL version contains

The DLL is never redistributed: users take it from their own NVIDIA driver or
Streamline package. The supported checkpoint is identified by its SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

KNOWN_DLL_SHA256 = {
    # nvngx_dlssnr.dll carrying the v19 neural-rendering checkpoint (WEIGHTS_HT resource, 153 packed tensors)
    "ceb6432f6fbdf44d886014bcd47241932bf8b67439feef9bbdd0961436662650": "nvngx_dlssnr.dll, neural-rendering checkpoint v19",
    # its packed intermediate as written by `nrk-weights extract`
    "08a39bcd6c032c5fec20821c44abfd99a8ad85bbff460e5cc945f35ba67d67a3": "dlssnr-weights-packed.safetensors (WEIGHTS_HT of the v19 DLL)",
    # the same intermediate as produced by the earlier standalone extractor
    "1febdf3e3fc868b48e330cea21903927303fb14cd87766169e24bbdf288a9473": "dlssnr-weights-packed.safetensors (v19, earlier extractor metadata)",
}


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(source: pathlib.Path, destination: pathlib.Path, resource_blob: bool = False) -> int:
    from . import extract_dlssnr_weights

    argv = [str(source), str(destination)] + (["--resource-blob"] if resource_blob else [])
    return extract_dlssnr_weights.main(argv) if _accepts_argv(extract_dlssnr_weights.main) else _run_with_argv(extract_dlssnr_weights.main, argv)


def _decode(source: pathlib.Path, destination: pathlib.Path) -> int:
    from . import unpack_dlssnr_weights

    return _run_with_argv(unpack_dlssnr_weights.main, [str(source), str(destination)])


def _mlx(source: pathlib.Path, destination: pathlib.Path) -> int:
    from . import package_neural_rendering_transformer

    return _run_with_argv(package_neural_rendering_transformer.main, [str(source), str(destination)])


def _coreml(source: pathlib.Path, destination: pathlib.Path, width: int, height: int, precision: str) -> int:
    try:
        import coremltools  # noqa: F401
    except ImportError:
        print("coremltools is not installed (pip install 'neuralrenderkit[coreml]'; macOS or Linux)", file=sys.stderr)
        return 2
    from . import convert_neural_rendering_coreml

    return _run_with_argv(
        convert_neural_rendering_coreml.main,
        [str(source), str(destination), "--precision", precision, "--width", str(width), "--height", str(height)],
    )


def _accepts_argv(function) -> bool:
    import inspect

    return len(inspect.signature(function).parameters) >= 1


def _run_with_argv(function, argv: list[str]) -> int:
    if _accepts_argv(function):
        return int(function(argv) or 0)
    saved = sys.argv
    try:
        sys.argv = [saved[0], *argv]
        return int(function() or 0)
    finally:
        sys.argv = saved


def _inspect(packed: pathlib.Path) -> int:
    from safetensors.numpy import safe_open

    with safe_open(str(packed), framework="numpy") as source:
        metadata = source.metadata() or {}
        names = list(source.keys())
        print(f"format {metadata.get('format', '?')}  tensors {len(names)}  source sha256 {metadata.get('source_sha256', '?')}")
        for name in sorted(names):
            shape = source.get_slice(name).get_shape()
            print(f"  {name}  {shape[0] if len(shape) == 1 else shape} bytes")
    print("decode support is decided per tensor family by `nrk-weights decode`; unknown families are reported there")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nrk-weights", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="DLL -> packed intermediate safetensors")
    extract.add_argument("dll", type=pathlib.Path); extract.add_argument("destination", type=pathlib.Path)
    extract.add_argument("--resource-blob", action="store_true", help="source is an already extracted WEIGHTS_HT resource")
    decode = commands.add_parser("decode", help="packed intermediate -> logical safetensors")
    decode.add_argument("packed", type=pathlib.Path); decode.add_argument("destination", type=pathlib.Path)
    mlx = commands.add_parser("mlx", help="logical safetensors -> .nrkmodel package for the Swift/MLX runtime")
    mlx.add_argument("logical", type=pathlib.Path); mlx.add_argument("destination", type=pathlib.Path)
    coreml = commands.add_parser("coreml", help="logical safetensors -> fixed-shape Core ML package")
    coreml.add_argument("logical", type=pathlib.Path); coreml.add_argument("destination", type=pathlib.Path)
    coreml.add_argument("--width", type=int, default=320); coreml.add_argument("--height", type=int, default=320)
    coreml.add_argument("--precision", default="float16", choices=("float16", "float32"))
    everything = commands.add_parser("all", help="DLL -> packed, logical and .nrkmodel (plus optional Core ML sizes)")
    everything.add_argument("dll", type=pathlib.Path); everything.add_argument("output_dir", type=pathlib.Path)
    everything.add_argument("--coreml", action="append", default=[], metavar="WxH", help="also build a Core ML package at this size (repeatable)")
    everything.add_argument("--coreml-precision", default="float16", choices=("float16", "float32"))
    everything.add_argument("--name", default="NeuralRendering", help="base name for the produced packages")
    inspect = commands.add_parser("inspect", help="list the tensors of a packed intermediate (for unknown DLL versions)")
    inspect.add_argument("packed", type=pathlib.Path)
    digest = commands.add_parser("sha256", help="print the SHA-256 and file version of a DLL and whether it is a known checkpoint")
    digest.add_argument("path", type=pathlib.Path)
    framegen = commands.add_parser("extract-fg", help="libnvidia-ngx-dlssg.so -> dense frame generation safetensors for nrk-video / FrameGenerator")
    framegen.add_argument("library", type=pathlib.Path); framegen.add_argument("destination", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sha256":
        from .extract_dlssnr_weights import pe_file_version

        value = sha256_of(args.path)
        version = pe_file_version(args.path.read_bytes()) or "no PE version info"
        print(f"{value}  {args.path}  version {version}  {KNOWN_DLL_SHA256.get(value, 'unknown checkpoint')}")
        return 0
    if args.command == "inspect":
        return _inspect(args.packed)
    if args.command == "extract-fg":
        from .extract_dlssg_weights import main as extract_fg

        return extract_fg([str(args.library), str(args.destination)])
    if args.command == "extract":
        return _extract(args.dll, args.destination, args.resource_blob)
    if args.command == "decode":
        return _decode(args.packed, args.destination)
    if args.command == "mlx":
        return _mlx(args.logical, args.destination)
    if args.command == "coreml":
        return _coreml(args.logical, args.destination, args.width, args.height, args.precision)
    if args.command == "all":
        out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
        print(f"dll sha256 {sha256_of(args.dll)}")
        packed = out / "dlssnr-weights-packed.safetensors"; logical = out / "dlssnr-weights-logical.safetensors"; mlx = out / f"{args.name}.nrkmodel"
        for step, function in (("extract", lambda: _extract(args.dll, packed)), ("decode", lambda: _decode(packed, logical)), ("mlx", lambda: _mlx(logical, mlx))):
            code = function()
            if code:
                print(f"{step} failed with code {code}", file=sys.stderr); return code
            print(f"{step}: ok")
        for size in args.coreml:
            width, height = (int(v) for v in size.lower().split("x"))
            code = _coreml(logical, out / f"{args.name}-{width}x{height}-{args.coreml_precision}.mlpackage", width, height, args.coreml_precision)
            if code:
                return code
            print(f"coreml {size}: ok")
        print(f"artifacts in {out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
