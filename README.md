# NeuralRenderKit

Runtime for two networks recovered from NVIDIA's DLSS libraries: the
neural-rendering transformer (detail, colour, tone) and the frame generator
(interpolated frames for video). Swift package and `nrk` CLI for Apple Silicon
(MLX/Metal, Core ML), Python package for every platform (PyTorch inference,
weight tooling, video conversion, web front end). Weights come from your own
copies of `nvngx_dlssnr.dll` and `libnvidia-ngx-dlssg.so`; nothing proprietary
is included, downloaded or redistributed.

NeuralRenderKit is independent software, not affiliated with or endorsed by
NVIDIA, and not a drop-in implementation of DLSS.

> **A note for the NVIDIA reader.** This port was worked out on a laptop and on
> GPU instances rented by the hour, some of which even booted. A pair of DGX
> Sparks would have replaced the rentals and would have a steady job here:
> experiments like this one, and the pet projects queued behind it. Hit me up on
> X: [@WaveCut](https://x.com/WaveCut).

![Input, default strength, processing scale 2 with detail 2](docs/assets/neural-rendering-control.png)

Neural rendering on a 1408×1600 game render, 1:1 crop: input, defaults, `--processing-scale 2 --detail-strength 2`.

[![Half-rate input, generated frames, withheld real frames](docs/assets/frame-generation-validate.png)](docs/assets/frame-generation-validate.mp4)

Frame generation: even frames in, generated frames, the withheld odd frames (click for the video).

## Requirements

- Python 3.10+ (macOS, Linux, Windows); PyTorch is installed as a dependency.
- Video: `ffmpeg` and `ffprobe` in `PATH`; optical-flow temporal mode: `pip install './python[video]'`.
- Metal backend: macOS 14+, Xcode with Swift 6.2, CMake, Ninja.
- Core ML packages: `pip install './python[coreml]'` (macOS or Linux).
- Web front end: `pip install './python[web]'`.

## Weights

| network | source file | command | output |
| --- | --- | --- | --- |
| neural rendering | `nvngx_dlssnr.dll` (file version 310.8.0.0, SHA-256 `ceb6432f…2650`) | `nrk-weights all nvngx_dlssnr.dll weights/ [--coreml 320x320]` | `weights/dlssnr-weights-logical.safetensors` (PyTorch), `weights/NeuralRendering.nrkmodel` (Metal), `weights/NeuralRendering-WxH-float16.mlpackage` (Core ML) |
| frame generation | `libnvidia-ngx-dlssg.so.310.7.0` (DLSS SDK 310.7.0) | `nrk-weights extract-fg libnvidia-ngx-dlssg.so.310.7.0 weights/framegen.safetensors` | `weights/framegen.safetensors` (both backends) |

`nrk-weights sha256 FILE` reports whether a DLL is a known checkpoint;
`nrk-weights inspect PACKED` lists the tensors of an unknown version.

## Install and build

```sh
python -m pip install './python[web,video]'
swift build -c release && scripts/prepare-mlx-metallib.sh "$(swift build -c release --show-bin-path)"   # macOS, Metal backend
```

The second command is required after every clean Swift build: it places
`mlx.metallib` next to the `nrk` binary.

## Commands

Still images:

```sh
nrk-torch run --weights weights/dlssnr-weights-logical.safetensors --input in.png --output out.png   # PyTorch: --device auto|cpu|cuda|mps, --precision reference|fast
.build/release/nrk render-image in.png weights/NeuralRendering.nrkmodel --output out.png --execution metal-fused --precision float16
.build/release/nrk render-image in.png weights/NeuralRendering-320x320-float16.mlpackage --output out.png --backend coreml --compute-units cpu-gpu
```

Core ML packages have a fixed extent: a 256×256 image runs on the 320×320
package, 1080p needs 1920×1088. Raw float32 NHWC tensors:
`nrk run MODEL --input in.f32 --input-format rgb-first-frame --width W --height H --output out.f32`,
`nrk-torch run --input in.f32 --width W --height H`.

Frame generation:

```sh
.build/release/nrk framegen a.png b.png --weights weights/framegen.safetensors --output between.png   # --factor 3|4 writes between-1.png …; --phase 0.25
nrk-video framegen in.mp4 out.mp4 --weights weights/framegen.safetensors                       # frame rate x2, audio copied
nrk-video framegen in.mp4 out.mp4 --weights weights/framegen.safetensors --backend nrk         # Metal through nrk framegen-stream; --batch 4 pairs per pass
nrk-video framegen in.mp4 slow.mp4 --weights weights/framegen.safetensors --mode slowmo --factor 4 --audio stretch
```

`--mode fps` multiplies the frame rate and keeps the duration; `--mode slowmo`
keeps the rate and stretches the clip. `--audio copy|stretch|none`: `stretch`
(slow motion only) uses FFmpeg `atempo`, pitch preserved.

Video through the neural renderer:

```sh
nrk-video convert in.mp4 out.mp4 --backend nrk --model weights/NeuralRendering.nrkmodel --temporal --encode-args "-c:v libx265 -crf 20 -preset slow"
nrk-video convert in.mp4 out.mp4 --weights weights/dlssnr-weights-logical.safetensors --device cuda --batch 4 --processing-scale 2 --detail-strength 2
nrk-video convert in.mp4 clip.mp4 --weights ... --start-frame 300 --frames 120 --decode-args "-vf scale=1280:-2"
nrk-video probe in.mp4
nrk-video compare in.mp4 out.mp4          # original | processed side by side in mpv
```

`--temporal` reprojects the previous output with motion (OpenCV DIS optical
flow, or engine motion through the Python API) and blends it with the learned
history weight; `--scene-cut` resets the history on a luma jump. Default
encoding: `-c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart`;
`--pix-fmt rgb48le` keeps 16-bit sources; `--status-interval` seconds between
progress lines. Temporal mode runs at the native scale.

Web front end:

```sh
nrk-web        # http://127.0.0.1:8181; --port, --no-browser, --native (pywebview window), --root DIR
```

Pages: Image (before/after slider), Video (effect chain: neural rendering and
frame generation in either order, side-by-side preview), Jobs (queue, progress,
cancel, downloads), Settings (weight paths, backend, device, theme). Jobs run
one at a time; results are stored under `~/NeuralRenderKit/outputs/<job>/`.
HTTP API: `GET /api/effects`, `POST /api/jobs` (multipart `file` + JSON
`effects`), `GET /api/jobs[/{id}]`, `POST /api/jobs/{id}/cancel`,
`GET /api/jobs/{id}/preview`, `GET /api/jobs/{id}/download/{n}`.

## Controls (neural rendering)

| option (`nrk run` / `nrk-torch run` / `nrk-video convert`) | default | effect |
| --- | --- | --- |
| `--profile standard\|natural\|cinematic\|neutral` | `standard` | style index and local tone/structure preset |
| `--processing-scale 1–4` | `1` | run the network on the frame resampled by this factor |
| `--detail-strength 0–8`, `--colour-strength 0–4`, `--detail-radius` | `1`, `1`, `4` | `result = input + colour·lowpass(change) + detail·highpass(change)` |
| `--intensity 0–1` | `1` | blend of the enhanced result over the input |
| `--control-mask rgb.f32` | none | red: blend, green: tone, blue: structure, per pixel |
| `--noise-frame-index` | `0` | deterministic noise seed; sessions advance it per frame |

## Python API

```python
from neuralrenderkit import NeuralRenderingPipeline, TemporalSession, FrameGenerator
pipeline = NeuralRenderingPipeline.from_safetensors("weights/dlssnr-weights-logical.safetensors", device="auto")
result = pipeline.enhance(image_float32_hwc, profile="standard", processing_scale=2, detail_strength=2)
session = TemporalSession(pipeline)           # frame sequences; session.process(frame[, motion=engine_uv_offsets])
generator = FrameGenerator.from_safetensors("weights/framegen.safetensors", device="auto")
middle = generator.generate(frame_a_uint8, frame_b_uint8, factor=2)[0]   # factor-1 frames at phases k/factor
```

## Accuracy and speed

| component | measurement |
| --- | --- |
| Neural rendering, Metal | `0.004–0.005` MAE against the NVIDIA DLL on 1152–1408 px game renders |
| Neural rendering, PyTorch | within `0.002` MAE of the Metal port; ~15 s per 1152×1216 frame on an M2 Max (reference graph) |
| Neural rendering, Core ML | `0.008–0.014` MAE against the DLL |
| Temporal path | Swift and Python agree within `0.0014` MAE per frame; against NVIDIA on a 64-frame static sequence: `0.0054` MAE (`42.3` dB) with the same drift from frame 0 as the vendor; motion, jitter and mask cases not captured |
| Frame generation | reproduces the library's output at `59.9` dB PSNR (max 3/255) on captured frames; five whole clips within `0.01–0.03` dB of the library (27.4–38.9 dB against withheld frames) |
| Frame generation, speed (M2 Max, 960×540 / 1920×1080) | Metal float16 `6.3 / 21` ms per frame on the GPU, `6.4 / 25` ms through `nrk-video framegen --backend nrk`; PyTorch/MPS float16 `5.3 / 17` ms |
| `nrk stream` (video, Metal) | about 11 fps at 512×448 on an M2 Max end to end (ffmpeg, optical flow and the pipe included; features generated on the GPU), identical to `nrk run-sequence` |

Not included: DLSS Super Resolution (measured, loses to Lanczos on realistic
content without engine motion vectors and jitter; see the note below) and the
frame generator's motion-vector, depth, HUD and inpainting inputs (no-ops for
plain video).

## Documentation

- [Frame generation](docs/frame-generation.md): the recovered graph, its verification against the library, whole-clip results, speed.
- [Super resolution](docs/super-resolution.md): what was measured and why it is not ported.
- [Embedding guide](docs/embedding.md): the Swift API for still frames, Core ML heads and the temporal reference.
- [Recovery notes](docs/recovery-notes.md): package format, the recovered neural-rendering graph, measured errors, temporal command-line reference.
- [Research notes](docs/research/): kernel captures and the first-frame preprocessor.

## Development

```sh
scripts/verify.sh                                                  # Swift tests, Python tests, public-tree audit
python -m unittest discover -s python -t python -p 'test_*.py'     # Python package tests only
SWIFTPM_MAXIMUM_CONCURRENT_JOBS=2 swift test                       # Swift tests only
scripts/audit-public-tree.sh .                                     # no DLLs, weights or captures in the tree
```

CI runs the Swift suite on macOS and the Python package on macOS, Linux and
Windows. The audit rejects executable binaries, CUDA fatbins, DLL and weight
files, large unreviewed files and absolute home paths; `weights/` and `.build/`
are ignored and must never be committed. See [SECURITY.md](SECURITY.md),
[CONTRIBUTING.md](CONTRIBUTING.md), [PUBLICATION.md](PUBLICATION.md) and
[NOTICE](NOTICE).

## License

Apache License 2.0 for the source. Model packages built from vendor libraries
keep the vendor's terms; do not redistribute them.
