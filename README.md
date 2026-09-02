# NeuralRenderKit

Neural rendering runtime with a recovered 71-block transformer graph: Swift
package and `nrk` CLI for Apple Silicon (MLX/Metal and Core ML), plus a
cross-platform Python package (PyTorch inference and weight tooling).
The weights come from your own copy of NVIDIA's `nvngx_dlssnr.dll`; nothing
proprietary is included, downloaded or redistributed.

NeuralRenderKit is independent software. It is not affiliated with, endorsed
by, or a drop-in implementation of NVIDIA DLSS or any other proprietary product.

## Quick start: from the DLL to an enhanced image

1. Take `nvngx_dlssnr.dll` from your NVIDIA driver or Streamline package
   (the supported checkpoint is file version `310.8.0.0`, SHA-256
   `ceb6432f…2650`) and put it in a folder of your choice, for example
   `weights/`.
2. Install the Python package (Python 3.10+, macOS, Linux or Windows) and
   convert the weights:

   ```sh
   python -m pip install ./python
   nrk-weights all weights/nvngx_dlssnr.dll weights/ --coreml 320x320
   ```

   This writes `weights/dlssnr-weights-logical.safetensors` (PyTorch),
   `weights/NeuralRendering.nrkmodel` (Swift/MLX) and, per `--coreml WxH`,
   `weights/NeuralRendering-WxH-float16.mlpackage` (Core ML; needs
   `pip install './python[coreml]'`, macOS or Linux). `nrk-weights sha256 FILE`
   tells whether a DLL is a known checkpoint; `nrk-weights inspect PACKED`
   lists what an unknown DLL version contains.
3. Run inference.

   PyTorch, any OS (`--device auto|cpu|cuda|mps`, `--precision reference|fast`):

   ```sh
   nrk-torch run --weights weights/dlssnr-weights-logical.safetensors \
     --input portrait.png --output portrait-enhanced.png
   ```

   Metal on Apple Silicon (macOS 14+, Xcode with Swift 6.2, CMake and Ninja):

   ```sh
   swift build -c release && scripts/prepare-mlx-metallib.sh "$(swift build -c release --show-bin-path)"
   .build/release/nrk render-image portrait.png weights/NeuralRendering.nrkmodel \
     --output portrait-enhanced.png --execution metal-fused --precision float16
   ```

   Core ML (fixed network extent: a `256×256` image runs on the `320×320`
   package, `1080p` needs `1920×1088`):

   ```sh
   .build/release/nrk render-image portrait.png weights/NeuralRendering-320x320-float16.mlpackage \
     --output portrait-enhanced.png --backend coreml --compute-units cpu-gpu
   ```

Raw float32 NHWC tensors go through `nrk run MODEL --input in.f32 --input-format rgb-first-frame --width W --height H --output out.f32`
and `nrk-torch run --input in.f32 --width W --height H`.

## Controls

| option (`nrk run` / `nrk-torch run`) | default | effect |
| --- | --- | --- |
| `--profile standard\|natural\|cinematic\|neutral` | `standard` | style index and local tone/structure preset |
| `--processing-scale 1–4` | `1` | run the network on the frame resampled by this factor |
| `--detail-strength 0–8`, `--colour-strength 0–4`, `--detail-radius` | `1`, `1`, `4` | `result = input + colour·lowpass(change) + detail·highpass(change)`; `--processing-scale 2 --detail-strength 2` is the photoreal recipe |
| `--intensity 0–1` | `1` | blend of the enhanced result over the input |
| `--control-mask rgb.f32` | none | red: blend, green: tone, blue: structure per pixel |
| `--noise-frame-index` | `0` | deterministic noise seed; `NeuralRenderingSession` advances it per frame |

## Video

`nrk-video` converts a whole file through FFmpeg: it decodes to raw RGB, runs
every frame through the pipeline and encodes with arguments you control,
copying the source audio. `--temporal` switches on the recovered temporal
path: the previous output is reprojected with motion vectors (OpenCV DIS
optical flow by default, `pip install './python[video]'`) into the network
input and the head's learned blend mixes it with the new prediction; a scene
cut (mean luma change above `--scene-cut`) resets the history.

```sh
nrk-video convert input.mp4 output.mp4 --backend nrk --model weights/NeuralRendering.nrkmodel \
  --temporal --encode-args "-c:v libx265 -crf 20 -preset slow"   # Apple Silicon: frames stream through `nrk stream` (Metal)
nrk-video convert input.mp4 output.mp4 --weights weights/dlssnr-weights-logical.safetensors \
  --device cuda --temporal --encode-args "-c:v libx265 -crf 20 -preset slow -pix_fmt yuv420p10le"
nrk-video convert input.mp4 output.mp4 --weights ... --device cuda --batch 4 \
  --processing-scale 2 --detail-strength 2                 # single-frame mode with the photoreal recipe
nrk-video convert input.mp4 clip.mp4 --weights ... --start-frame 300 --frames 120 --decode-args "-vf scale=1280:-2"
nrk-video probe input.mp4
nrk-video compare input.mp4 output.mp4                    # original | processed side by side in mpv
```

`--backend nrk` keeps decoding, motion estimation and encoding in Python and
sends frames to the Swift runtime over pipes (`nrk stream`, about `4 fps` at
`512×448` on an M2 Max, identical output to `nrk run-sequence`); the PyTorch
backend is the cross-platform path.
Default encoding is `-c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -movflags +faststart`;
`--pix-fmt rgb48le` keeps 16-bit sources; a status line is printed every
`--status-interval` seconds. Temporal mode runs at the native scale (the
detail/colour split still applies to the output); engines with their own
motion vectors feed `TemporalSession.process(frame, motion=...)` with
normalised history-UV offsets (`normalize_pixel_motion` converts pixel
motion). The Python temporal path matches the Swift reference
(`nrk run-sequence --pipeline portable`) within `0.0014` MAE per frame on
real weights. There is no player of our own: playback and A/B checks go
through mpv.

## Python API

```python
from neuralrenderkit import NeuralRenderingPipeline, TemporalSession
pipeline = NeuralRenderingPipeline.from_safetensors("weights/dlssnr-weights-logical.safetensors", device="auto")
result = pipeline.enhance(image_float32_hwc, profile="standard", processing_scale=2, detail_strength=2)
session = TemporalSession(pipeline)          # frame sequences with reprojected history (motion from optical flow)
enhanced = session.process(frame)            # or session.process(frame, motion=engine_uv_offsets)
```

Measured against the NVIDIA DLL on native game renders (1152–1408 px): the
Metal port is within `0.004–0.005` MAE, the PyTorch pipeline within `0.002` of
the Metal port. The PyTorch reference graph is unoptimised (about `15 s` per
`1152×1216` frame on an M2 Max); use Metal on Apple Silicon for speed.

## Documentation

- [Embedding guide](docs/embedding.md): the Swift API for still frames, Core ML heads and the temporal reference.
- [Recovery notes](docs/recovery-notes.md): package format, the recovered graph, every measured error, temporal command-line reference, roadmap.
- [Research notes](docs/research/): kernel captures and the first-frame preprocessor.

## Development

`scripts/verify.sh` runs the Swift tests, the Python tool and package tests and
the public-tree audit (no DLLs, weights or captures may enter the tree); CI runs
it on macOS and the Python package on macOS, Linux and Windows.
`python -m unittest discover -s python -t python -p 'test_*.py'` runs the Python
package tests alone.

## Status

| Capability | State |
| --- | --- |
| Recovered 71-block neural-rendering graph on MLX/Metal | Working; `0.004–0.005` MAE against the NVIDIA DLL on native game renders |
| PyTorch pipeline (`python/`, CPU/CUDA/MPS) | Working; within `0.002` MAE of the Metal port |
| Fixed-shape Core ML head | Working; `0.008–0.014` MAE against the DLL, lower fidelity than Metal |
| Bring-your-own-DLL weight tooling (`nrk-weights`) | Working on macOS, Linux and Windows (CI matrix) |
| Still images (`nrk render-image`, `nrk-torch run`) and raw tensors (`nrk run`) | Working |
| Temporal reference (`nrk run-sequence`, Python `TemporalSession`) | Working; Swift and Python agree within `0.0014` MAE; end-to-end parity with NVIDIA's temporal path open |
| Video conversion (`nrk-video`, FFmpeg decode/encode, audio copy) | Working; single-frame and temporal modes (optical-flow or engine motion, learned history blend) |

## Safety

```sh
SWIFTPM_MAXIMUM_CONCURRENT_JOBS=2 swift test
scripts/audit-public-tree.sh .
```

The public-tree audit checks tracked and non-ignored untracked files. It rejects executable binary magic, CUDA fatbins, DLL/private-oracle filenames, unexpected safetensors, large unreviewed files, and local absolute home paths. Generated Metal artifacts and downloaded packages stay under `.build/` and must never be committed.

See [SECURITY.md](SECURITY.md) for the package threat boundary,
[CONTRIBUTING.md](CONTRIBUTING.md) for fixture and evidence rules,
[PUBLICATION.md](PUBLICATION.md) for the legal/release gate, and [NOTICE](NOTICE)
for dependency and trademark attribution. The architecture and research basis
live in [the design document](docs/superpowers/specs/2026-08-30-neural-render-kit-design.md).

## License

NeuralRenderKit source is licensed under Apache License 2.0. External model packages retain their own licenses. Do not redistribute a model merely because the runtime can load it.
