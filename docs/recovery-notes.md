# Model packages and recovery notes

Moved verbatim from the former README: the package format, the recovered neural-rendering graph, every measured error and the parity numbers.


A model package is data, never executable code:

```text
Example.nrkmodel/
  manifest.json
  weights.safetensors
```

Schema version 1 declares the package identifier, reviewed architecture identifier, named input/output tensors, state cadence/reset policy, optional named recurrent tensors, exact weight names/shapes/types, and the lowercase SHA-256 of the weight file. Shape symbols such as `height` and `width` must bind consistently at runtime. Contradictory contracts such as stateless models with history tensors or recurrent models with frame-independent cadence are rejected.

The loader rejects malformed manifests, nested or absolute weight paths, missing files, digest mismatches, unexpected tensors, and shape/type mismatches before GPU execution. A digest verifies integrity, not authorship: applications remain responsible for package provenance and model licensing.

The executable architecture identifier is `nrk.neural-rendering-transformer.v1`; `nrk.pixel-affine.v1` is the test fixture that exercises package plumbing.

The experimental neural-rendering architecture executes its complete 71-block
MLX graph only with a user-supplied external package. Eager execution is the
portable Mac candidate, not yet an NVIDIA or cross-backend oracle. The opt-in
`metal-fused` mode uses native Metal simdgroup matrices for the wide branched
FFNs. Whole-graph compilation remains rejected because it changes the recovered
custom-kernel semantics. A deterministic CPU reference for the no-history first-frame branch
lets an RGB float32 frame run without manually constructing 16 features:

The current external conversion format is `dlssnr-logical-v18`. The verified
v20 conversion contains 649 logical tensors and has SHA-256
`95b0d2688a62f17e1aa233b9819e0e3966e37561543d0a29973ace9ba456b972`; v18
replaces the split-family feed-forward tensors with `group_expand_weight`
(8×64×256) and `group_project_weight` (8×256×64).
On an external RTX vendor oracle, independently supplied inputs put global
blocks 31–38 within `0.0336–0.1544` MAE, and the recovered block-39 bridge
matches `98.39%` of FP8 bytes (`0.000535` MAE). The corrected fused-upsample
payload order and residual reduce the portable structural first-frame run to
RGB correlation `0.999775` and mean absolute display error `0.00450`. A literal
SM89 reproduction of block 48 matches its complete FFN byte-for-byte; with the
recovered bit-affine attention approximation, the whole block reaches `91.84%`
byte agreement and `0.00589` raw E4M3 MAE.
Teacher-forced vendor stage captures later exposed six graph errors. The
single-head window blocks store their `64×64` attention bias in the fused
kernel's `mma` fragment order; `NeuralRenderingAttentionBiasLayout` undoes that
permutation at load time and moves blocks 1–4 from `0.08–0.36` to `0.005–0.02`
MAE against the captures (correlation above `0.9996`); the multi-head tables
are already logical and are not remapped. The deterministic-noise hash is the
PCG `RXS-M-XS` output step whose shift is `(v >> 28) + 4`; with the missing
`+ 4` restored, noise channels 0 and 1 match the vendor's pooled noise at
correlation `0.9997`. The multi-head feed-forward publishes its gated
expansion, per-head branch sum, and residual as E4M3, which halves the
teacher-forced error of blocks 5–14. Block 0 runs its window attention on
the full-resolution adapter output and average-pools afterwards, which takes
its teacher-forced error from `0.0254` to `0.0075`. Together they moved the
four-face Cyberpunk gate from `0.0265` to `0.0180` MAE against NVIDIA, the
face effect correlations from `0.2–0.4` to `0.5–0.7`, and the 4K effect
correlation from `0.28` to `0.45`; the remaining gap then sat in the split
blocks. Per-launch captures of every vendor kernel showed that the split-family
feed-forward (blocks 23–30 and 40–47) is a per-64-channel-group `64 → 256 → 64`
MLP with the quadratic-gate activation behind an E4M3-published first
projection, not a gate-times-value branch structure, and that the 16-head
split blocks store their attention bias in the same fragment order as the
single-head blocks (the 2/4/8-head tables stay logical). With both fixed every
block is within `0.011–0.05` teacher-forced (global blocks `0.03–0.09`), the
four game-face gates reach `0.0079–0.0141` MAE against NVIDIA with effect
correlations `0.91–0.96`, and the Mac change magnitude matches NVIDIA's.
Captures of the kernel inputs then fixed the third deterministic-noise
channel (its radius word multiplies by `0xcaa5b80d`; block 0 moves from
`0.126` to `0.020` against the vendor's block-0 output) and the fused
transitions: the downsample kernels pool the unpublished half-precision block
output and publish it as E4M3 before the projection, and the decoder merges
publish the merged tensor as E4M3 before the block. Finally, the post
block's full-resolution skip is that block-0 output published as E4M3, not
the adapter projection: the projection carries the three noise channels
straight into the head as pixel-level grain (7–12× NVIDIA's pixel-band power
on native renders). With the block-0 skip the four game-face gates reach
`0.0054/0.0065/0.0047/0.0049` MAE against NVIDIA with effect correlations
`0.975–0.988`, and five native game-face crops (1152–1408 px, DLL goldens)
sit at `0.0041–0.0048` MAE with high-pass correlation `0.77–0.89`.
This is substantial pipeline recovery, not the still-missing NVIDIA parity
gate; proprietary weights and captures remain outside the repository.

```sh
nrk run MODEL.nrkmodel \
  --input rgb.f32 --input-format rgb-first-frame \
  --profile standard \
  --output output.f32 --height 128 --width 128
```

`--processing-scale 2 --detail-strength 2 --colour-strength 1` adds the
community photoreal recipe around the single pass (network at 2×, then an
independent low-pass colour and high-pass detail recombination); see
[docs/embedding.md](docs/embedding.md).

`standard` is the default live-control profile and deliberately enables the
nominal captured effect: shipping-default model slot 0, style 0, tone 1,
structure 1, skin `-1`, and no
automatic mask. `natural` and `cinematic` select styles 1 and 2 with the same
strengths; `neutral` is the diagnostic tone-0/structure-0 profile used for
oracle comparisons. Model selection is a separate create-time axis: the
official UI presents choices labeled Model A, B, and C, while public hosts
expose numeric slots 1, 2, and 3 plus shipping-default slot 0. The v19
safetensors proves only slot-0 behavior. Supporting A/B/C requires capturing
all slots and determining whether they swap full weights, smaller adapters, or
create-time graph conditioning; they must not be modeled as profile/style
switches without that evidence.

The same command exposes every underlying recovered control. `Style` is supplied as
an integer index and normalized by `1/128`. Automatic masking uses local
structure as the fallback skin strength. An RGB `ControlMask` takes precedence
over automatic masking: red controls final effect intensity, green scales local
tone, and blue scales local structure. It may be logical-size or supplied as a
point-transformed backing resource.

```sh
nrk run MODEL.nrkmodel \
  --input rgb.f32 --input-format rgb-first-frame \
  --output output.f32 --height 128 --width 128 \
  --style-index 64 --local-tone 0.75 --local-structure 0.25 \
  --auto-mask enabled --skin-structure -1

nrk run MODEL.nrkmodel \
  --input rgb.f32 --input-format rgb-first-frame \
  --control-mask control-mask-rgb.f32 --intensity 0.8 \
  --output output.f32 --height 128 --width 128

nrk run MODEL.nrkmodel \
  --input backing-rgb.f32 --input-format rgb-first-frame \
  --input-transform 0,0,256,128,256,128 \
  --control-mask backing-control-mask.f32 \
  --control-mask-transform 0,0,256,128,256,128 \
  --output output.f32 --height 128 --width 128
```

The command applies the recovered first-frame residual composition and writes
three-channel display RGB; ordinary `--input-format model` execution still
exposes the four-channel neural head declared by the external package. This
proves a raw-RGB first-frame path, not vendor parity. Non-full current/control
transforms, masked temporal behavior, and caller-owned jitter correction are
executable. A two-frame NVIDIA baseline now reaches aggregate correlation above
`0.9997` and MAE `0.00480`, but tone, structure, auto-mask, and ControlMask
variants do not yet pass the same gate. See [the preprocessor evidence note](docs/research/2026-08-31-dlssnr-first-frame-preprocessor.md).

For linear-HDR sources, `NeuralRenderingDisplayCodec` provides the surrounding
display contract recovered from a live integration: it preserves the untouched
source, produces the model's sRGB proxy with a highlight soft knee, then folds a
complete model picture back through luminance-ratio composition rather than an
unstable inverse tone curve. `MLXNeuralRenderingDisplayCodec` implements the
same encode/resolve contract as device-resident Metal kernels. Both paths are
byte-exact no-ops at zero transfer strength. They are separate from the
four-channel neural-head postprocessor so callers can A/B the raw checkpoint
contract and the display codec independently while NVIDIA golden capture is
still pending.

## Temporal command-line reference

The same external head backend can now run an ordered RGB sequence through the
portable temporal reference. Motion files contain two float32 channels. The
default `normalized-uv` format is added directly to the current pixel center;
`pixel` applies the recovered signed scale/extent conversion. Files default to
the processing size. Optional six-field transforms let current color and motion
come from backing resources and configure retained-history sampling:

```sh
nrk run-sequence MODEL.nrkmodel \
  --input-format rgb-temporal-reference \
  --pipeline device-resident --execution eager \
  --motion-format pixel --motion-scale-x 1 --motion-scale-y 1 \
  --profile neutral \
  --jitter-delta-x 0 --jitter-delta-y 0 \
  --input-transform 0,0,256,128,256,128 \
  --motion-transform 0,0,256,128,256,128 \
  --history-transform 8,0,112,128,128,128 \
  --control-mask-transform 0,0,256,128,256,128 --intensity 0.8 \
  --height 128 --width 128 --output-dir outputs \
  --input color-0.f32 --motion motion-0.f32 --depth depth-0.f32 \
  --control-mask control-mask-0.f32 \
  --input color-1.f32 --motion motion-1.f32 --depth depth-1.f32 \
  --control-mask control-mask-1.f32
```

Replace the model path and select `--backend coreml --compute-units cpu-gpu
--pipeline portable --preprocessor metal` to run the converted Core ML head
through the same temporal reference.

When a backing NHWC texture is larger than its integer-aligned processing
subrect, `NeuralRenderingAlignedSubrect` extracts the exact compact logical view
for color, history, motion, or control inputs while preserving backing row
stride. For fractional or extent-mismatched history and motion,
`NeuralRenderingTextureTransform` represents the recovered
`(base, extent, resource size)` mapping. DLL texture-object construction fixes
history to normalized clamp-to-edge linear filtering and motion to normalized
clamp-to-edge point filtering. The CPU, standalone Metal, and device-resident
MLX paths implement those semantics; non-full current color and explicit
ControlMask are point-sampled by the same transform adapter in the first-frame
CLI before both feature assembly and postprocessing.

`NeuralRenderingTemporalReferenceBackend` conforms to the same
`NeuralRenderBackend` protocol, owns display history, enforces consecutive-frame
reset semantics, uses current-pixel motion because this DLL binds no depth guide,
applies the recovered five-tap history filter, and composes the sigmoid-capped
temporal output. The dormant closest-depth branch remains an explicit structural
mode; `DepthInverted` affects only that mode. Normalized motion is an independent reference contract, not a claim that
raw game-engine MVec values already use the same units or sign.
`MLXNeuralRenderingDeviceTemporalBackend` implements the same lifecycle and
math while retaining MLX arrays across feature processing, head inference,
postprocessing, and history. Its direct temporal/post kernels remain exact for
identical inputs; device-generated deterministic noise is accepted only within
the measured `0.0002` display gate against the portable CPU-base oracle.

For pixel motion, the recovered host conversion is now executable:

```text
normalizedX = (pixelMotionX * scaleX + jitterDeltaX) / effectiveMotionWidth
normalizedY = (pixelMotionY * scaleY + jitterDeltaY) / effectiveMotionHeight
```

`jitterDeltaX/Y` is `previousJitter - currentJitter` in pixels. It is added after
engine-unit `MVecScale`, so it must not be multiplied by that scale. Pass zero
when motion already includes render jitter. `normalizePixelMotion` preserves
signed scales and rejects non-finite scale/jitter values or non-positive extents;
`run-sequence` accepts jitter only with `--motion-format pixel` and reports
`jitterDeltaPixels` in its JSON result. A two-frame exact-checkpoint A/B produced
byte-identical retained-history output for zero motion plus X jitter `1` and raw
X motion `-0.5` with scale `-2`. This proves the local adapter path, not NVIDIA
output parity. Library history/motion subrect sampling and first-frame
current/control resource adapters are also executable.

Each jitter option may appear once and be broadcast to the sequence, or once per
`--input` in frame order. The JSON result always reports
`jitterDeltaPixelsPerFrame` and retains the shorter `jitterDeltaPixels` field for
a constant sequence.

External RTX captures stay outside the tree and can be verified with
`Tools/compare_neural_rendering_golden_bundle.py`; the bundle schema and current
performance evidence are documented in the linked research note. Generate the
manifest in place from contiguous `color/motion/depth/output-NNNNNN.f32` capture
files, then generate the Apple candidate without manually reconstructing scale,
depth, frame order, jitter, or hashes:

```sh
python Tools/make_neural_rendering_golden_bundle.py PRIVATE_BUNDLE \
  --height 128 --width 128 \
  --motion-convention pixel-current-to-previous \
  --motion-scale-x 1 --motion-scale-y 1 \
  --motion-width 128 --motion-height 128 \
  --depth-inverted false \
  --jitter-delta 0,0 --jitter-delta 0.25,-0.5
python Tools/run_neural_rendering_golden_bundle.py \
  PRIVATE_BUNDLE MODEL.nrkmodel candidate-outputs --dry-run
python Tools/run_neural_rendering_golden_bundle.py \
  PRIVATE_BUNDLE MODEL.nrkmodel candidate-outputs
python Tools/compare_neural_rendering_golden_bundle.py \
  PRIVATE_BUNDLE candidate-outputs --atol 0.001
```

The current runner intentionally accepts only full-rect pixel motion whose
declared motion extent equals the processing size. It fails closed instead of
inventing a resource transform absent from the capture schema. The manifest
generator refuses an existing manifest, symlinks, missing/gapped frames, or an
incorrect raw byte count; it copies no capture data into the source tree.

The recovered MUFU sequence and NVIDIA's published approximation bounds give a
conservative `1/128` absolute ceiling for each half-rounded deterministic-noise
feature. A stronger local stress changed this checkpoint's composed RGB by at
most `0.0010376`; this is sensitivity evidence, not permission to weaken a
bundle-specific NVIDIA comparison threshold.

## Roadmap

1. Close exact PyTorch↔MLX and NVIDIA gates for the corrected 71-block graph before calling either implementation an oracle.
2. Establish external NVIDIA golden parity across jitter, disocclusion, masking, and temporal history cases.
3. Remove the remaining host boundaries with accelerator-resident texture input, guide generation, and final display output.
4. Optimize the same checkpoint through MLXFast, Core ML, and fused Metal while continuously comparing against exact eager output.
5. Package the source-only pluggable component, tests, video demo, and legal/publication boundary for community release.

The performance target is documented 1080p24 where hardware and model complexity permit it, with predictable lower-resolution, lower-cadence, or offline fallbacks where they do not.
