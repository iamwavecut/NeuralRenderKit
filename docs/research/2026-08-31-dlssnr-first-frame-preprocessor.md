# DLSS NR first-frame preprocessor recovery

Status: executable local reference, not an NVIDIA-parity claim.

## Scope

The recovered model core consumes 16 features per pixel. This note began with
the first-frame branch and now also covers the observed temporal branch: motion
reprojection, history filtering, null depth, texture transforms, and temporal
postprocessing. The executable reference covers disabled masking,
automatic-mask strengths, a same-size full-rect explicit RGB control mask, and
fractional or extent-mismatched history/motion resources. Non-full current color
and ControlMask transforms are now executable in the first-frame path; masked
temporal behavior is also executable. Jitter interaction and NVIDIA output
parity remain open.

The public CPU reference is
`NeuralRenderingFirstFramePreprocessor.makeFeatureTensor(from:)`. The CLI can
feed an RGB float32 tensor through this reference and the external MLX model:

```sh
nrk run MODEL.nrkmodel \
  --input rgb.f32 --input-format rgb-first-frame \
  --output output.f32 --height 128 --width 128
```

This is the first path that starts from ordinary RGB rather than a caller-made
16-channel tensor. It does not make a video frame temporally equivalent to the
vendor runtime.

## Evidence snapshot

- Source image SHA-256:
  `ceb6432f6fbdf44d886014bcd47241932bf8b67439feef9bbdd0961436662650`.
- Fifteen CUDA ELF images occur in the PE image. Module 0 begins at file offset
  `913712`, has reconstructed ELF length `1946976`, and SHA-256
  `ca9320dca78e73676969f05c979b59bb705699f54953702314db66fbcff1c14c`.
- Disassembler: NVIDIA `nvdisasm` 12.4.127, package SHA-256
  `7fb724041f607718519ecb473995aca861db267749bd675a52b0f79702f33c00`.
- Kernel:
  `cc_tinlayout_fused_pre_block_swin_1h_32_1_fp8`.
- The first feature vector is committed to shared memory by `STS.128` at SASS
  offsets `0x19d0` and `0x19f0`. The isolated disassembly slice hashes to
  `eae0881f3ea5f81fa42f0e32f65dd47390537ecbe897cda87221f65fc0dfe13e`.
- Ghidra 12.1.3 identifies the NGX parameter reader at `0x180019f30`, feature
  evaluation at `0x180018620`, network guide binding at `0x180021bb0`, and the
  pre-layer launch builder at `0x180060cf0`. Decompiler output remains external.

No DLL, CuBIN, SASS dump, or recovered weights are stored in this repository.

## Recovered feature order

For each pixel, the first-frame feature vector is:

| Channels | Meaning |
| --- | --- |
| 0–2 | Three deterministic Gaussian values, rounded to FP16 |
| 3 | Constant `1` |
| 4–6 | Current RGB: `half((half(rgb) - 0.5) * 0.125)` |
| 7–9 | Current RGB again; this is the no-history fallback |
| 10 | Normalized style value |
| 11 | Local tone strength |
| 12 | Local structure strength |
| 13–14 | `-1` sentinels in the no-mask, auto-mask-disabled branch |
| 15 | Zero |

The RGB order was not inferred from register names. Tiny PTX probes were
assembled for `sm_89` and disassembled to establish `F2FP.PACK_AB`, `PRMT`, and
texture destination ordering. NVIDIA's PTX definition confirms that generic
`prmt` selects four bytes from `{b, a}` according to the four nibbles in the
control word: [PTX `prmt` semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt).

## Recovered mask branches

The raw Windows x64 call at `0x1800225a7..0x180022636` fixes all 18 arguments
to the guide-binding function. The final pre-kernel slice at SASS offsets
`0x1700..0x1910` then fixes the conditional values and their FP16 packing.
As a cross-check for the undocumented opcodes, pinned NVLift commit
[`99c1782`](https://github.com/purseclab/Sass-LLVM-Lifter/blob/99c1782267e520f1145be84264906831c253c2da/src/s2lir/Instruction.py)
lifts `FSEL a,b,p` as `p ? a : b` and `FMNMX a,b,p` as `p ? min(a,b) : max(a,b)`.
The disabled branch is `[tone, structure, -1, -1, 0]`; an independently
captured `t0-s0` feature tensor contains those literal sentinels.

With automatic masking enabled and no explicit mask, features 11–15 are:

```text
tone
max(skinStrength, autoMaskStrength) >= 0 ? 1 : localStructure
any strength enabled ? (skinStrength >= 0 ? skinStrength : localStructure) : -1
any strength enabled ? (autoMaskStrength >= 0 ? autoMaskStrength : localStructure) : -1
0
```

The host resolves `SkinStructureStrength < 0` to local structure and supplies
local structure as the automatic-mask structure strength. The Swift API keeps
the lower-level effective pair visible so the SASS sentinel branch remains
testable.

For an explicit ControlMask, the pre-kernel texture component mask is `0x6`
(green and blue), while the matching post-kernel variant adds a `0x1` red-only
fetch. The recovered full-rect contract is therefore:

```text
feature11 = half(mask.g * localTone)
feature12 = half(mask.b * localStructure)
blend     = saturate(mask.r * intensity)
outputRGB = saturate(inputRGB + blend * (predictedRGB - inputRGB))
```

`predictedRGB` is the already clamped neural prediction, so scaling the raw head
residual is not equivalent. The CLI and CPU reference implement this contract
for both same-size masks and point-transformed backing resources. The DLL clears
`UseAutoMask` when `ControlMask` is present; the public API gives the explicit
mask the same precedence.

Module 0 contains base, `simple_blend`, and `control_mask` post-kernel families,
each with full-rect and transformed-resource variants. Comparing their final
tails recovers one generalized order:

```text
temporalRGB = predictedRGB + alpha * (filteredHistoryRGB - predictedRGB)
factor      = saturate(intensity * (controlMask.r if present else 1))
outputRGB   = saturate(inputRGB + factor * (temporalRGB - inputRGB))
```

The control variant differs from simple blend only by an additional red-only
texture fetch and multiplication of the final factor. Green and blue remain
pre-head tone/structure controls. Thus intensity and mask red gate the completed
temporal result, not the raw residual or alpha alone.

## Deterministic noise

All integer arithmetic below wraps to 32 bits. For pixel `(x, y)` and the
kernel's frame counter `f`:

```text
seed = y * 0xd8163841
     ^ x * 0x8da6b343
     ^ f * 0x9e3779b9
     ^ 0x243f6a88

dynamic(v) = (v ^ (v >> ((v >> 28) + 4))) * 0x108ef2d9
mixed      = dynamic(seed) ^ (dynamic(seed) >> 22)
```

Four values are derived from `mixed` with affine salts. Each is passed through
`dynamic`; then `((v >> 30) ^ (v >> 8)) + 1`, scaled by `2^-24`, yields a value
in `(0, 1]`:

```text
a = uniform(mixed * 0xcaa5b80d + 0x21dd796b)   <!-- corrected 2026-09-02: the kernel multiplies by 0xcaa5b80d, not 0xcaa55b0d -->
b = uniform(mixed * 0x83232c31 + 0x3463e0ac)
c = uniform(mixed * 0x2c9277b5 + 0xac564b05)
d = uniform(mixed * 0xfa6dc5f9 + 0x4712a88e)
```

The stored values are:

```text
rA = sqrt(-2 * ln(a))
rB = sqrt(-2 * ln(c))

noise[0] = half(rB * cos(2*pi*d))
noise[1] = half(rB * sin(2*pi*d))
noise[2] = half(rA * cos(2*pi*b))
```

The SASS sequence is literal: `MUFU.LG2`, multiply by the float32 `ln(2)` and
`-2`, `MUFU.SQRT`, angle conversion through float32 `2π` and `1/(2π)`,
`MUFU.SIN/COS`, float32 multiplication, then `F2F.F16.F32`. NVIDIA's
[PTX approximation bounds](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-lg2)
specify maximum relative `lg2` error `2^-22` outside `(0.5, 2)`, relative
`sqrt` error `2^-23`, and absolute `sin/cos` error `2^-20.5` over this kernel's
angle range.

Those published bounds establish a conservative feature-space ceiling even
before a Windows oracle is available. Uniforms are in `[2^-24, 1]`, so the
largest exact Box-Muller radius is `5.768108`. Propagating the worst `lg2` error
and float32 multiply rounding gives at most `9.83979e-6` radicand error. The
square-root continuity bound plus approximate `sqrt`, trigonometric, and product
errors totals `0.00314166` before half conversion. Two independent worst-case
FP16 roundings add `0.00390625`, for `0.00704791`; the comparison ceiling is
therefore rounded up to the exact power of two `1/128 = 0.0078125` per stored
noise feature.

`dynamic` is the PCG `RXS-M-XS` output step (multiplier `0x108ef2d9`,
final `v ^ (v >> 22)`), whose variable shift is `(v >> 28) + 4`. An earlier
transcription omitted the `+ 4`; with a zero shift one input in sixteen folds
`v ^ v` to zero, its uniform collapses to `2^-24`, and the Box–Muller radius
saturates, which inflated the per-pixel noise standard deviation to `1.41` with
kurtosis `7.6`. Against the vendor's pooled-noise feature selectors of the
320×320 first-frame capture, the corrected generator matches channels 0 and 1
with correlation `0.9997` (standard deviation `1.00`, kurtosis `2.98`), and the
earlier form correlates at `0.01`. Channel 2 matches the angle `2π·b`
(correlation `0.786 = π/4`, the value expected for the right angle and an
independent Rayleigh radius), so its radius is drawn from a word other than
`a`, `b`, `c`, or `d`. The pre-kernel SASS (`pre-post-fp8.sass`, offsets
`0x470–0x930`) nevertheless computes exactly `sqrt(-2·ln a)·cos(2π·b)` for that
channel with the same salted words, so the disagreement lies between the
kernel and the pooled-selector capture rather than in the recovered formula;
neighbouring-pixel, XOR-swapped, pooled-coordinate, and block-aggregated radius
candidates all stay at or below correlation `0.87`. Channel 2 is therefore
implemented as the SASS reads and is statistically right but not proved
sample-exact against that capture.

This is a formal ceiling, not the expected NVIDIA error and not an end-to-end
RGB threshold. To measure model sensitivity, an ignored lab probe perturbed all
three half-rounded noise channels by `1/128`; subsequent FP16 rounding made the
actual maximum feature change even larger, `0.00878906`. Through this exact
checkpoint, all-plus and all-minus patterns changed first-frame display RGB by
at most `0.00103760`; channel-split by `0.00032043`; checkerboard by
`0.00013733`. Worst raw-head drift was `0.0363674`, confirming that head and
display tolerances are not interchangeable. Actual NVIDIA captures remain the
authority for the tighter observed distribution and final pipeline threshold;
literal tests continue to protect channel order, hash constants, wrapping
arithmetic, and first-frame fallback semantics.

## Recovered first-frame output composition

The four model-head channels are not RGBA. The first three are RGB residuals;
the fourth is a temporal blend logit. In the RGB post-kernel, the neural
residual is scaled by `1/32`, added to color represented at `1/8` scale, and
then converted back to display range. The first-frame expression therefore
simplifies to:

```text
displayRGB = clamp(inputRGB + 0.25 * half(head[0:3]), 0, 1)
```

The fourth channel is unused when no temporal reference is available. In the
temporal path it becomes a blend factor:

```text
alpha = clamp(sigmoid(head[3]) * half(blend_scale), 0, 1)
```

The recovered package's `blend_scale` is approximately `0.7397`. The
post-kernel then blends the neural RGB with a filtered reference color. This is
why exposing the raw four-channel head as an image was incorrect; the raw-RGB
CLI now applies `NeuralRenderingFirstFramePostprocessor` and writes three-channel
display RGB.

## Kernel parameter map

The host dispatch copies a 264-byte launch structure to constant bank 0 at
offset `0x160`. Relevant recovered fields are:

| Constant offset | Host meaning |
| --- | --- |
| `0x180` | Optional explicit ControlMask handle |
| `0x188`, `0x1a0`, `0x1b8`, `0x1d0`, `0x1e8` | Texture subrect/extent transforms |
| `0x200`, `0x204` | Motion-vector X/Y scales after extent normalization |
| `0x208` | Depth-inverted flag |
| `0x20c` | Local tone strength |
| `0x210` | Local structure strength |
| `0x214` | Style index scaled by `1/128` |
| `0x218` | Effective skin structure strength |
| `0x21c` | Effective automatic-mask structure strength |
| `0x220` | Automatic-mask path enabled |
| `0x224` | Fixed `1/16`; doubled to the RGB scale `1/8` |
| `0x228` | Frame counter used by the noise hash |
| `0x230`, `0x234` | Output height and width |
| `0x238` | Output tensor address |
| `0x240` | Packed model-weight address |

Texture operations identify five semantic inputs: current color, depth, motion,
reprojected history color, and optional control mask. The first-frame branch
copies current RGB into the history slots before it reaches depth/motion
sampling.

## Recovered host-side guide contract

The CPU path removes the remaining ambiguity for the common full-rect motion
contract. `MVecScaleX` and `MVecScaleY` default to `1`; the DLL preserves their
sign and writes these normalized factors to the launch block:

```text
motionFactorX = MVecScaleX / effectiveMVecWidth
motionFactorY = MVecScaleY / effectiveMVecHeight
historyUV     = currentUV + sampledMVec * motionFactor
```

`effectiveMVecWidth/Height` use the declared MVec subrect extent when nonzero,
otherwise the backing texture or active network extent. Each six-float texture
transform has literal form
`(baseX, baseY, extentWidth, extentHeight, 1/resourceWidth, 1/resourceHeight)`.
The pre-kernel launch block maps them as follows:

| Constant offset | Transform |
| --- | --- |
| `0x188` | Reprojected history |
| `0x1a0` | Motion vectors |
| `0x1b8` | Depth |
| `0x1d0` | Explicit control mask |
| `0x1e8` | Current color |

`NeuralRenderingAlignedSubrect` implements the exact integer-aligned subset in
which the declared extent equals the processing size. It extracts a compact
logical NHWC view using the backing resource row stride before preprocessing.
For the general history/motion case, `NeuralRenderingTextureTransform` retains
the six-field mapping and applies it to logical pixel centers and history taps.

The DLL resolves `cuTexObjectCreate` dynamically. Its constructor at
`0x180077a60` creates slot 0 with `filterMode=1`, slot 2 with `filterMode=0`,
all three address modes set to `1`, and flags `0x02`. The
[CUDA Driver API texture types](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)
map these values to linear, point, clamp, and normalized coordinates. Evaluate
selects slot 0 for reprojected history and slot 2 for current color, motion, and
ControlMask. The executable CPU, Metal, and MLX paths therefore linearly filter
history and point-sample motion with clamp-to-edge behavior.
`NeuralRenderingTextureTransform.pointSample` applies the same generic point
mapping to current color and ControlMask. `nrk run` uses it once before both
first-frame preprocessing and postprocessing, so red, green, blue, and source
RGB select identical logical pixels.

The earlier depth anomaly is now resolved for this DLL. It reads every
`DepthSubrect*` parameter, but the only call to `FUN_18003f490` passes both a
null depth texture handle and an all-zero depth transform. SASS checks the handle
at `c[0x178..0x17c]` and branches around all five depth fetches when it is null;
the X/Y selection offsets remain zero. Motion is therefore sampled at the
current pixel, and `DepthInverted` has no effect in the observed path. The
five-sample closest-depth code remains a structurally recovered, explicit mode,
not the default behavior of this surfaced build.

Two independent community implementations use the same practical convention.
The pinned [`DLSS5-Feeder` shader](https://github.com/jlrouzies-fr/DLSS5-Feeder/blob/be60b3f5cb607359930362723bc4e49f2065b9df/shaders/DLSS5_Feed.fx)
documents `prev_uv = uv + mv`, emits current-to-previous vectors in pixels, and
uses scale `(1, 1)`. The pinned [D3D12 video-player backend](https://gitlab.com/JessicaNataliaMods/dlss-5-video-player/-/blob/ae5203f498810c0e1ee44e51ed78fd259ac4effd/src/DLSSBackend.cpp)
does the same. These projects validate the host convention, not model-output
parity.

## Temporal branch recovered so far

The temporal branch and its history/motion texture sampling are now
structurally and executably recovered.

1. The observed build binds no depth guide, so motion is sampled at the current
   pixel. Its dormant structural branch would compare the current depth with
   four diagonals and retain the minimum or maximum according to
   `DepthInverted`.
2. The sampled motion is scaled by the two normalized MVec factors and added
   to the current normalized coordinate before history sampling.
3. History RGB is reconstructed with a five-sample Catmull–Rom approximation,
   centered, scaled by `1/8`, rounded to FP16, and placed in channels 7–9.

For fractional coordinate `t` in `[0, 1]`, the cubic weights are:

```text
w0 = -0.5*t + t*t - 0.5*t*t*t
w1 = 1 - 2.5*t*t + 1.5*t*t*t
w2 = 0.5*t + 2*t*t - 1.5*t*t*t
w3 = -0.5*t*t + 0.5*t*t*t
g  = w1 + w2
```

The hardware-linear middle coordinate is `base + w2/g`. Instead of a full
4×4 fetch, the kernel samples a cross of five positions:

```text
(left,  middleY)  weight w0x*gY
(middleX, top)    weight gX*w0y
(middleX, middleY) weight gX*gY
(middleX, bottom) weight gX*w3y
(right, middleY)  weight w3x*gY
```

Their weighted RGB sum is divided by the sum of those five weights. Outer
coordinates are clamped to valid pixel centers. This formula is evidence-backed
from the SASS dataflow. Pixel-unit sign, scaling, null-depth behavior, linear
history filtering, point motion sampling, normalized coordinates, and edge
clamping are now recovered. Jitter is caller-owned in this build and its host
conversion is executable; independent NVIDIA end-to-end numerical parity remains
open. The Swift temporal reference keeps normalized UV offsets as its portable
boundary.

## Executable temporal reference contract

`NeuralRenderingTemporalReferencePreprocessor` now defaults to the observed
null-depth behavior and applies motion from the current pixel. Its explicit
`closest-depth` mode retains the dormant five-sample structural branch for
experiments. The motion tensor is deliberately defined in normalized history-UV
offsets: positive motion is added to the current normalized pixel center. Color
and depth define the logical processing shape. By default history and motion
share it; configured `NeuralRenderingTextureTransform` values may instead map
them from different backing-resource extents.

`NeuralRenderingTemporalReferencePostprocessor` interprets the fourth head
channel as a sigmoid blend logit and combines the predicted RGB with the
reprojected history from feature channels 7–9. The default blend cap is the
recovered FP16 value `0.73974609375`.

`NeuralRenderingTemporalReferenceBackend` wraps any pluggable four-channel head
backend. It conforms to `NeuralRenderBackend`, owns display history, requires
consecutive frame contexts, resets on stream/discontinuity/shape/gap/failure
boundaries, and forwards only the generated 16-channel tensor to the wrapped
backend. The CLI exercises this path with `rgb-temporal-reference` sequences.

This contract makes the temporal math executable without falsely assigning a
meaning to arbitrary engine motion values. `normalizePixelMotion` implements the
recovered pixel contract, including signed scales, effective extents, and an
explicit previous-minus-current pixel jitter delta. Engines with a different
convention still need an explicit adapter. `nrk run-sequence` exposes the same
path as `--motion-format pixel --motion-scale-x X --motion-scale-y Y
--jitter-delta-x JX --jitter-delta-y JY`; normalized UV remains the default.

### Jitter ownership

The surfaced `nvngx_dlssnr.dll` and `sl.dlss_nr.dll` contain no NR jitter
parameter key. The complete core parameter reader, pre-layer setter ABI, wrapper
forwarding path, and recovered launch constants contain resources, transforms,
MVec scale, depth/reset, and controls but no jitter field. SASS computes history
coordinates from the current UV plus sampled MVec times its factor, with no
additive jitter constant. The adjacent DLSS upscaler plugin does contain
`Jitter.Offset.X/Y`, which makes its absence from the NR path material rather
than a string-search assumption.

The pinned public Streamline [`Constants`](https://github.com/NVIDIA-RTX/Streamline/blob/e8aaa6eaac968711fb62473d4ae8256dde20919b/include/sl_consts.h)
describe `jitterOffset` in pixels and distinguish motion vectors that already
include jitter; the public [DLSS programming guide](https://github.com/NVIDIA-RTX/Streamline/blob/e8aaa6eaac968711fb62473d4ae8256dde20919b/docs/ProgrammingGuideDLSS.md)
also keeps jitter outside its matrices. These neighboring public contracts help
interpret the common constants, but the NR ownership conclusion comes from the
NR binaries and launch dataflow: this build expects motion in the rendered,
jittered coordinate domain.

For unjittered current-to-previous pixel motion, the caller conversion is:

```text
jitterDelta = previousJitter - currentJitter
normalizedMotion = (pixelMotion * MVecScale + jitterDelta) / effectiveExtent
```

Jitter is already in pixels and is therefore not multiplied by `MVecScale`.
Already-jittered motion uses a zero delta. Both axes must be finite. A controlled
two-frame exact-checkpoint test compared zero motion plus `(1, 0)` jitter against
raw X motion `-0.5` at X scale `-2`; their retained-history output files were
byte-identical, while the pre-implementation run differed on frame two. This is
local algebra and routing proof, not an NVIDIA golden comparison.

`VisionOpticalFlowGuideProvider` supplies a native macOS video fallback. Apple's
[`VNGenerateOpticalFlowRequest`](https://developer.apple.com/documentation/vision/vngenerateopticalflowrequest)
revision 1 produced literal pixel motion in a controlled translation probe. With
the current frame as handler input and the previous frame as targeted input, a
four-pixel rightward translation measured approximately `-4` pixels, which the
core converter maps to `-4/width`. Revision 2 measured approximately `-5.45` on
the same `128×96` non-square input and is intentionally rejected for this
reference contract. The provider resets its retained frame on lifecycle breaks
and currently pairs motion with flat depth.

`MetalNeuralRenderingTemporalFeaturePreprocessor` is the first optimized
implementation behind the same pluggable protocol. It deliberately keeps noise
and current-frame assembly on the CPU oracle, optionally performs structural
closest-depth selection, and always performs five-tap Catmull-Rom history
sampling plus FP16 feature rounding in a runtime-compiled Metal kernel with fast
math disabled. A spatially varying-motion test matches CPU bytes in both modes,
and a nonuniform transform fixture matches CPU output byte-for-byte.

`NeuralRenderingCPUTemporalFeaturePreprocessor` and
`MLXNeuralRenderingDeviceTemporalBackend` expose the same optional history and
motion transforms at construction. The stateful MLX actor validates backing
resource dimensions before dispatch and forwards the values into MLXFast
kernels. It now generates base features on-device and fuses them with history
sampling after frame one. Non-noise base channels and fused-versus-sequential
device features are byte-exact; Metal deterministic noise is gated separately.

For first-frame backing resources, `nrk run` accepts the same literal transform
order as `--input-transform` and `--control-mask-transform`:
`baseX,baseY,extentWidth,extentHeight,resourceWidth,resourceHeight`. A 256×128
resource mapped to 128×128 selected the independently calculated odd columns;
the transformed and pre-sampled exact-checkpoint runs were byte-identical.

`nrk run-sequence` exposes `--input-transform`, `--motion-transform`, and
`--history-transform` with the same field order. Current color is point-sampled
once at ingress. Motion remains a backing-resource tensor so the selected CPU,
Metal, or MLX temporal implementation applies recovered point sampling;
pixel-format normalization uses the declared motion extent. Retained display
history stays in logical storage and accepts a transform whose resource size
matches that storage. In an exact-package three-run fixture, transformed
256×128 current/motion files matched independently sampled 128×128 files on
both frames, while a non-full history transform changed frame two versus
identity history.

The same command accepts one `--control-mask` per input, optional
`--control-mask-transform`, and global `--intensity`. Masks are point-sampled
once at ingress, green/blue enter temporal base features, and red gates the
completed temporal RGB after history blending. Mask appearance is intentionally
outside lifecycle descriptors, so it does not reset retained history. Portable
CPU/Metal and device-resident MLX paths execute the same checkpoint and stay
inside the measured device display tolerance below.

`run-sequence` also accepts `--style-index`, `--local-tone`,
`--local-structure`, `--auto-mask`, and `--skin-structure`. One shared
`NeuralRenderingFeatureControls` value now feeds the first frame and every
history frame through the CPU, standalone Metal, and fused device-resident MLX
paths. Explicit ControlMask input disables automatic masking. Focused tests
prove identical non-noise feature channels across all three implementations;
the first frame no longer silently falls back to default controls.

### Official model and control taxonomy (2026-09-01)

NVIDIA's [3D-Guided Neural Rendering article](https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/)
and accompanying [developer-controls video](https://www.youtube.com/watch?v=58FagrSqC4M)
resolve an ambiguity in the pre-release ABI: model selection and live controls
are separate axes.

- The UI offers choices labeled Models A, B, and C. The video selects Model B
  by default and shows visibly different outputs for A/B/C at identical
  structure strength. NVIDIA describes different parameter weights, but the
  material does not prove whether these are complete checkpoints, shared
  backbones plus adapters, or create-time conditioning.
- Global Structure Intensity controls high-frequency AO, contact shadows,
  reflections, and subsurface scattering. Global Tone Intensity controls
  low-frequency lighting and color response. The demonstrated nominal values
  are 1.00/1.00.
- Model Automask recognizes the base character and exposes an independent
  character strength. Engine-provided masks can define any number of groups;
  each group has its own structure and tone values.
- The model consumes more than final color in the production SDK: NVIDIA names
  renderer-internal buffers such as albedo, surface normals, lighting data, and
  motion vectors as 3D guidance. The leaked feature-18 build visibly binds only
  a subset of that production contract.

The inspected `nvngx_dlssnr.dll` contains `CG2RFindWeightByPreset`,
`ResolvePresetToDescriptor`, and explicit fallback logging for unavailable
presets. Its PE resource tree contains only one `WEIGHTS_HT` resource
(`147,695,410` bytes), and its descriptor strings name one embedded network,
`CC_Control_History_Blend_Quantize_With_Teacher_honest_tench_2026_07_04_22_30_weights`
with backbone `CC_SILVER_AARDWOLD`. This proves that the inspected build can
fall back to one shipping-default network for unavailable slots. It does not
prove how the production A/B/C choices share or replace weights.

### Community host parameter survey

The public feature-18 hosts do not agree on one default bundle, but they expose
the same NGX keys. The source snapshots inspected on 2026-09-01 are:

| Host | Default numeric model slot / style | Strength defaults | Mask default | Lifetime behavior |
| --- | --- | --- | --- | --- |
| [ComfyUI-DLSS5-NR](https://github.com/lisitskyaa/ComfyUI-DLSS5-NR) `4fb7b53` | preset 3, Natural/style 1 | intensity 1, tone 1, structure 1, skin -1 | off | rebuilds for style or model-preset changes; writes strengths again at evaluate |
| [Zonnery player](https://github.com/Zonnery/dlss5-nr-player) `e9c37be` | preset 3, Natural/style 1 | integer intensity 1, tone 1, structure 1, skin -1 | off | writes tuning at feature creation; reset changes on the first frame and seek |
| [Merserk visual enhancer](https://github.com/Merserk/dlss5-visual-enhancer) `365d159` | preset 0, Default/style 0 | intensity 1, tone 1, structure 1, skin -1 | off | builds a worker session from one validated settings bundle |
| OptiScaler experimental DLSSNR `d087111` | preset 0, Default/style 0 | intensity 1, tone 1, structure 1, skin -1; transfer 1, colour 1, ratio cap 2 | on | treats every model control as create-time state and rebuilds after a debounce |

All four set `DLSSNR.Hint.Render.Preset`, `DLSSNR.Style`,
`DLSSNR.Intensity`, `DLSSNR.LocalToneStrength`,
`DLSSNR.LocalStructureStrength`, `DLSSNR.SkinStructureStrength`, and
`DLSSNR.UseAutoMask`. ComfyUI and Zonnery call style 0 `default`, style 1
`natural`, and style 2 `cinematic`. OptiScaler independently characterizes
style 0 as the strongest standard profile, style 1 as gentler, and style 2 as
film-oriented. Community UIs expose numeric model slots 0 through 3: slot 0 is
shipping default, while slots 1/2/3 are the only plausible carriers for the
official A/B/C selection. Their exact ordering is not published and must be
verified from create logs and distinct weight hashes.
The selection is latched when feature 18 is created and is not a live input
channel. The inspected one-network DLL may fall back for unavailable slots, so
the apparent selection still requires a create log and weight hash to prove it.

The v19 checkpoint in this repository was captured with shipping-default model
slot 0 and the
same nominal style-0, intensity-1, tone-1, structure-1, skin--1, auto-mask-off
bundle used by the Windows oracle. `NeuralRenderingControlProfile.standard` is
therefore the default live-control profile. `natural`, `cinematic`, and
`neutral` change controls within that same shipping-default checkpoint.
Calling one of them Model A/B/C would be false. Complete support first requires
four create-time captures (0/1/2/3), descriptor and weight-upload hashes,
dispatch comparison, and matched visual outputs. Only differing weight bundles
would justify three separately published checkpoints.

## Local execution evidence

On a Mac14,6 running macOS 26.5.2, a release build processed the deterministic
`128×128×3` RGB pattern `value[i] = (i mod 257) / 256` through the first-frame
preprocessor, complete external eager MLX package, and recovered display
postprocessor five times in fresh processes.

- Preprocessor median: `1,396,667 ns`.
- Backend execution median across fresh processes: `117,585,250 ns`.
- Postprocessor median: `512,542 ns`.
- All five `128×128×3` outputs were byte-identical, SHA-256
  `8f39b9cd18e93764d532adfeccc0210d6a145e19dec8e140363cd4f909826c5c`.
- All 49,152 output values were finite and lay in `[0.0012846, 1]`.

These measurements prove deterministic local plumbing. They are neither a
steady-state throughput benchmark nor a golden comparison with NVIDIA.

For the temporal path, a ten-frame `128×128` release sequence with zero
normalized motion and flat depth was repeated three times in fresh processes.
The median run took `519,133,834 ns` end to end and `383,985,084 ns` inside the
MLX head backend. That is approximately `51.91 ms/frame` (`19.26 fps`) end to
end versus `38.40 ms/frame` (`26.04 fps`) in the head backend; the scalar CPU
reference plus file I/O contributes roughly `13.52 ms/frame`. All ten frame
outputs were byte-identical across all three runs. The first and last hashes
were respectively
`8f39b9cd18e93764d532adfeccc0210d6a145e19dec8e140363cd4f909826c5c`
and
`bdaa1b2dbb3cf7424c6533c0498591afb796ae60284fe0fa2745727a4f8d88f1`.

This is already interactive at the tiny reference size, but it misses 30 fps.
The next optimization target is therefore measured: move feature assembly and
five-tap history sampling off the scalar CPU before touching lower-impact code.

A five-second `sample` profile confirmed that `sampleLinear`, repeated
`Data`→array conversion, and short-lived RGB/weight arrays dominated the
non-MLX path. Replacing per-pixel arrays with fixed RGB tuples and writing the
16-channel output into preallocated storage preserved every output byte. In
three follow-up ten-frame runs, median end-to-end time fell to
`477,417,126 ns` (`47.74 ms/frame`, `20.95 fps`). Median non-backend overhead
fell from about `13.52` to `6.74 ms/frame`. Backend variance prevents treating
the overall delta as a controlled A/B speedup, but the isolated CPU reduction
is approximately two-fold and the before/after output directories compare
byte-for-byte equal.

The explicit FP16 mode, after its expensive first initialization, ran the same
sequence in `45.62–46.66 ms/frame`. Relative to FP32, maximum absolute output
drift across ten frames was `0.0006734`, mean absolute drift was approximately
`0.0000802`, and mean squared error was approximately `1.09e-8`. FP16 remains
an opt-in experiment until NVIDIA oracle tolerances are available.

The same external package now runs through the packaged macOS player rather than
only the raw-tensor CLI. A release, ad-hoc-signed app decoded a ten-frame,
`128×128`, 5 fps H.264 control clip derived from the temporal inputs, supplied
zero normalized motion plus flat depth, retained consecutive frame context, and
displayed frame 9 at `53.0 ms` with zero dropped frames. The process emitted no
runtime error. The saved 960×572 proof screenshot has SHA-256
`1e02f9708c8d5dc00bada9fb307053179701d28139d4b522277270c9ac333a2c` and
remains an external test artifact.

That smoke run also exposed a deployment issue invisible to `swift run`: MLX
loads `mlx.metallib` relative to the executable, while the first app bundle
contained only the Mach-O binary. `scripts/package-player-app.sh` now copies the
generated library into `Contents/MacOS`, signs the Metal library before signing
the enclosing app, and verifies the nested signature. The full verification
script checks that resource explicitly. This proves portable app packaging on
the local Mac; it is not notarization or a distributable release signature.

For the ten-frame `128×128` temporal benchmark, CPU and Metal preprocessing were
run in an alternating A/B/A/B/A/B sequence. Median results were:

| Preprocessor | End to end | Non-head overhead |
| --- | ---: | ---: |
| CPU | `45.27 ms/frame` | `5.88 ms/frame` |
| Metal | `42.92 ms/frame` | `4.40 ms/frame` |

All corresponding outputs were byte-identical, including the established first
and last frame hashes. Metal reduced median non-head overhead by approximately
`1.47 ms/frame` and end-to-end latency by approximately `2.35 ms/frame`. The
kernel pipeline was created before the timed frame loop, so runtime compilation
is not hidden in these values. The remaining cost is architectural: the optimized
kernel still returns a materialized `HostTensor`, which is then uploaded again by
the MLX head.

Before this final A/B run, `MLXNeuralRenderer` also stopped materializing an
intermediate Swift `[Float]` and now constructs its copied MLX input directly
from `Data`. Relative to the preceding snapshot, median non-head overhead fell by
about one additional millisecond per frame on both CPU and Metal paths. This is a
host-allocation reduction, not the zero-copy handoff described below.

MLX Swift's managed raw-pointer boundary was independently characterized with a
page-aligned allocation. A post-construction memory change was observed by the
next MLX operation without invoking the fallback finalizer, proving that a shared
Metal allocation can become an `MLXArray` without a host copy.

`MLXNeuralRenderingDeviceTemporalBackend` now removes the feature readback and
re-upload entirely. Base features are generated by one MLXFast Metal kernel on
the first frame; later frames fuse base generation and five-tap history sampling
in one dispatch, with no intermediate 16-channel buffer. The 71-block head,
recovered display composition, and retained display history stay as MLX arrays
until final RGB materialization. The public video factory, player, and
`run-sequence` expose this as `device-resident`; `portable` remains the
independently readable CPU-base oracle.

Two numerical details were required for literal eager equivalence. The temporal
feature Metal kernel disables contraction and reassociation. The head logit is
rounded to FP16 before sigmoid, so the postprocessor uploads one complete
65,536-entry float32 alpha table computed with the CPU oracle instead of relying
on a different Metal `exp` implementation. Final blend contraction is disabled.
The focused history, transform, sigmoid, and postprocess operations match their
portable inputs byte-for-byte. Direct base tests require channels 3–15 exactly
and apply the formal `1/128` feature ceiling only to deterministic noise.

Five alternating ten-frame release runs measured:

| Exact pipeline | Median end to end | Relative result |
| --- | ---: | ---: |
| Portable + Metal preprocessor | `44.37 ms/frame` | baseline |
| Device-resident eager | `39.67 ms/frame` | `11.86%` faster |

All corresponding exact-mode output files were byte-identical. The
device-resident path reaches approximately `25.21 fps` at `128×128`; it is not a
30-fps result.

Whole-graph `mlx.compile` still crashes the current runtime. The explicit
`block-compiled` mode instead compiles eight regular window-block sequences and
uses the non-precise softmax variant there; split-window and global-block
compilation were removed after full-pipeline A/B runs failed to improve latency.
Constant residual sines and global zero attention bias are retained once instead
of reconstructed per frame. In five final alternating pairs, device-resident block compilation
measured `36.97 ms/frame` (`27.05 fps`) versus `40.09 ms/frame` eager, an `8.42%`
speedup in that run. Across ten temporal frames its maximum absolute error was
`0.0001921`, mean absolute error `5.11e-8`, and minimum PSNR `112.86 dB`. It is an
opt-in fast path, not the correctness oracle or a real-time claim.

A same-state binary factorial compared the original 46 per-block compiled
functions with eight compiled regular sequences. Sequence compilation measured
`36.59 ms/frame` versus `37.33 ms/frame`, a `1.09%` improvement, and all ten
outputs were byte-identical between the two fast strategies. Compiling whole
split/global stages looked much faster in isolation but was `3.95%` slower in a
full-pipeline factorial, so those variants were removed.

The separate `int8-fast` mode keeps regular sequence compilation and quantizes
only the eight global FFN expansion matrices using affine INT8, group size 32.
Quantizing both expansion and projection failed the head quality gate; MXFP8 had
substantially larger single-matrix error and was also rejected. In a five-run
A/B/C temporal matrix, eager, `block-compiled`, and expansion-only `int8-fast`
measured `40.80`, `37.15`, and `36.80 ms/frame` respectively. INT8 reached
`27.17 fps`; relative to eager its ten-frame maximum absolute error was
`0.0003025`, mean absolute error `4.65e-5`, and minimum PSNR `83.20 dB`.

A matched FP16 device-resident experiment did not beat FP32: its median was
`41.93 ms/frame`. Its ten-frame maximum absolute drift versus FP32 was
`0.0006734`, mean absolute drift `0.0000802`, and minimum PSNR `78.74 dB`; FP32
therefore remains the default.

A fresh 100-frame release profile on the current toolchain measured CPU base
feature construction at `1.0595 ms/frame`. Moving it to a separate GPU dispatch
reduced host overhead but improved total latency only `0.149 ms`, so that form
was rejected. Fusing base generation into the steady history kernel measured
`25.9406 ms/frame` end to end versus `26.9105 ms/frame` for the CPU-base device
path, a `0.9699 ms` (`3.60%`) improvement and `38.55 fps` at `128×128`.

The acceptance matrix used 100 different color, motion, depth, and ControlMask
frames with intensity `0.8`. Across 4,915,200 composed RGB values, fused device
versus portable CPU-base plus Metal-history error was maximum `0.000151664`, mean
`1.16e-8`, and RMSE `7.23e-7`; the device test gate is `0.0002`. Direct non-noise
features, fused/sequential GPU features, history filtering, and postprocessing
remain exact. On the same varying fixture, device measured `27.4498` versus
portable `29.8192 ms/frame` (`7.95%` faster). These are real-time results only at
the reference processing size, not at native video resolution.

An opt-in release video test then exercised the public proxy boundary with 48
1920×1080 BGRA frames, 128×128 fused device processing, block compilation, and
processing-size output. It measured p50 `27.3028 ms` and p95 `28.5632 ms`, below
the `41.6667 ms` 24-fps budget. The view can scale that processing buffer to its
Metal drawable without first creating a full-resolution intermediate.

The rejected source-size variants are equally important: default vImage
down/upscale measured p50/p95 `66.4406/71.7727 ms`; retained Metal Core Image
measured `65.2681/70.8237 ms`. Therefore the documented real-time player mode is
1080p input with `--output-size processing`, not native-1080p inference or a
1920×1080 output pixel buffer. The performance test skips debug builds and must
be run in release with `mlx.metallib` beside the release XCTest executable.

## Independent PyTorch and Core ML head

`Tools/neural_rendering_reference.py` independently expresses all 71 recovered
blocks in PyTorch using only locally reviewed operations and external logical
safetensors. Primitive tests cover gate, cosine residual, attention, window
ordering, split/global blocks, pooling, retained-skip merges, and upsampling.
The earlier `2.2888e-5` maximum PyTorch↔MLX result belonged to the pre-v19 graph
and is no longer a current parity claim. With v19, both implementations execute
the complete graph, but the deterministic 65,536-value head comparison still
fails. After the corrected fused-upsample checkpoint and recovered bit-affine
attention, the current maximum absolute difference is `7.012697`, mean absolute
difference `0.570492`, and RMSE `1.166860`; all 65,536 values are finite. Moving
the half-FMA quadratic gate into a third native Metal dispatch improves those
figures to maximum `6.928343`, mean `0.441243`, and RMSE `0.867063`. This still
does not close the gate. Matching SM89 QMMA accumulation and the remaining
normalization reductions without platform-dependent rounding remains required
before either path can be called an exact portable oracle.

`Tools/convert_neural_rendering_coreml.py` wraps the NHWC graph in a fixed NCHW
`16→4` boundary and emits an external ML Program. Two trace-compatibility changes
were required without changing parity: dtype-safe tensor maximum replaced a
scalar `clamp_min`, and 2×2 downsampling became the sum of four rank-4 strided
slices because Core ML rejects rank-6 tensors. Focused conversion regressions
cover both cases. The generated float16 package was `279 MiB`; it contains the
user-supplied weights and remains outside the repository.

Core ML CPU+GPU loaded the package, returned finite `(1,4,128,128)` output, and
matched the PyTorch head with maximum error `0.04158`, mean error `0.00601`, and
RMSE `0.01076`. After two warmups, 20 direct predictions had p50 `17.85 ms`
(`56.0 fps`). The first prediction took `276.7 ms`; model load took about
`3.85 s`. A CPU+NeuralEngine load remained in ANE compilation for more than six
minutes and was terminated; CPU+GPU is the only measured recommendation.

The Swift Core ML backend now supports different fixed input/output channel
counts, and `NeuralRenderingReferenceCoreMLVideoRenderer` connects this head to
the existing temporal lifecycle. Three fresh 100-frame release CLI runs with
Metal preprocessing measured `26.29`, `24.67`, and `24.64 ms/frame` end to end,
for a median `40.53 fps`. All 300 outputs were byte-identical across runs. Against
exact portable MLX over the same 100 frames, maximum temporal error was
`0.0008724`, mean absolute error `3.88e-5`, mean RMSE `6.02e-5`, and minimum PSNR
`83.32 dB`. The packaged player accepts the Core ML head explicitly; UI
inspection could not be completed in this run because the Mac screen was locked,
but the app process launched and was then terminated cleanly.

## External golden bundle

Private NVIDIA captures can now use a checked, portable directory contract
without entering the repository:

```json
{
  "schemaVersion": 1,
  "height": 128,
  "width": 128,
  "motionConvention": "normalized-history-uv-offset",
  "depthInverted": false,
  "frames": [
    {
      "jitterDeltaPixels": [0.0, 0.0],
      "color":  {"file": "color-000.f32",  "sha256": "..."},
      "motion": {"file": "motion-000.f32", "sha256": "..."},
      "depth":  {"file": "depth-000.f32",  "sha256": "..."},
      "output": {"file": "output-000.f32", "sha256": "..."}
    }
  ]
}
```

Every filename must be one safe relative component. The validator checks all
digests and exact RGB/RG/R byte counts before comparing numbered candidate
outputs:

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

The report aggregates finite counts, maximum and mean absolute error, MSE,
RMSE, PSNR, and per-frame metrics. The bundle remains external; only the schema,
validator, and synthetic unit tests are publishable. A capture may alternatively
declare `motionConvention: "pixel-current-to-previous"`; that form must also
record finite `motionScaleX/Y` and positive `motionWidth/Height`, and the
candidate run must use the matching `nrk run-sequence` pixel-motion options.
Pixel-motion frames may record finite two-number `jitterDeltaPixels`; absent
values mean zero. The candidate runner preserves those values in frame order,
prints its exact command in dry-run mode, and executes that same vector otherwise.
It currently rejects motion extents unequal to the processing size because the
schema has no backing-resource transform to justify them.

`make_neural_rendering_golden_bundle.py` discovers contiguous six-digit output
indices and requires matching `color`, `motion`, `depth`, and `output` raw files
for every frame. It verifies exact byte counts, rejects symlinks and an existing
manifest, writes relative filenames plus SHA-256 atomically, and self-validates
through the same loader used by comparison. It never copies private pixels into
the repository.

### Windows NVIDIA oracle bootstrap

Pinned community source
[`DLSS5-Feeder@6a5d259e`](https://github.com/jlrouzies-fr/DLSS5-Feeder/tree/6a5d259e19b0e7d73930fb646313b5f07cf9aa4d)
provides a useful, independently published bootstrap for the missing Windows
half. Its
[`dlss5-feed-host64.cpp`](https://github.com/jlrouzies-fr/DLSS5-Feeder/blob/6a5d259e19b0e7d73930fb646313b5f07cf9aa4d/host/dlss5-feed-host64.cpp)
creates a standalone D3D12 device/swapchain, initializes NGX, creates a same-size
DLAA feature, supplies color/depth/motion resources, and repeatedly evaluates it
so the DLSS 5 add-on can attach. This is closer to an oracle harness than a game
mod because its `--test` mode needs no game process.

It is not yet a golden producer: the pinned test uses synthetic GPU textures,
depends on the external RenoDX add-on to route DLSS feature 1 into feature 18,
and neither uploads this repository's raw fixtures nor reads the neural output
back into the manifest format.

The independently published
[`ComfyUI-DLSS5-NR@4fb7b53`](https://github.com/lisitskyaa/ComfyUI-DLSS5-NR/tree/4fb7b53a11bb7ddce1824e8a0a5d2fcfa8e43521)
bridge is another useful host-side cross-check. It creates feature 18 directly
on D3D12, stages float32 RGB through RGBA16F textures, supplies full color and
output subrects, and preserves internal state when reset is clear. Its observed
parameter set includes `UICorrection=0`, `DepthInverted=1`, unit scaling and
motion-vector scales, but no depth or motion resource. It delegates all neural
work to `nvngx_dlssnr.dll`; it therefore helps validate the Windows oracle and
null-guide host contract, not the extracted graph or standalone inference.

### RunPod runtime boundary

A bounded Linux/Proton experiment subsequently ran the exact private snippet on
an RTX 3090 and an RTX PRO 4000 Blackwell. The snippet exports minimum GPU
architecture `0x1b0`; the 3090 reports `0x170` and was rejected before feature
creation. Blackwell passed the hardware gate. Direct D3D12 stopped at an
unimplemented `NvAPI_D3D12_CreateCuModule`. D3D11 provided a narrower path:
DXVK already implements named CuBIN creation, while dxvk-nvapi lacked only
`NvAPI_D3D11_CreateCubinComputeShaderExV2` and
`NvAPI_D3D11_GetCudaMergedTextureSamplerObject` discovery.

A source-built dxvk-nvapi compatibility shim added those two ABI surfaces. Its
Windows unit binary passed 31,698 assertions under CrossOver. On Blackwell, the
private core then created and destroyed 69 kernels. Every observed ExV2
structure had size 80, `dynSharedMemBytes == 0`, and `flags == 0`, so the
existing named-CuBIN DXVK path is semantically sufficient; no extended shared
memory launch is required. The names divide into clear preprocessing families:
clear/reduction/histogram/auto-exposure/luma, HDR/LDR and motion variants,
downsample/Lanczos, and input/output depth/motion variants. No kernel was
launched because snippet initialization still rejected the direct caller.

The remaining platform error now has a concrete source-level explanation. The
pinned
[`OptiScaler_DLSSNR@d087111`](https://github.com/liaokuokk/OptiScaler_DLSSNR/tree/d08711107b799f9e3db3846c766b6d8aa09c72d9)
integration documents that the snippet resolves its caller's return address and
requires the owning module path to contain `nvngx.dll`. Its
[`forwarder README`](https://github.com/liaokuokk/OptiScaler_DLSSNR/blob/d08711107b799f9e3db3846c766b6d8aa09c72d9/OptiScaler/dlssnr/forwarder/README.md)
also records the load-bearing no-tail-call rule. The direct oracle called from
an EXE and therefore necessarily received `FAIL_PlatformError` even after all
69 kernel objects were valid. An independently built, correctly named source
forwarder and the driver's own capability parameter block subsequently passed
that caller gate: D3D11 changed from `FAIL_PlatformError` to
`FAIL_FeatureNotSupported`. It returned no handle, which falsifies D3D11 as the
execution API rather than merely exposing another caller-name failure.

The remaining D3D12 gap was the seven-function CUDA module lifecycle: module
creation, function enumeration and creation, packed/raw kernel-chain launch,
and function/module destruction. External, ignored lab forks of exact
dxvk-nvapi and vkd3d-proton revisions now implement that bridge. The bounded
ELF64 parser recognizes the actual 1,946,976-byte `EM_CUDA` module, enumerates
38 global functions, resolves the recovered pre/post kernel names, and derives
the pre-kernel's one raw parameter from `.nv.info`. The full dxvk-nvapi unit
binary passed 31,771 assertions in 21 cases; a focused run with the actual
external module passed 3,525 assertions in two D3D12 cases. A hash-locked
installer was also tested against the exact pinned GE-Proton archive and
replaces only its x86-64 dxvk-nvapi and vkd3d-proton source slots before prefix
creation. These compatibility changes and all proprietary inputs remain outside
the public tree.

The external D3D12 bridge now creates Feature 18, executes the complete observed
kernel sequence, uploads the checked guide resources, and reads RGB float32
output. The proprietary runtime, captures, and bridge binaries remain outside
the public tree. This changes the next task from runtime bootstrap to numerical
graph recovery and controlled parity cases.

### Checkpoint graph recovery update, 2026-09-01

The external `dlssnr-logical-v17` conversion contains 649 ordinary logical
tensors, decodes all 153 source tensors, and has SHA-256
`214a81d31494103bad7b23edb0158c2da14f80b5a476aebe524760ad50780980`.
The corresponding local package is called v19; neither artifact is part of the
repository.

The new vendor evidence corrected several graph-level assumptions:

- split and global QKV matrices are stored head-interleaved and must be grouped
  into semantic Q, K, and V columns;
- global Q is normalized, multiplied by `sqrt(32)` and its learned head scale,
  then republished as E4M3; K is normalized and V remains the projected value;
- the global attention kernel consumes those published tensors without another
  `1/sqrt(32)` divisor;
- the four-phase window-origin cycle is `(0,0)`, `(-4,-4)`, `(0,-4)`,
  `(-4,0)` in `(y,x)` order;
- block 30 pads 12×12 to 16×16 before pooling to 8×8 and retains its unpooled
  12×12 output across the global stage;
- the global entry and exit repacks are exact inverse bit permutations;
- block 39 nearest-doubles 8×8, crops back to 12×12, and adds the retained
  block-30 skip multiplied by `inp_upsample_sin`;
- fused-upsample payloads store grouped FFN expansion, FFN output projection,
  and the latent upsample projection in that physical order;
- fused-upsample FFN residuals own the complete `upsampled latent + skip×sin`
  merge, not the skip path alone;
- window attention publishes Q, K, V, probabilities, and the attended value as
  E4M3 and uses a half bit-affine exponential approximation instead of ordinary
  framework softmax;
- the executable structural schedule pools the projected adapter before block
  0 and nearest-doubles once more before the block-70 full-resolution merge.

Paired same-run captures give these independent stage results:

- block 30, after the complete recovered layout: about `0.259` MAE;
- global FFN expansion: `0.00042` MAE and `97.7%` byte agreement;
- global FFN contraction plus residual: `0.00131` MAE and `98.3%` byte
  agreement;
- global Q/K/V publication: `0.00007–0.00092` MAE and `98.8–99.1%` byte
  agreement;
- global blocks 31–38 on their own vendor inputs: `0.0336–0.1544` MAE;
- block-39 bridge: `0.000535` MAE and `98.39%` byte agreement;
- regular decoder blocks 49–54: `0.052–0.079` MAE; blocks 57–61:
  `0.133–0.160`; blocks 63–65: `0.299–0.327`.

The corrected full native first-frame runner starts from the ordinary captured
frame, constructs the 320×320 feature extent, executes the same checkpoint, and
composes the first 128×128 display region. With the structural pool-then-block
schedule and nearest final upsampling, v19 reaches RGB correlation
`0.99978115`, mean absolute display error `0.00447982`, RMSE `0.00620430`, and
maximum error `0.05103505`. A block-then-pool/bilinear variant compensates
numerically to correlation `0.99995269` and mean error `0.00214789`, but conflicts
with the recovered structural schedule and is not treated as proof.

A later full-frame gate removed another false fixed-shape assumption. At
`512×512`, the vendor pre-kernel used a `72×64` grid and the global 2D-to-1D
repack launched over 96 tokens; all eight global-attention launches retained
that variable token extent. The bit-affine softmax is therefore not limited to
one 64-token array. Its Metal implementation now recomputes weights in a second
pass instead of storing a fixed local array, preserving the recovered half
accumulation while accepting any positive even token count. Large pointwise
FFNs are row-chunked independently, so attention retains complete-frame
semantics.

With those changes, an untouched `3840×2160` face frame completed through the
same v19 checkpoint on an M2 Max. Vendor alignment produced a `2176×3840`
network extent; float16 `metal-fused` execution took `6.80 s`, of which `5.60 s`
was the 71-block head. On the matched NVIDIA output, vendor-versus-input MAE
was `0.016803`, Mac-versus-input `0.013764`, and Mac-versus-vendor `0.018134`.
The visual gap remains obvious on the face. It does not originate in the raw
tone/structure control magnitude: after the fused pre-kernel plus block 0,
standard-versus-neutral MAE was `0.108385` on NVIDIA and `0.108286` locally.
The first material control-conditioned divergence is therefore later than
block 0.

Direct block-48 CuBIN execution then isolates the remaining numerical gap. The
correct matrix order and merged residual reduce the ordinary high-level block
candidate from `1.90551` to `0.08243` MAE. A literal SM89 QMMA runner plus the
recovered half-FMA gate reproduces the entire FFN path byte-for-byte. Extending
it through the bit-affine attention gives `91.8447%` byte agreement, `0.00588972`
MAE, `0.0325465` RMSE, and correlation `0.9999365` for the complete natural
block. The remaining difference is confined to exact Q/K normalization,
padding, and attention reduction details rather than weight routing or graph
ownership.

The same conclusion now covers every fused-upsample width. Standalone launches
of blocks 48, 56, 62, and 66 are byte-identical to their paired vendor captures.
At each width, independently zeroing the expansion or contraction region yields
the same output, while zeroing the skip input with the FFN branch disabled still
leaves a nonzero latent residual. This independently proves both the physical
`FFN → projection → upsample` payload order and merged-residual ownership across
all four decoder transitions.

The same v19 weights execute all 71 blocks through both the local MLX model and
the public package backend on Apple Silicon. Focused Python tests, focused Swift
operator tests, the full local model, and the package backend pass. Exact
PyTorch↔MLX agreement, the remaining split-decoder and fused-upsample numerical
gaps, and NVIDIA pixel parity remain open.

A fresh device-resident run then used the private, checked two-frame NVIDIA
`t0-s0` capture with pixel motion scale `(1,1)`, style `0`, tone `0`, structure
`0`, no mask, and no jitter. Frame 0 reached correlation `0.999736422`, MAE
`0.004941570`, RMSE `0.006875903`, and maximum error `0.055357993`. Frame 1
reached correlation `0.999737843`, MAE `0.004656801`, RMSE `0.006790468`, and
maximum error `0.079196393`. Across both frames, MAE was `0.004799186`, RMSE
`0.006833319`, maximum error `0.079196393`, and PSNR `43.307 dB`.

This is the first end-to-end NVIDIA two-frame measurement, not a passing parity
gate. Rendering frame 1 as a reset frame produced MAE `0.004679122` and RMSE
`0.006632329`; retained history slightly improved MAE but worsened RMSE and the
maximum error, so temporal blending is not yet independently proved. The same
matrix also falsified control-response parity:

| Vendor variant | Frame 0 MAE | Frame 1 MAE |
| --- | ---: | ---: |
| tone `1`, structure `0` | `0.030224286` | `0.014319204` |
| tone `0`, structure `1` | `0.040359043` | `0.030652577` |
| auto-mask, skin `0` | `0.042364735` | `0.034515232` |
| full RGB ControlMask | `0.038972568` | `0.032195050` |

The controls are now routed correctly and identically on Apple; the remaining
error is in the checkpoint's control response through the still-inexact early
graph and downstream arithmetic. Empirical feature rescaling is rejected
because it improves isolated cases while contradicting the recovered launch
constants and fails other variants.

A differential block-0 check on the RTX oracle narrows that statement further.
For tone, the recovered structural candidate and vendor have mean absolute
block-0 deltas `0.0857389` and `0.0848053`; for structure they are `0.0569031`
and `0.0556724`. Candidate-versus-vendor delta correlations are `0.9098` and
`0.8207`. The control-channel scale and input-adapter response are therefore
substantially correct. The remaining pixelwise delta MAE, about `0.0246` for
both variants, points to exact early-block spatial/numerical semantics rather
than a missing scalar control transform.

Stage isolation with the vendor-pooled feature selectors confirms the same
boundary. The v19 input adapter reaches `68.56%` byte agreement, `0.002936` MAE,
and correlation `0.999294`. Adding only the FFN residual yields `54.52%` byte
agreement, `0.009733` MAE, and correlation `0.999097`. The pure attention branch
has `20.08%` byte agreement and `0.009508` MAE; after its recovered cosine
residual the correlation rises to `0.995180` at `0.010845` MAE. Weight routing,
control-channel scaling, and coarse block ownership are therefore no longer the
leading hypothesis. Exact HMMA/QMMA accumulation, Q/K normalization, and
attention reductions in block 0 are the next narrow gate.

The same integration exposed a separate display codec surrounding the model:
linear HDR is preserved untouched, mapped to an sRGB proxy with a soft highlight
knee, and the complete model picture is folded back through luminance-ratio
composition instead of an inverse tone curve. That contract is now implemented
independently as `NeuralRenderingDisplayCodec` and as device-resident
`MLXNeuralRenderingDisplayCodec` Metal kernels. Focused Swift tests cover both
ratio branches, exact zero-strength passthrough, proxy encoding, and CPU/Metal
agreement. The RenoDX-derived ratio/OkLab design is credited under its MIT
license in `NOTICE`.

### Single-head window-attention bias layout, 2026-09-02

Teacher-forced vendor stage captures of blocks 1–4 (logical `[1,160,160,32]`
inputs and outputs of the 320×320 first-frame run) isolated the largest
per-block error to the attention mixing: the error concentrated in a handful of
output channels, correlated `0.906` with intra-window token variance, and
disappeared on flat windows, while removing the bias entirely *improved* every
block. Least-squares recovery of the effective `64×64` mixing and a learned
bias through the unchanged pipeline reached MAE `0.0064` on block 1, and an
exhaustive search over permutations of the twelve `(query, key)` token-index
bits against that learned table found one clean solution:

```text
stored = ry2<<11 | rx2<<10 | ky2<<9 | kx2<<8 | ry0<<7 | rx1<<6 | rx0<<5
       | ky0<<4 | kx1<<3 | ky1<<2 | ry1<<1 | kx0
```

The single-head tables are stored in the fused kernel's `mma` fragment order
(`[query quadrant][key quadrant]` tiles of `16×16`, then an 8-row group and the
C-fragment register/column pairs), and the logical decoder had reshaped them
row-major. The map was fitted on blocks 1 and 3 and confirmed out of sample on
blocks 2 and 4; `NeuralRenderingAttentionBiasLayout` (Swift) and
`recover_attention_bias_layout` (Python) apply it at load time to every
single-head window block except block 0:

| block | before MAE / corr | after MAE / corr |
| --- | --- | --- |
| 1 | `0.0826` / `0.9785` | `0.0050` / `0.9998` |
| 2 | `0.1216` / `0.9889` | `0.0166` / `0.9996` |
| 3 | `0.1922` / `0.9837` | `0.0199` / `0.9997` |
| 4 | `0.3643` / `0.9707` | `0.0205` / `0.9999` |

The 2/4/8-head tables are already local as decoded and must not be remapped;
block 0 and the 16-head split blocks match neither layout and remain open. The
end-to-end face gate did not move with this fix alone (Cyberpunk Mac-versus-NVIDIA
`0.02650` → `0.02627`), so the dominant remaining error sits in the other block
families and is being isolated with the same golden-driven method.

### Branched feed-forward publication points, 2026-09-02

The vendor's natural block-5 captures (FFN branch and FFN residual, plus the
paired global launches 57–62) isolate three E4M3 publication points that the
reference had left in fp16 inside the multi-head (2/4/8-head) window blocks:
the gated expansion before the branch projection, the per-head branch sum
before the output projection, and the FFN residual before the attention and
skip read it. With them the block-5 FFN residual moves from `0.0386` MAE and
`80.7%` exact bytes to `0.0077` and `95.3%`, and teacher-forced blocks 5–14
improve `2.1–3.5×` (fitted on block 5 only; blocks 6 and 8–14 are out of
sample). The single-head `window_block` keeps its residual in registers:
rounding it there is `2.5×` worse on blocks 1–4. The global FFN publishes its
gated expansion as E4M3 (launch 58, `98.3%` exact), taking the residual from
`0.0108` to `0.0038` MAE. Given the vendor FFN residual as input, the 2-head
attention and residual reproduce block 5 at `90.4%` exact bytes, so the
multi-head attention tables, head grouping, scale, QKV order, and window
origins are confirmed; the remaining per-block error is one-ulp sensitivity.

Still open after these captures: the third noise channel's radius stream,
block 0's own bias layout (the single-head map halves its error; a learned
table floors at `0.0345` because the input is still inexact), the pooled
downsample transitions (`7%` of elements more than two ulps off with correct
weights), the global attention branch (`0.14` relative with exact FFN and V;
the Q/K/attended captures do not decode with the available maps), and every
8-head, split, and decoder block, for which no natural input/output pair was
captured.

### Block-0 schedule, 2026-09-02

With the corrected noise and the single-head bias layout in place, the
vendor-pooled feature selectors give an exact teacher-forced input for block 0.
A bias learned through the unchanged pool-then-block pipeline cannot be
explained by any of the `12!` index permutations of the decoded table (best
correlation `0.58` against a `0.98` calibration on block 1), and `99%` of its
softmax mass sits inside the query's own `4×4` pooled quadrant, i.e. inside one
native `8×8` window. The fused `pre_block_swin_1h_32_1_ds` kernel therefore runs
the block-0 window attention on the full-resolution adapter output, then
average-pools `2×2` and publishes the pooled result once. Under that order the
ordinary single-head fragment map is the unique best layout (rank 1 of `12!`,
no improving move), and block 0 reaches `0.0075` MAE, correlation `0.9995`,
and `62.8%` exact bytes on the hybrid input (`0.0254` under the earlier
schedule); chained blocks 0–4 on the native input improve from `0.777` to
`0.365` MAE at block 4. `NeuralRenderingEncoderStem` and
`NeuralRenderingModel.forward` now project, transform at full resolution, pool,
and round once. The full-resolution skip for block 70 is that block-0 output
published as E4M3, not the adapter projection (2026-09-02 correction: the
adapter projection carries the three noise channels straight into the post
merge and the head, which showed as 7–12× the DLL's pixel-band power on sharp
inputs; with the block-0 output the four 256 gates move from `0.0118` to
`0.0055` MAE in the reference chain and every spectral band sits within 8% of
the DLL). This supersedes the earlier "pool-then-block" structural reading.

End to end, the four corrections of this date move the `256×256` Cyberpunk
face from `0.0265` to `0.0180` MAE against the NVIDIA output and its effect
correlation from `0.37` to `0.73` (Kingdom Come `0.0410` → `0.0361`, Last of
Us `0.0200` → `0.0180`, Skyrim `0.0271` → `0.0280`, effect correlations
`0.5–0.6`); the untouched `3840×2160` frame reaches effect correlation `0.45`
at `0.0197` MAE with a least-squares effect scale of `0.38`, so the Mac effect
now points the vendor's way but remains weaker. Eager and fused execution agree
to `0.0027` display MAE while diverging by `19%` in head space after 38 blocks,
which quantifies the E4M3 chain amplification that also bounds end-to-end
parity.

## Remaining falsifiers

The following are not yet proved and must not be presented as compatible:

1. NVIDIA jitter parity, including whether each capture's MVec already includes
   render jitter and how it combines with transformed resources.
2. NVIDIA parity for disocclusions and temporal blending; the first measured
   two-frame baseline remains below the required threshold.
3. Observed NVIDIA MUFU distribution and full-output threshold; the conservative
   half-noise feature ceiling is established at `1/128`.
4. Exact v19 PyTorch↔MLX agreement and the remaining split-decoder/attention
   numerical gaps.
5. End-to-end parity against captured NVIDIA outputs, including style,
   tone/structure, auto-mask, ControlMask, the surrounding display codec,
   postprocessing, and temporal history.

The next correctness gate is to reduce the existing first-frame capture to a
declared numerical threshold, then repeat it for style, ControlMask, transformed
two-frame, disocclusion, and caller-jitter cases with the exact recorded motion
convention. Those cases must pass before the normalized temporal reference can
be called vendor-compatible.

## Publication boundary

This repository may publish only the generic runtime, independently reviewable
source, synthetic/open fixtures, and documentation. The NVIDIA binary, embedded
weights, extracted CUDA modules, and private oracle captures stay external.
Reverse-engineered compatibility code requires a separate legal review before a
public release; successful local execution is not permission to redistribute
vendor artifacts or claim endorsement.

### Global attention logit cap (hypothesis, opt-in), 2026-09-02

The global blocks 31–38 run as `cc_vit_1d_*` kernels (repack 2D→1D, gated
FFN expand/contract, QKV, attention, projection) over every token of the
/64 grid: 64 tokens at the 320×320 capture extent, about 2,160 at 4K. They
carry no `attn_bias`, and the port never used `blockN.layer3.attention_scalar`
(31–38: 0.109, −0.248, 0.00005, −0.010, 0.186, 0.041, 0.082, −558.5). The
vendor softmax approximation is a bit-affine exponential whose input domain is
about [−6, +6] with no row-max subtraction, while block-31 logits on the
teacher-forced pair (launches 57→62) span [−10, +17]. On that pair a hard cap
`min(scores, 3)` takes the attention branch from rel 0.139 to 0.059 and the
block output from rel 0.071 to 0.021 (58 % exact bytes); the optimum is 3.0
(2.5 and 3.5 give 0.072), soft caps are worse, a sink logit or a gate driven by
`attention_scalar` does nothing, and the Q/K scale is confirmed. The window
families are unaffected by a cap of 6 and harmed by 3, so the behaviour is
specific to the `vit_1d` kernels. End to end the cap is mixed: at 4K the port
starts producing the DLL's strong localized changes (p99 0.046 → 0.119 versus
the DLL's 0.117) but the mean error rises (0.0197 → 0.0235), and the 256-pixel
faces move both ways (Cyberpunk effect correlation 0.73 → 0.70, Skyrim
0.57 → 0.74). It is therefore exposed only as an opt-in parameter
(`cosine_attention(logit_cap=)` / `cosineAttention(logitCap:)`, constants
`EXPERIMENTAL_GLOBAL_ATTENTION_LOGIT_CAP` and
`experimentalGlobalAttentionLogitCap`), pending goldens for blocks 32–38 that
would settle a per-block value.
