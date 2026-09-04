# Super resolution: measured, not ported

DLSS Super Resolution (`nvngx_dlss.dll`, run through the vendor's library on
an RTX 5090) was evaluated on realistic content to decide whether a port is
worth the work. It is not, for the reasons below.

## What the network needs

The model expects engine-produced motion vectors and a per-frame jitter
sequence that matches them. Finished video and photographs have neither, and
the conditions cannot be reproduced convincingly from the outside.

## Measurements

Protocol: downsample the original, upscale with the library, compare with the
original crop (PSNR); Lanczos resampling as the baseline.

| test | result |
| --- | --- |
| 30 photographic single frames, 2× and 3× | loses to Lanczos in all 30, by 4.84 dB on average |
| 16-frame history with a TAA-style jitter sequence | 2.5–3 dB worse than single frames; accumulation exhausted by the third frame |
| real video, optical flow standing in for engine motion | trails Lanczos by 1.85 dB at the first frame and 4.87 dB by the twenty-fourth |
| control: deliberately wrong motion vectors | 42.7 % of output pixels change, so the vectors reach the network; the verdict is about the model, not the plumbing |

| scale, motion | Lanczos | DLSS SR | DLSS SR + jitter |
| --- | --- | --- | --- |
| 2×, zero motion | 40.29 | 34.29 | 34.59 |
| 2×, jittered | 40.29 | 31.80 | 31.87 |
| 3×, zero motion | 33.31 | 30.37 | 30.69 |
| 3×, jittered | 33.31 | 28.92 | 28.80 |

## Consequence

For enlarging realistic content an ordinary resampler is the better tool; the
detail pass of the neural-rendering network is what adds anything. Super
resolution is out of scope for this package.
