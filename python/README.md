# neuralrenderkit (Python)

PyTorch inference pipeline and bring-your-own-DLL weight tooling for the
recovered neural-rendering transformer of [NeuralRenderKit](../README.md).

```sh
python -m pip install .            # numpy, torch, safetensors, pillow
python -m pip install '.[coreml]'  # optional Core ML conversion (macOS, Linux)

nrk-weights all nvngx_dlssnr.dll weights/ --coreml 320x320   # your own DLL -> packed, logical, MLX, Core ML
nrk-torch run --weights weights/dlssnr-weights-logical.safetensors --input in.png --output out.png
```

```python
from neuralrenderkit import NeuralRenderingPipeline
pipeline = NeuralRenderingPipeline.from_safetensors("weights/dlssnr-weights-logical.safetensors", device="auto")
result = pipeline.enhance(image_float32_hwc, profile="standard", processing_scale=2, detail_strength=2)
```

`precision="reference"` (default) computes in float32 with the recovered
E4M3/half rounding points; `"fast"` runs the same graph in float16 on CUDA or
MPS. No weights are included or downloaded: the DLL comes from your own NVIDIA
driver or Streamline package and every artifact stays on your machine.
