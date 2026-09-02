"""Synthetic logical weights with the recovered graph's names and shapes (from weight_spec.json)."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

SPEC = json.loads((pathlib.Path(__file__).resolve().parents[1] / "neuralrenderkit" / "weight_spec.json").read_text())["tensors"]


def synthetic_weights(seed: int = 0, scale: float = 0.05) -> dict[str, torch.Tensor]:
    generator = np.random.default_rng(seed)
    weights = {}
    for name, entry in SPEC.items():
        shape = entry["shape"]
        if name.endswith(("attn_scale", "ffn_cos_skip", "attn_cos_skip", "inp_merge_sin", "inp_merge_cos", "inp_upsample_sin", "sin", "blend_scale")):
            values = generator.uniform(0.5, 1.0, size=shape)
        elif name.endswith("attn_bias"):
            values = generator.normal(0, 0.1, size=shape)
        else:
            values = generator.normal(0, scale, size=shape)
        dtype = torch.float16 if entry["dtype"] == "F16" else torch.float32
        weights[name] = torch.from_numpy(values.astype(np.float32)).to(dtype)
    return weights


def write_logical_safetensors(path: pathlib.Path, weights: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file(weights, str(path), metadata={"format": "dlssnr-logical-v18", "fully_logical": "true"})
