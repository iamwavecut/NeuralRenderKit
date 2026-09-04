"""DLSS Frame Generation for plain video (PyTorch port).

The library's frame generator takes two consecutive frames plus game motion
vectors and depth.  Without motion vectors (video), every MV-derived stage
collapses to constants and the whole generator reduces to:

    A, B            full-res RGB in [0, 1]
    a, b = box2(A), box2(B)                       # 2x2 mean, padded to x16
    err  = blur3(mean_rgb |a - b|)                # [1/8, 3/4, 1/8] separable
    x    = [a, err, b, err, 0, t]                 # 10 channels (k_initial_merge)
    block0: stem (3 conv+pool) -> 8 residual convs -> 3 heads -> 8 linear outputs
    f0, m0, r0 = block0(x)  upsampled x2 (bilinear)
    block1 on [warp(a, 2 f0A), warp(b, 2 f0B), f0, m0, 0, r0, t]   (18 channels)
    flow = 2 f0 + f1, mask = m0 + m1               # half-res export
    out  = sigmoid(mask)^ * warp(A, 2 flowA^) + (1 - sigmoid(mask)^) * warp(B, 2 flowB^)

where ^ is bilinear upsampling to full resolution.  Every stage was matched
against memory snapshots of the shipping library (see the research notes).

Weights are the 42 fp16 blobs of ``libnvidia-ngx-dlssg.so`` re-laid out as
dense ``[Cout, Cin, kh, kw]`` tensors (``nrk-weights extract-fg``); they are
never distributed with this package.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .pipeline import resolve_device

_LEAK = 0.01


def _act(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.leaky_relu(x, _LEAK), -6.0, 6.0)


def _conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, w, b, padding=w.shape[-1] // 2)


def _pad_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.shape[1] == channels:
        return x
    return F.pad(x, (0, 0, 0, 0, 0, channels - x.shape[1]))


def _up2(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


def _warp(img: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """Backward bilinear warp of img [N,C,H,W] by absolute pixel offsets fx, fy [N,H,W] (border clamped)."""
    n, _, h, w = img.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=img.device, dtype=img.dtype),
        torch.arange(w, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    gx = (xs[None] + fx) * (2.0 / max(w - 1, 1)) - 1.0
    gy = (ys[None] + fy) * (2.0 / max(h - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def box2(x: torch.Tensor) -> torch.Tensor:
    """2x2 box mean, padded with zeros to a multiple of 16 rows and columns (the library's tile size)."""
    _, _, h, w = x.shape
    y = F.avg_pool2d(x[:, :, : h // 2 * 2, : w // 2 * 2], 2)
    hp = (y.shape[2] + 15) // 16 * 16
    wp = (y.shape[3] + 15) // 16 * 16
    return F.pad(y, (0, wp - y.shape[3], 0, hp - y.shape[2]))


def photometric_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """mean_rgb |a - b| blurred with the separable [1/8, 3/4, 1/8] kernel (border replicated)."""
    e = (a - b).abs().mean(1, keepdim=True)
    k = torch.tensor([0.125, 0.75, 0.125], device=e.device, dtype=e.dtype)
    k2 = (k[:, None] * k[None, :])[None, None]
    return F.conv2d(F.pad(e, (1, 1, 1, 1), mode="replicate"), k2)


@dataclass
class _Block:
    """One synthesis block: stem convs (act + 2x2 avg pool each), residual pairs, three
    activated heads on the x2-upsampled trunk, one linear 8-channel head on the x2-upsampled heads."""

    stem: list[tuple[torch.Tensor, torch.Tensor]]
    res: list[tuple[torch.Tensor, torch.Tensor]]
    heads: list[tuple[torch.Tensor, torch.Tensor]]
    out_w: torch.Tensor
    out_b: torch.Tensor

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for w, b in self.stem:
            x = F.avg_pool2d(_act(_conv(_pad_channels(x, w.shape[1]), w, b)), 2)
        skip = x
        for i, (w, b) in enumerate(self.res):
            y = _act(_conv(x, w, b))
            if i % 2 == 0:
                skip, x = x, y
            else:
                x = y + skip
        heads = [_act(_conv(_up2(x), w, b)) for w, b in self.heads]
        spans = [(0, 4), (4, 5), (5, 8)]
        outs = [_conv(_up2(heads[i]), self.out_w[lo:hi], self.out_b[lo:hi]) for i, (lo, hi) in enumerate(spans)]
        return outs[0], outs[1], outs[2]  # flow (4), mask (1), residual (3)


FRAMEGEN_TENSORS = (
    [f"block0.stem{i}.{p}" for i in range(3) for p in ("weight", "bias")]
    + [f"block0.res{i}.{p}" for i in range(8) for p in ("weight", "bias")]
    + [f"block0.bot0.head{i}.{p}" for i in range(3) for p in ("weight", "bias")]
    + ["block0.bot1.weight", "block0.bot1.bias"]
    + [f"block1.stem{i}.{p}" for i in range(2) for p in ("weight", "bias")]
    + [f"block1.res{i}.{p}" for i in range(8) for p in ("weight", "bias")]
    + [f"block1.bot0.head{i}.{p}" for i in range(3) for p in ("weight", "bias")]
    + ["block1.bot1.weight", "block1.bot1.bias"]
)


class FrameGenerator:
    """Interpolates frames between two consecutive video frames.

    ``precision`` is ``"reference"`` (float32) or ``"fast"`` (float16 on GPU devices).
    """

    def __init__(self, weights: dict[str, torch.Tensor], *, device: str | torch.device = "auto", precision: str = "reference"):
        missing = [k for k in FRAMEGEN_TENSORS if k not in weights]
        if missing:
            raise ValueError(f"frame generation weights are missing {len(missing)} tensors, e.g. {missing[:3]}")
        self.device = resolve_device(device)
        if precision not in ("reference", "fast"):
            raise ValueError("precision must be 'reference' or 'fast'")
        self.dtype = torch.float16 if (precision == "fast" and self.device.type != "cpu") else torch.float32
        get = lambda k: weights[k].to(self.device, self.dtype)  # noqa: E731
        pair = lambda name: (get(f"{name}.weight"), get(f"{name}.bias"))  # noqa: E731

        def block(prefix: str, stems: int) -> _Block:
            return _Block(
                stem=[pair(f"{prefix}.stem{i}") for i in range(stems)],
                res=[pair(f"{prefix}.res{i}") for i in range(8)],
                heads=[pair(f"{prefix}.bot0.head{i}") for i in range(3)],
                out_w=get(f"{prefix}.bot1.weight"),
                out_b=get(f"{prefix}.bot1.bias"),
            )

        self.block0 = block("block0", 3)
        self.block1 = block("block1", 2)

    @classmethod
    def from_safetensors(cls, path: str | pathlib.Path, **kwargs: Any) -> "FrameGenerator":
        from safetensors.torch import load_file

        return cls(load_file(str(path)), **kwargs)

    # -- stages ---------------------------------------------------------------
    @torch.no_grad()
    def synthesize(self, a_full: torch.Tensor, b_full: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Half-resolution export [N, 8, h, w]: flowA.xy, flowB.xy, mask logit, residual.rgb.

        ``t`` is one phase for the whole batch or a tensor of N phases."""
        a = box2(a_full)
        b = box2(b_full)
        err = photometric_error(a, b)
        zero = torch.zeros_like(err)
        phases = torch.as_tensor(t, device=a.device, dtype=a.dtype).reshape(-1, 1, 1, 1)
        phase = torch.zeros_like(err) + phases
        cand_a = torch.cat([a, err], 1)
        cand_b = torch.cat([b, err], 1)
        f0, m0, r0 = self.block0(torch.cat([cand_a, cand_b, zero, phase], 1))
        f0, m0, r0 = _up2(f0), _up2(m0), _up2(r0)
        warped_a = _warp(cand_a, 2 * f0[:, 0], 2 * f0[:, 1])
        warped_b = _warp(cand_b, 2 * f0[:, 2], 2 * f0[:, 3])
        f1, m1, r1 = self.block1(torch.cat([warped_a, warped_b, f0, m0, zero, r0, phase], 1))
        return torch.cat([2 * f0 + f1, m0 + m1, r0 + r1], 1)

    @torch.no_grad()
    def compose(self, a_full: torch.Tensor, b_full: torch.Tensor, export: torch.Tensor) -> torch.Tensor:
        """Full-resolution frames [N,3,H,W] from the half-res export (main_kernel 083)."""
        n, _, h, w = a_full.shape
        _, _, hc, wc = export.shape
        dev, dt = a_full.device, a_full.dtype
        ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=dt), torch.arange(w, device=dev, dtype=dt), indexing="ij")
        coarse = torch.cat([export[:, 0:4], torch.sigmoid(export[:, 4:5])], 1)
        # coarse sample position u = min(x/2, wc-1): plain half-scale, index-clamped
        u = (xs / 2).clamp(max=wc - 1.0)
        v = (ys / 2).clamp(max=hc - 1.0)
        gx = u * (2.0 / max(wc - 1, 1)) - 1.0
        gy = v * (2.0 / max(hc - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], -1)[None].expand(n, -1, -1, -1)
        up = F.grid_sample(coarse, grid, mode="bilinear", padding_mode="border", align_corners=True)
        m = up[:, 4:5]
        wa = _warp(a_full, 2 * up[:, 0], 2 * up[:, 1])
        wb = _warp(b_full, 2 * up[:, 2], 2 * up[:, 3])
        return (m * wa + (1 - m) * wb).clamp(0, 1)

    # -- public API -----------------------------------------------------------
    def _to_tensor(self, frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(frame, torch.Tensor):
            x = frame
            if x.ndim == 3 and x.shape[-1] == 3:
                x = x.permute(2, 0, 1)
            if x.ndim == 3:
                x = x[None]
        else:
            arr = np.asarray(frame)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None]
        return x.to(self.device, self.dtype)

    @torch.no_grad()
    def interpolate(self, a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor, t: float = 0.5) -> torch.Tensor:
        """Frame at phase ``t`` (0 = a, 1 = b) as [1, 3, H, W] in [0, 1] on the model device."""
        a_t = self._to_tensor(a)
        b_t = self._to_tensor(b)
        if a_t.shape != b_t.shape:
            raise ValueError(f"frames differ in shape: {tuple(a_t.shape)} vs {tuple(b_t.shape)}")
        return self.compose(a_t, b_t, self.synthesize(a_t, b_t, t))

    @torch.no_grad()
    def generate(self, a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor, factor: int = 2) -> list[np.ndarray]:
        """``factor - 1`` intermediate frames between a and b as uint8 HxWx3 arrays (phases k/factor),
        computed as one batch."""
        if factor < 2:
            raise ValueError("factor must be >= 2")
        a_t = self._to_tensor(a)
        b_t = self._to_tensor(b)
        n = factor - 1
        phases = torch.tensor([k / factor for k in range(1, factor)], device=a_t.device, dtype=a_t.dtype)
        a_n, b_n = a_t.expand(n, -1, -1, -1), b_t.expand(n, -1, -1, -1)
        out = self.compose(a_n, b_n, self.synthesize(a_n, b_n, phases))
        frames = (out.permute(0, 2, 3, 1).float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        return [frames[i] for i in range(n)]

    @torch.no_grad()
    def generate_pairs(self, frames: list[np.ndarray], factor: int = 2) -> list[list[np.ndarray]]:
        """For consecutive frames f0..fk, the generated frames of every pair (f_i, f_{i+1}) in one batch
        of (k * (factor - 1)) samples; returns one list per pair."""
        if factor < 2:
            raise ValueError("factor must be >= 2")
        if len(frames) < 2:
            return []
        pairs = len(frames) - 1
        n = factor - 1
        a_t = torch.cat([self._to_tensor(f) for f in frames[:-1]], 0)
        b_t = torch.cat([self._to_tensor(f) for f in frames[1:]], 0)
        a_n = a_t.repeat_interleave(n, 0)
        b_n = b_t.repeat_interleave(n, 0)
        phases = torch.tensor([k / factor for k in range(1, factor)] * pairs, device=a_t.device, dtype=a_t.dtype)
        out = self.compose(a_n, b_n, self.synthesize(a_n, b_n, phases))
        arr = (out.permute(0, 2, 3, 1).float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        return [[arr[p * n + i] for i in range(n)] for p in range(pairs)]
