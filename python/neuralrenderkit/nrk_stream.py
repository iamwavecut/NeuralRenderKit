"""Metal backend for the converter: frames stream through ``nrk stream`` over pipes.

Protocol (per frame): little-endian uint32 flags (bit 0 = reset history), then
float32 colour (H, W, 3); in temporal mode also motion (H, W, 2) as normalised
history-UV offsets and depth (H, W, 1). One float32 RGB frame comes back per
input frame.
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from .composition import compose_detail
from .temporal import FlowMotionEstimator, zero_motion

RESET_FLAG = 1


def find_nrk(explicit: str | None = None) -> str:
    import os

    candidates = [explicit, os.environ.get("NRK_BINARY"), shutil.which("nrk")]
    root = Path(__file__).resolve().parents[2]
    candidates += [str(root / ".build" / "release" / "nrk"), str(root / ".build" / "debug" / "nrk")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("nrk binary not found: build it with `swift build -c release --product nrk`, pass --nrk PATH or set NRK_BINARY")


class NRKStreamSession:
    """Frame-sequence processor backed by the Swift Metal runtime (macOS, Apple Silicon)."""

    def __init__(
        self,
        model_package: str | Path,
        width: int,
        height: int,
        *,
        temporal: bool = True,
        motion: Callable | str = "flow",
        scene_cut_threshold: float = 0.3,
        nrk: str | None = None,
        profile: str = "standard",
        intensity: float = 1.0,
        execution: str = "metal-fused",
        precision: str = "float16",
        processing_scale: float = 1.0,
        detail_strength: float = 1.0,
        colour_strength: float = 1.0,
        detail_radius: float = 4.0,
    ):
        self.width, self.height, self.temporal = width, height, temporal
        self.scene_cut_threshold = scene_cut_threshold
        self.detail = (detail_strength, colour_strength, detail_radius)
        if not temporal:
            self.motion = zero_motion
        elif motion == "flow":
            self.motion = FlowMotionEstimator()
        elif motion == "zero":
            self.motion = zero_motion
        elif callable(motion):
            self.motion = motion
        else:
            raise ValueError("motion must be 'flow', 'zero' or a callable")
        command = [
            find_nrk(nrk), "stream", str(model_package), "--width", str(width), "--height", str(height),
            "--mode", "temporal" if temporal else "first-frame", "--execution", execution, "--precision", precision,
            "--profile", profile, "--intensity", str(intensity),
        ]
        if not temporal:
            # the recipe is applied by the Swift side in first-frame mode (resampling included)
            command += ["--processing-scale", str(processing_scale), "--detail-strength", str(detail_strength),
                        "--colour-strength", str(colour_strength), "--detail-radius", str(detail_radius)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.previous: np.ndarray | None = None
        self.frame_index = 0
        self.scene_cuts = 0
        self.frames = 0
        self._depth = np.ones((height, width, 1), dtype="<f4").tobytes()

    def _is_scene_cut(self, frame: np.ndarray) -> bool:
        if self.previous is None or self.scene_cut_threshold <= 0:
            return False
        luma = lambda f: f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722
        return float(np.abs(luma(frame) - luma(self.previous)).mean()) > self.scene_cut_threshold

    def process_frame(self, frame: np.ndarray, *, motion: np.ndarray | None = None) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame must be ({self.height}, {self.width}, 3)")
        flags = 0
        if self._is_scene_cut(frame):
            flags |= RESET_FLAG; self.scene_cuts += 1; self.frame_index = 0
        payload = [struct.pack("<I", flags), np.ascontiguousarray(frame.astype("<f4")).tobytes()]
        if self.temporal:
            if motion is None:
                motion = zero_motion(frame, frame) if (self.previous is None or flags & RESET_FLAG) else self.motion(frame, self.previous)
            payload += [np.ascontiguousarray(np.asarray(motion, dtype="<f4")).tobytes(), self._depth]
        try:
            self.process.stdin.write(b"".join(payload)); self.process.stdin.flush()
            expected = self.height * self.width * 3 * 4
            data = bytearray()
            while len(data) < expected:
                chunk = self.process.stdout.read(expected - len(data))
                if not chunk:
                    raise RuntimeError(f"nrk stream ended early: {self.process.stderr.read().decode(errors='replace')[-500:]}")
                data += chunk
        except BrokenPipeError as error:
            raise RuntimeError(f"nrk stream failed: {self.process.stderr.read().decode(errors='replace')[-500:]}") from error
        output = np.frombuffer(bytes(data), dtype="<f4").reshape(self.height, self.width, 3)
        self.previous = frame; self.frame_index += 1; self.frames += 1
        if self.temporal:
            detail, colour, radius = self.detail
            output = compose_detail(frame, output, detail_strength=detail, colour_strength=colour, radius=radius)
        return output

    def close(self) -> dict:
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        stderr = self.process.stderr.read().decode(errors="replace")
        self.process.wait()
        summary = {}
        for line in stderr.strip().split("\n"):
            if line.startswith("{"):
                try:
                    summary = json.loads(line)
                except json.JSONDecodeError:
                    pass
        if self.process.returncode:
            raise RuntimeError(f"nrk stream exited with {self.process.returncode}: {stderr[-500:]}")
        return summary


class NRKFrameGenStream:
    """Frame generation through ``nrk framegen-stream`` (Metal): push frames in order, get the
    ``factor - 1`` generated frames between the previous frame and the new one back as uint8."""

    def __init__(self, weights: str | Path, width: int, height: int, *, factor: int = 2, precision: str = "float16", nrk: str | None = None):
        self.width, self.height, self.factor = width, height, factor
        command = [find_nrk(nrk), "framegen-stream", "--weights", str(weights), "--width", str(width), "--height", str(height),
                   "--factor", str(factor), "--precision", precision]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        """Send one uint8 (H, W, 3) frame; returns the generated frames after the first push."""
        frame = np.asarray(frame)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame must be ({self.height}, {self.width}, 3)")
        payload = np.ascontiguousarray(frame.astype(np.float32) / 255.0 if frame.dtype == np.uint8 else frame.astype("<f4")).tobytes()
        try:
            self.process.stdin.write(payload); self.process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(f"nrk framegen-stream failed: {self.process.stderr.read().decode(errors='replace')[-500:]}") from error
        self.frames += 1
        if self.frames == 1:
            return []
        expected = self.height * self.width * 3 * 4
        outputs = []
        for _ in range(self.factor - 1):
            data = bytearray()
            while len(data) < expected:
                chunk = self.process.stdout.read(expected - len(data))
                if not chunk:
                    raise RuntimeError(f"nrk framegen-stream ended early: {self.process.stderr.read().decode(errors='replace')[-500:]}")
                data += chunk
            values = np.frombuffer(bytes(data), dtype="<f4").reshape(self.height, self.width, 3)
            outputs.append((np.clip(values, 0, 1) * 255.0 + 0.5).astype(np.uint8))
        return outputs

    def close(self) -> dict:
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        stderr = self.process.stderr.read().decode(errors="replace")
        self.process.wait()
        for pipe in (self.process.stdout, self.process.stderr):
            if pipe is not None:
                pipe.close()
        for line in stderr.strip().split("\n"):
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        return {}
