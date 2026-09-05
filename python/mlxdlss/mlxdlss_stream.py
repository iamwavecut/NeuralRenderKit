"""Metal backend for the converter: frames stream through ``mlxdlss stream`` over pipes.

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


def find_mlxdlss(explicit: str | None = None) -> str:
    import os

    candidates = [explicit, os.environ.get("MLXDLSS_BINARY"), shutil.which("mlxdlss")]
    root = Path(__file__).resolve().parents[2]
    candidates += [str(root / ".build" / "release" / "mlxdlss"), str(root / ".build" / "debug" / "mlxdlss")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("mlxdlss binary not found: build it with `swift build -c release --product mlxdlss`, pass --mlxdlss PATH or set MLXDLSS_BINARY")


class MLXDLSSStreamSession:
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
        mlxdlss: str | None = None,
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
            find_mlxdlss(mlxdlss), "stream", str(model_package), "--width", str(width), "--height", str(height),
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
                    raise RuntimeError(f"mlxdlss stream ended early: {self.process.stderr.read().decode(errors='replace')[-500:]}")
                data += chunk
        except BrokenPipeError as error:
            raise RuntimeError(f"mlxdlss stream failed: {self.process.stderr.read().decode(errors='replace')[-500:]}") from error
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
            raise RuntimeError(f"mlxdlss stream exited with {self.process.returncode}: {stderr[-500:]}")
        return summary


class MLXDLSSFrameGenStream:
    """Frame generation through ``mlxdlss framegen-stream`` (Metal): push frames in order and collect the
    ``factor - 1`` generated frames of every consecutive pair, in stream order, as uint8 arrays.

    The server computes ``batch`` pairs per pass, so a pair's frames come back once its window is
    complete (``push`` returns them) or when the input ends (``finish`` returns the rest)."""

    def __init__(self, weights: str | Path, width: int, height: int, *, factor: int = 2, precision: str = "float16",
                 mlxdlss: str | None = None, batch: int = 4):
        self.width, self.height, self.factor, self.batch = width, height, factor, max(1, int(batch))
        command = [find_mlxdlss(mlxdlss), "framegen-stream", "--weights", str(weights), "--width", str(width), "--height", str(height),
                   "--factor", str(factor), "--batch", str(self.batch), "--format", "u8", "--precision", precision]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0
        self.pending_pairs = 0

    def _read_frames(self, count: int) -> list[np.ndarray]:
        expected = self.height * self.width * 3
        outputs = []
        for _ in range(count):
            data = bytearray()
            while len(data) < expected:
                chunk = self.process.stdout.read(expected - len(data))
                if not chunk:
                    raise RuntimeError(f"mlxdlss framegen-stream ended early: {self.process.stderr.read().decode(errors='replace')[-500:]}")
                data += chunk
            outputs.append(np.frombuffer(bytes(data), dtype=np.uint8).reshape(self.height, self.width, 3).copy())
        return outputs

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        """Send one uint8 (H, W, 3) frame; returns the generated frames of every pair whose window
        just completed (``batch * (factor - 1)`` frames, or none)."""
        frame = np.asarray(frame)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame must be ({self.height}, {self.width}, 3)")
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        payload = np.ascontiguousarray(frame).tobytes()
        try:
            self.process.stdin.write(payload); self.process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(f"mlxdlss framegen-stream failed: {self.process.stderr.read().decode(errors='replace')[-500:]}") from error
        self.frames += 1
        if self.frames == 1:
            return []
        self.pending_pairs += 1
        if self.pending_pairs < self.batch:
            return []
        outputs = self._read_frames(self.pending_pairs * (self.factor - 1))
        self.pending_pairs = 0
        return outputs

    def finish(self) -> list[np.ndarray]:
        """End the input and return the generated frames of the pairs still pending."""
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            self.process.stdin = None
        outputs = self._read_frames(self.pending_pairs * (self.factor - 1)) if self.pending_pairs else []
        self.pending_pairs = 0
        return outputs

    def close(self) -> dict:
        """Finish the stream (discarding any pending output) and return the server's JSON summary."""
        try:
            self.finish()
        except RuntimeError:
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
