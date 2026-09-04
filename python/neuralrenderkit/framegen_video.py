"""Frame generation for whole videos through FFmpeg.

Two modes: ``fps`` keeps the duration and multiplies the frame rate (generated
frames are inserted between the originals; audio is copied), ``slowmo`` keeps
the frame rate and stretches the duration by ``factor`` (audio is stretched
with FFmpeg's ``atempo`` — pitch-preserving WSOLA — copied as-is, or dropped).
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

from .framegen import FrameGenerator
from .video import DEFAULT_ENCODE_ARGS, VideoToolError, find_tool, probe

AUDIO_MODES = ("copy", "stretch", "none")


def atempo_chain(ratio: float) -> str:
    """FFmpeg ``atempo`` filter chain for a playback-speed ``ratio`` (0.5 = twice as long).

    A single ``atempo`` accepts 0.5..100, so slower ratios are split into 0.5 steps
    (``atempo=0.5,atempo=0.5`` for x4) with one final factor in range."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    steps: list[float] = []
    remaining = ratio
    while remaining < 0.5:
        steps.append(0.5)
        remaining /= 0.5
    while remaining > 100.0:
        steps.append(100.0)
        remaining /= 100.0
    steps.append(remaining)
    return ",".join(f"atempo={step:g}" for step in steps)


@dataclass
class FrameGenOptions:
    mode: str = "fps"                 # "fps" (rate x factor) or "slowmo" (duration x factor)
    factor: int = 2
    audio: str = "copy"               # copy | stretch (slowmo only) | none
    frame_limit: int | None = None
    encode_args: list[str] | None = None
    decode_args: list[str] = field(default_factory=list)
    overwrite: bool = False
    status_interval: float = 60.0
    backend: str = "torch"            # torch | nrk (Metal, macOS: frames stream through `nrk framegen-stream`)
    nrk: str | None = None
    nrk_weights: str | None = None    # dense safetensors for the Swift runtime (defaults to the torch weights path)
    nrk_precision: str = "float16"


@dataclass
class FrameGenResult:
    input_frames: int
    output_frames: int
    seconds: float
    width: int
    height: int
    input_fps: float
    output_fps: float
    output: Path


def interpolate_video(
    source: str | Path,
    destination: str | Path,
    generator: FrameGenerator | None,
    options: FrameGenOptions | None = None,
    *,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    log: Callable[[str], None] | None = None,
) -> FrameGenResult:
    """Decode ``source``, generate ``factor - 1`` frames between every consecutive pair, encode."""
    options = options or FrameGenOptions()
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    source = Path(source); destination = Path(destination)
    if options.mode not in ("fps", "slowmo"):
        raise VideoToolError("mode must be 'fps' or 'slowmo'")
    if options.factor < 2:
        raise VideoToolError("factor must be at least 2")
    if options.audio not in AUDIO_MODES:
        raise VideoToolError(f"audio must be one of {AUDIO_MODES}")
    if options.backend not in ("torch", "nrk"):
        raise VideoToolError("backend must be 'torch' or 'nrk'")
    if options.backend == "torch" and generator is None:
        raise VideoToolError("the torch backend needs a FrameGenerator")
    if options.backend == "nrk" and not options.nrk_weights:
        raise VideoToolError("the nrk backend needs nrk_weights (dense frame generation safetensors)")
    if options.audio == "stretch" and options.mode != "slowmo":
        raise VideoToolError("audio 'stretch' only applies to slowmo (fps mode keeps the duration; use 'copy')")
    if destination.exists() and not options.overwrite:
        raise VideoToolError(f"destination exists: {destination} (pass overwrite)")
    ffmpeg_tool = find_tool("ffmpeg", ffmpeg)
    info = probe(source, ffprobe=ffprobe)
    frame_bytes = info.width * info.height * 3
    in_rate = Fraction(info.frame_rate)
    out_rate = in_rate * options.factor if options.mode == "fps" else in_rate

    decode = [ffmpeg_tool, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source), *options.decode_args]
    if options.frame_limit is not None:
        decode += ["-frames:v", str(options.frame_limit)]
    decode += ["-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    encode = [ffmpeg_tool, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
              "-s", f"{info.width}x{info.height}", "-r", str(out_rate), "-i", "-"]
    if options.audio != "none" and info.has_audio:
        encode += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0"]
        if options.audio == "stretch":
            encode += ["-filter:a", atempo_chain(1.0 / options.factor), "-c:a", "aac", "-b:a", "192k"]
        else:
            encode += ["-c:a", "copy"]
        if options.mode == "slowmo" and options.audio == "copy":
            log("note: audio copied unchanged under a slowed video; use audio 'stretch' to keep them in sync")
        if options.frame_limit is not None:
            encode += ["-shortest"]
    else:
        encode += ["-map", "0:v:0"]
    encode += list(DEFAULT_ENCODE_ARGS if options.encode_args is None else options.encode_args)
    encode += [str(destination)]

    expected = info.frame_count
    if expected is not None and options.frame_limit is not None:
        expected = min(expected, options.frame_limit)
    stream = None
    if options.backend == "nrk":
        from .nrk_stream import NRKFrameGenStream

        stream = NRKFrameGenStream(options.nrk_weights, info.width, info.height, factor=options.factor, precision=options.nrk_precision, nrk=options.nrk)
    decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    started = time.perf_counter(); last_status = started
    frames_in = 0; frames_out = 0

    def write(frame: np.ndarray) -> None:
        nonlocal frames_out
        encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
        frames_out += 1

    try:
        previous: np.ndarray | None = None
        while True:
            chunk = decoder.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape(info.height, info.width, 3)
            frames_in += 1
            if stream is not None:
                for generated in stream.push(frame):
                    write(generated)
            elif previous is not None:
                for generated in generator.generate(previous, frame, options.factor):
                    write(generated)
            write(frame)
            previous = frame
            now = time.perf_counter()
            if now - last_status >= options.status_interval:
                elapsed = now - started; rate = frames_in / elapsed if elapsed else 0.0
                eta = f" eta {int((expected - frames_in) / rate)} s" if expected and rate else ""
                log(f"STATUS {time.strftime('%H:%M:%S')} frames {frames_in}/{expected if expected is not None else '?'} {rate:.2f} fps{eta}")
                last_status = now
        if stream is not None:
            stream.close()
        encoder.stdin.close()
        decoder_error = decoder.stderr.read().decode(errors="replace").strip()
        encoder_error = encoder.stderr.read().decode(errors="replace").strip()
        decoder.wait(); encoder.wait()
    finally:
        for process in (decoder, encoder):
            if process.poll() is None:
                process.kill()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
    if decoder.returncode:
        raise VideoToolError(f"ffmpeg decode failed: {decoder_error}")
    if encoder.returncode:
        raise VideoToolError(f"ffmpeg encode failed: {encoder_error}")
    seconds = time.perf_counter() - started
    log(f"DONE {frames_in} -> {frames_out} frames in {seconds:.1f} s ({frames_out / seconds if seconds else 0:.2f} fps out) -> {destination}")
    return FrameGenResult(frames_in, frames_out, seconds, info.width, info.height, float(in_rate), float(out_rate), destination)
