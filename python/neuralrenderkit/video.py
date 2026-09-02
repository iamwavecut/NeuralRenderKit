"""Video conversion through FFmpeg: decode frames, enhance them, encode the result.

FFmpeg decodes the source to raw RGB on a pipe, frames go through the pipeline
one by one (or in batches on GPU devices), and a second FFmpeg process encodes
them with user-controlled codec arguments while copying the source audio.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .pipeline import NeuralRenderingPipeline
from .temporal import BLEND_SCALE, TemporalOptions, TemporalSession

DEFAULT_ENCODE_ARGS = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
PIXEL_FORMATS = {"rgb24": (np.uint8, 255.0, 3), "rgb48le": ("<u2", 65535.0, 6)}


class VideoToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_rate: Fraction
    frame_count: int | None
    duration: float | None
    has_audio: bool
    pixel_format: str | None
    codec: str | None

    @property
    def fps(self) -> float:
        return float(self.frame_rate)


def find_tool(name: str, explicit: str | None = None) -> str:
    path = explicit or shutil.which(name)
    if not path or not Path(path).exists() and not shutil.which(path):
        raise VideoToolError(f"{name} not found; install FFmpeg or pass --{name}")
    return path


def probe(path: str | Path, *, ffprobe: str | None = None) -> VideoInfo:
    tool = find_tool("ffprobe", ffprobe)
    command = [tool, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", "-count_packets", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise VideoToolError(f"ffprobe failed: {completed.stderr.strip()}")
    data = json.loads(completed.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoToolError(f"no video stream in {path}")
    rate = Fraction(video.get("avg_frame_rate") or video.get("r_frame_rate") or "25/1")
    if rate == 0:
        rate = Fraction(video.get("r_frame_rate") or "25/1")
    count = None
    for key in ("nb_frames", "nb_read_packets"):
        if video.get(key) not in (None, "N/A"):
            count = int(video[key]); break
    duration = None
    for source in (video, data.get("format", {})):
        if source.get("duration") not in (None, "N/A"):
            duration = float(source["duration"]); break
    if count is None and duration is not None:
        count = int(round(duration * float(rate)))
    return VideoInfo(
        width=int(video["width"]), height=int(video["height"]), frame_rate=rate, frame_count=count, duration=duration,
        has_audio=any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
        pixel_format=video.get("pix_fmt"), codec=video.get("codec_name"),
    )


@dataclass
class ConvertOptions:
    start_frame: int = 0
    frame_limit: int | None = None
    batch: int = 1
    pixel_format: str = "rgb24"
    decode_args: list[str] = field(default_factory=list)
    encode_args: list[str] | None = None
    audio: str = "copy"
    overwrite: bool = False
    status_interval: float = 60.0
    enhance: dict[str, Any] = field(default_factory=dict)
    temporal: bool = False
    motion: str = "flow"           # temporal mode: 'flow' (optical flow) or 'zero'
    scene_cut_threshold: float = 0.3
    blend_scale: float = BLEND_SCALE
    backend: str = "torch"         # 'torch' (this pipeline) or 'nrk' (Swift Metal runtime via `nrk stream`, macOS)
    model_package: str | None = None
    nrk: str | None = None
    execution: str = "metal-fused"
    precision: str = "float16"


@dataclass
class ConvertResult:
    frames: int
    seconds: float
    width: int
    height: int
    fps: float
    output: Path
    scene_cuts: int = 0


def convert(
    source: str | Path,
    destination: str | Path,
    pipeline: NeuralRenderingPipeline,
    options: ConvertOptions | None = None,
    *,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    log: Callable[[str], None] | None = None,
) -> ConvertResult:
    """Decode ``source`` with FFmpeg, enhance every frame, encode to ``destination``."""
    options = options or ConvertOptions()
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    source = Path(source); destination = Path(destination)
    if destination.exists() and not options.overwrite:
        raise VideoToolError(f"destination exists: {destination} (pass overwrite)")
    if options.pixel_format not in PIXEL_FORMATS:
        raise VideoToolError(f"pixel format must be one of {tuple(PIXEL_FORMATS)}")
    if options.batch < 1:
        raise VideoToolError("batch must be at least 1")
    ffmpeg_tool = find_tool("ffmpeg", ffmpeg)
    info = probe(source, ffprobe=ffprobe)
    dtype, scale, bytes_per_pixel = PIXEL_FORMATS[options.pixel_format]
    frame_bytes = info.width * info.height * bytes_per_pixel
    decode = [ffmpeg_tool, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source), *options.decode_args]
    if options.start_frame:
        decode += ["-vf", f"select=gte(n\\,{options.start_frame})", "-fps_mode", "passthrough"]
    if options.frame_limit is not None:
        decode += ["-frames:v", str(options.frame_limit)]
    decode += ["-an", "-f", "rawvideo", "-pix_fmt", options.pixel_format, "-"]
    encode = [ffmpeg_tool, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", options.pixel_format,
              "-s", f"{info.width}x{info.height}", "-r", str(info.frame_rate), "-i", "-"]
    if options.audio == "copy" and info.has_audio:
        encode += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
        if options.start_frame or options.frame_limit is not None:
            encode += ["-shortest"]
    else:
        encode += ["-map", "0:v:0"]
    encode += list(DEFAULT_ENCODE_ARGS if options.encode_args is None else options.encode_args)
    encode += [str(destination)]
    enhance = dict(options.enhance)
    finish_keys = {"detail_strength", "colour_strength", "detail_radius", "intensity"}
    finish_options = {k: v for k, v in enhance.items() if k in finish_keys}
    prepare_options = {k: v for k, v in enhance.items() if k not in finish_keys}
    expected = info.frame_count
    if expected is not None:
        expected = max(0, expected - options.start_frame)
        if options.frame_limit is not None:
            expected = min(expected, options.frame_limit)
    session = None
    stream = None
    if options.backend == "nrk":
        from .nrk_stream import NRKStreamSession

        if not options.model_package:
            raise VideoToolError("backend 'nrk' needs --model MODEL.nrkmodel")
        stream = NRKStreamSession(
            options.model_package, info.width, info.height, temporal=options.temporal, motion=options.motion,
            scene_cut_threshold=options.scene_cut_threshold, nrk=options.nrk, profile=enhance.get("profile", "standard"),
            intensity=enhance.get("intensity", 1.0), execution=options.execution, precision=options.precision,
            processing_scale=enhance.get("processing_scale", 1.0), detail_strength=enhance.get("detail_strength", 1.0),
            colour_strength=enhance.get("colour_strength", 1.0), detail_radius=enhance.get("detail_radius", 4.0),
        )
    elif options.temporal:
        session = TemporalSession(
            pipeline,
            options=TemporalOptions(
                profile=enhance.get("profile", "standard"), blend_scale=options.blend_scale, intensity=enhance.get("intensity", 1.0),
                scene_cut_threshold=options.scene_cut_threshold, detail_strength=enhance.get("detail_strength", 1.0),
                colour_strength=enhance.get("colour_strength", 1.0), detail_radius=enhance.get("detail_radius", 4.0),
            ),
            motion=options.motion,
        )
        if enhance.get("processing_scale", 1.0) != 1.0:
            raise VideoToolError("temporal mode runs at the native scale (processing_scale must be 1)")
    decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    started = time.perf_counter(); last_status = started; frames = 0
    def emit_temporal(frame):
        nonlocal frames
        image = np.clip(stream.process_frame(frame) if stream is not None else session.process(frame), 0, 1) * scale + 0.5
        encoder.stdin.write(np.ascontiguousarray(image.astype(dtype)).tobytes())
        frames += 1

    def emit(prepared_batch):
        nonlocal frames
        network_started = time.perf_counter()
        heads = pipeline.run_features_batch(np.stack([p.features for p in prepared_batch]))
        network_seconds = (time.perf_counter() - network_started) / len(prepared_batch)
        for prepared, head in zip(prepared_batch, heads):
            result = pipeline.finish(prepared, head, network_seconds=network_seconds, **finish_options)
            image = np.clip(result.image, 0, 1) * scale + 0.5
            encoder.stdin.write(np.ascontiguousarray(image.astype(dtype)).tobytes())
            frames += 1
    try:
        pending = []
        while True:
            chunk = decoder.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            frame = np.frombuffer(chunk, dtype=dtype).reshape(info.height, info.width, 3).astype(np.float32) / np.float32(scale)
            if session is not None or stream is not None:
                emit_temporal(frame)
            else:
                pending.append(pipeline.prepare(frame, frame_index=options.start_frame + frames + len(pending), **prepare_options))
                if len(pending) >= options.batch:
                    emit(pending); pending = []
            now = time.perf_counter()
            if now - last_status >= options.status_interval:
                elapsed = now - started; rate = frames / elapsed if elapsed else 0.0
                eta = f" eta {int((expected - frames) / rate)} s" if expected and rate else ""
                log(f"STATUS {time.strftime('%H:%M:%S')} frames {frames}/{expected if expected is not None else '?'} {rate:.2f} fps{eta}")
                last_status = now
        if pending:
            emit(pending)
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
    if decoder.returncode:
        raise VideoToolError(f"ffmpeg decode failed: {decoder_error}")
    if encoder.returncode:
        raise VideoToolError(f"ffmpeg encode failed: {encoder_error}")
    seconds = time.perf_counter() - started
    cuts = session.scene_cuts if session is not None else (stream.scene_cuts if stream is not None else 0)
    log(f"DONE frames {frames} in {seconds:.1f} s ({frames / seconds if seconds else 0:.2f} fps){f', scene cuts {cuts}' if options.temporal else ''} -> {destination}")
    return ConvertResult(frames, seconds, info.width, info.height, info.fps, destination, cuts)


def compare_command(original: str | Path, processed: str | Path, *, player: str | None = None) -> list[str]:
    """mpv command showing the original and the processed video side by side."""
    tool = find_tool("mpv", player)
    return [tool, str(original), f"--external-file={processed}", "--lavfi-complex=[vid1][vid2]hstack[vo]", "--keep-open=yes"]
