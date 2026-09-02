import pathlib
import shutil
import subprocess
import tempfile
import unittest

from neuralrenderkit import NeuralRenderingPipeline
from neuralrenderkit.video import ConvertOptions, VideoToolError, compare_command, convert, probe
from neuralrenderkit import video_cli

from .synthetic import synthetic_weights

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def make_clip(path: pathlib.Path, frames: int = 4, size: str = "64x48") -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000", "-frames:v", str(frames), "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True,
    )


@unittest.skipUnless(FFMPEG, "ffmpeg and ffprobe are required")
class VideoTests(unittest.TestCase):
    def test_probe_reports_size_rate_frames_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = pathlib.Path(directory) / "clip.mp4"
            make_clip(clip)
            info = probe(clip)
            self.assertEqual((info.width, info.height), (64, 48))
            self.assertAlmostEqual(info.fps, 10.0, places=3)
            self.assertEqual(info.frame_count, 4)
            self.assertTrue(info.has_audio)

    def test_convert_keeps_frame_count_size_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            clip = root / "clip.mp4"; out = root / "out.mp4"
            make_clip(clip, frames=3)
            pipeline = NeuralRenderingPipeline(synthetic_weights(), device="cpu")
            logs = []
            result = convert(clip, out, pipeline, ConvertOptions(batch=2, status_interval=0), log=logs.append)
            self.assertEqual(result.frames, 3)
            info = probe(out)
            self.assertEqual((info.width, info.height), (64, 48))
            self.assertEqual(info.frame_count, 3)
            self.assertTrue(info.has_audio)
            self.assertTrue(any(line.startswith("STATUS") for line in logs))
            with self.assertRaises(VideoToolError):
                convert(clip, out, pipeline, ConvertOptions())

    def test_frame_window_and_encode_args(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            clip = root / "clip.mp4"; out = root / "out.mkv"
            make_clip(clip, frames=5)
            pipeline = NeuralRenderingPipeline(synthetic_weights(), device="cpu")
            result = convert(clip, out, pipeline, ConvertOptions(start_frame=2, frame_limit=2, audio="none", encode_args=["-c:v", "libx264", "-qp", "0"], status_interval=1e9), log=lambda _: None)
            self.assertEqual(result.frames, 2)
            info = probe(out)
            self.assertEqual(info.frame_count, 2)
            self.assertFalse(info.has_audio)


class CompareCommandTests(unittest.TestCase):
    def test_compare_command_builds_side_by_side_mpv_invocation(self):
        try:
            command = compare_command("a.mp4", "b.mp4", player=shutil.which("mpv") or "mpv")
        except VideoToolError:
            self.skipTest("mpv not installed")
        self.assertIn("--lavfi-complex=[vid1][vid2]hstack[vo]", command)
        self.assertIn("--external-file=b.mp4", command)

    def test_cli_parser_exposes_ffmpeg_passthrough(self):
        parser = video_cli.build_parser()
        args = parser.parse_args(["convert", "in.mp4", "out.mp4", "--weights", "w.safetensors", "--encode-args", "-c:v libx265 -crf 20", "--decode-args", "-vf scale=640:-2"])
        self.assertEqual(args.encode_args, "-c:v libx265 -crf 20")
        self.assertEqual(args.decode_args, "-vf scale=640:-2")
