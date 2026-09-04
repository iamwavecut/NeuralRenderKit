import pathlib
import shutil
import subprocess
import tempfile
import unittest

from neuralrenderkit.framegen import FrameGenerator
from neuralrenderkit.framegen_video import FrameGenOptions, atempo_chain, interpolate_video
from neuralrenderkit.video import VideoToolError, probe

from .test_framegen import synthetic_framegen_weights

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class AtempoTests(unittest.TestCase):
    def test_single_step_in_range(self):
        self.assertEqual(atempo_chain(0.5), "atempo=0.5")
        self.assertEqual(atempo_chain(1.0), "atempo=1")
        self.assertEqual(atempo_chain(2.0), "atempo=2")

    def test_slow_ratios_split_into_half_steps(self):
        self.assertEqual(atempo_chain(0.25), "atempo=0.5,atempo=0.5")
        self.assertEqual(atempo_chain(1 / 3), "atempo=0.5,atempo=0.666667")

    def test_invalid_ratio(self):
        with self.assertRaises(ValueError):
            atempo_chain(0)


def _synthetic_video(path: pathlib.Path, frames: int = 6, fps: int = 10, audio: bool = True) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={fps}:duration={frames / fps}"]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={frames / fps}", "-c:a", "aac", "-shortest"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
class InterpolateVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = FrameGenerator(synthetic_framegen_weights(), device="cpu")

    def test_fps_mode_doubles_the_rate_and_keeps_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _synthetic_video(src, frames=6, fps=10)
            result = interpolate_video(src, dst, self.generator, FrameGenOptions(mode="fps", factor=2), log=lambda _m: None)
            self.assertEqual(result.input_frames, 6)
            self.assertEqual(result.output_frames, 11)
            info = probe(dst)
            self.assertAlmostEqual(info.fps, 20.0, places=3)
            self.assertTrue(info.has_audio)

    def test_slowmo_mode_keeps_the_rate_and_stretches_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _synthetic_video(src, frames=6, fps=10)
            result = interpolate_video(src, dst, self.generator, FrameGenOptions(mode="slowmo", factor=4, audio="stretch"), log=lambda _m: None)
            self.assertEqual(result.output_frames, 6 + 5 * 3)
            info = probe(dst)
            self.assertAlmostEqual(info.fps, 10.0, places=3)
            self.assertEqual(info.frame_count, 21)
            self.assertTrue(info.has_audio)

    def test_audio_none_drops_the_track(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _synthetic_video(src, frames=3, fps=10)
            interpolate_video(src, dst, self.generator, FrameGenOptions(audio="none"), log=lambda _m: None)
            self.assertFalse(probe(dst).has_audio)

    def test_option_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            src = pathlib.Path(directory) / "src.mp4"; dst = pathlib.Path(directory) / "dst.mp4"
            _synthetic_video(src, frames=2, fps=10, audio=False)
            for options in (FrameGenOptions(factor=1), FrameGenOptions(mode="fps", audio="stretch"), FrameGenOptions(mode="other")):
                with self.assertRaises(VideoToolError):
                    interpolate_video(src, dst, self.generator, options, log=lambda _m: None)
