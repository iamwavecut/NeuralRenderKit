import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from .synthetic import synthetic_weights, write_logical_safetensors

try:
    from neuralrenderkit.nrk_stream import NRKStreamSession, find_nrk
    NRK = find_nrk()
except Exception:  # binary absent (Linux, Windows, or not built)
    NRK = None


@unittest.skipUnless(NRK, "nrk binary is required (macOS build)")
class NRKStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from neuralrenderkit.tools import cli as weights_cli

        cls.directory = tempfile.mkdtemp()
        weights = pathlib.Path(cls.directory) / "weights.safetensors"
        write_logical_safetensors(weights, synthetic_weights())
        cls.package = pathlib.Path(cls.directory) / "Synthetic.nrkmodel"
        assert weights_cli.main(["mlx", str(weights), str(cls.package)]) == 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_temporal_stream_returns_one_frame_per_input_and_resets_on_cut(self):
        session = NRKStreamSession(self.package, 64, 48, temporal=True, motion="zero", scene_cut_threshold=0.3, nrk=NRK)
        frame = np.random.default_rng(0).random((48, 64, 3)).astype(np.float32) * 0.2
        first = session.process_frame(frame)
        second = session.process_frame(frame)
        third = session.process_frame(np.clip(frame + 0.7, 0, 1))
        summary = session.close()
        self.assertEqual(first.shape, (48, 64, 3)); self.assertTrue(np.isfinite(second).all() and np.isfinite(third).all())
        self.assertEqual(session.scene_cuts, 1)
        self.assertEqual(summary.get("frames"), 3)

    def test_first_frame_stream_applies_the_recipe_on_the_swift_side(self):
        session = NRKStreamSession(self.package, 64, 48, temporal=False, nrk=NRK, processing_scale=2, detail_strength=2)
        frame = np.random.default_rng(1).random((48, 64, 3)).astype(np.float32)
        output = session.process_frame(frame)
        summary = session.close()
        self.assertEqual(output.shape, (48, 64, 3)); self.assertEqual(summary.get("mode"), "first-frame")
