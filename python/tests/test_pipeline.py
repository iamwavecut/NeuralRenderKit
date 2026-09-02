import pathlib
import tempfile
import unittest

import numpy as np
import torch

from neuralrenderkit import NeuralRenderingPipeline, NeuralRenderingSession, load_weights
from neuralrenderkit.pipeline import validate_weights

from .synthetic import synthetic_weights, write_logical_safetensors


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = synthetic_weights()

    def test_validate_rejects_missing_and_misshaped_tensors(self):
        weights = dict(self.weights)
        weights.pop("block70.layer0.out_gain")
        with self.assertRaises(ValueError):
            validate_weights(weights)
        weights = dict(self.weights)
        weights["block70.layer0.out_gain"] = torch.zeros(4, 16)
        with self.assertRaises(ValueError):
            validate_weights(weights)

    def test_load_weights_checks_format_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "weights.safetensors"
            write_logical_safetensors(path, self.weights)
            loaded = load_weights(path)
            self.assertEqual(len(loaded), len(self.weights))
            from safetensors.torch import save_file

            bad = pathlib.Path(directory) / "bad.safetensors"
            save_file(self.weights, str(bad), metadata={"format": "something-else", "fully_logical": "true"})
            with self.assertRaises(ValueError):
                load_weights(bad)

    def test_enhance_runs_the_recovered_graph_on_a_vendor_aligned_extent(self):
        pipeline = NeuralRenderingPipeline(self.weights, device="cpu")
        image = np.random.default_rng(0).random((40, 48, 3)).astype(np.float32)
        result = pipeline.enhance(image)
        self.assertEqual(result.image.shape, (40, 48, 3))
        self.assertEqual(result.network_extent, (320, 320))
        self.assertTrue(np.isfinite(result.image).all())
        self.assertTrue((result.image >= 0).all() and (result.image <= 1).all())
        self.assertIn("network", result.timings)

    def test_processing_scale_and_detail_recipe_keep_the_logical_size(self):
        pipeline = NeuralRenderingPipeline(self.weights, device="cpu")
        image = np.random.default_rng(1).random((32, 40, 3)).astype(np.float32)
        result = pipeline.enhance(image, processing_scale=2, detail_strength=2, colour_strength=1)
        self.assertEqual(result.image.shape, (32, 40, 3))
        self.assertEqual(result.network_extent, (320, 320))

    def test_control_mask_requires_unit_scale_and_profile_is_validated(self):
        pipeline = NeuralRenderingPipeline(self.weights, device="cpu")
        image = np.zeros((8, 8, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            pipeline.enhance(image, control_mask=np.ones_like(image), processing_scale=2)
        with self.assertRaises(ValueError):
            pipeline.enhance(image, profile="vivid")

    def test_session_advances_the_noise_frame_index(self):
        pipeline = NeuralRenderingPipeline(self.weights, device="cpu")
        session = NeuralRenderingSession(pipeline)
        image = np.random.default_rng(2).random((16, 16, 3)).astype(np.float32)
        first = session.process(image).image
        second = session.process(image).image
        self.assertEqual(session.frame_index, 2)
        self.assertFalse(np.array_equal(first, second))
        session.reset()
        np.testing.assert_array_equal(session.process(image).image, first)
