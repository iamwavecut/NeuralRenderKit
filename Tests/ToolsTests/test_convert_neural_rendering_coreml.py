import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Tools"))

import convert_neural_rendering_coreml
import neural_rendering_reference

try:
    import coremltools as ct
except ImportError:
    ct = None


class ConvertNeuralRenderingCoreMLTests(unittest.TestCase):
    def test_nchw_wrapper_preserves_sixteen_to_four_channel_contract(self):
        class FirstFour(torch.nn.Module):
            def forward(self, value):
                return value[..., :4] * 2

        wrapper = convert_neural_rendering_coreml.NCHWHeadWrapper(FirstFour())
        sample = torch.arange(96, dtype=torch.float32).reshape(1, 16, 2, 3)

        output = wrapper(sample)

        self.assertEqual(tuple(output.shape), (1, 4, 2, 3))
        torch.testing.assert_close(output, sample[:, :4] * 2, rtol=0, atol=0)

    def test_existing_destination_is_rejected_before_opening_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            destination = root / "existing.mlpackage"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "destination already exists"):
                convert_neural_rendering_coreml.convert_weights(
                    root / "missing.safetensors",
                    destination,
                    precision="float16",
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_fixed_extent_must_match_recovered_graph_contract(self):
        with self.assertRaisesRegex(ValueError, "at least 128 and multiples of 64"):
            convert_neural_rendering_coreml.convert_weights(
                pathlib.Path("missing.safetensors"),
                pathlib.Path("missing.mlpackage"),
                precision="float16",
                height=320,
                width=319,
            )

    @unittest.skipUnless(ct is not None, "coremltools is optional")
    def test_cosine_residual_converts_without_scalar_dtype_mismatch(self):
        class Residual(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cosine", torch.tensor([0.5, 0.25]))

            def forward(self, value):
                return neural_rendering_reference.cosine_residual(
                    value,
                    value * 2,
                    self.cosine,
                )

        sample = torch.ones(1, 1, 1, 2)
        traced = torch.jit.trace(Residual().eval(), sample, strict=True)

        converted = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
            inputs=[ct.TensorType(name="value", shape=sample.shape, dtype=np.float32)],
        )

        self.assertIsNotNone(converted)

    @unittest.skipUnless(ct is not None, "coremltools is optional")
    def test_downsample_converts_without_rank_six_tensor(self):
        class Downsample(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.tensor([[2.0, -1.0]]))

            def forward(self, value):
                return neural_rendering_reference.downsample(
                    value,
                    weight=self.weight,
                )

        sample = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)
        traced = torch.jit.trace(Downsample().eval(), sample, strict=True)

        converted = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
            inputs=[ct.TensorType(name="value", shape=sample.shape, dtype=np.float32)],
        )

        self.assertIsNotNone(converted)


if __name__ == "__main__":
    unittest.main()
