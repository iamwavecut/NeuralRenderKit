import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

import numpy
from safetensors.numpy import save_file


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Tools"))

import package_neural_rendering_transformer


class PackageNeuralRenderingTransformerTests(unittest.TestCase):
    def test_logical_safetensors_becomes_external_model_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "logical.safetensors"
            save_file(
                {
                    "half": numpy.zeros((2, 3), dtype=numpy.float16),
                    "single": numpy.ones((4,), dtype=numpy.float32),
                },
                source,
                metadata={
                    "format": "dlssnr-logical-v17",
                    "fully_logical": "true",
                },
            )
            destination = root / "NeuralRendering.nrkmodel"

            package_neural_rendering_transformer.package(source, destination)

            packaged_weights = destination / "weights.safetensors"
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(
                manifest["architecture"], "nrk.neural-rendering-transformer.v1"
            )
            self.assertEqual(manifest["inputs"][0]["shape"], [1, "height", "width", 16])
            self.assertEqual(manifest["outputs"][0]["shape"], [1, "height", "width", 4])
            self.assertEqual(
                manifest["weights"]["sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["weights"]["tensors"],
                [
                    {"dataType": "float16", "name": "half", "shape": [2, 3]},
                    {"dataType": "float32", "name": "single", "shape": [4]},
                ],
            )
            self.assertEqual(packaged_weights.read_bytes(), source.read_bytes())

    def test_nonlogical_source_and_existing_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "packed.safetensors"
            save_file(
                {"packed": numpy.zeros((4,), dtype=numpy.uint8)},
                source,
                metadata={"format": "dlssnr-WEIGHTS_HT-packed-u8-v2"},
            )
            with self.assertRaisesRegex(ValueError, "source format"):
                package_neural_rendering_transformer.package(
                    source, root / "model.nrkmodel"
                )

            destination = root / "existing.nrkmodel"
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                package_neural_rendering_transformer.package(source, destination)


if __name__ == "__main__":
    unittest.main()
