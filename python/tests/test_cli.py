import pathlib
import tempfile
import unittest

import numpy as np

from mlxdlss import cli
from mlxdlss.tools import cli as weights_cli

from .synthetic import synthetic_weights, write_logical_safetensors


class CLITests(unittest.TestCase):
    def test_run_writes_an_image_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            weights = root / "weights.safetensors"
            write_logical_safetensors(weights, synthetic_weights())
            source = root / "input.f32"
            np.random.default_rng(0).random((16, 24, 3)).astype("<f4").tofile(source)
            output = root / "output.f32"
            code = cli.main(["run", "--weights", str(weights), "--input", str(source), "--output", str(output), "--width", "24", "--height", "16", "--device", "cpu"])
            self.assertEqual(code, 0)
            self.assertEqual(np.fromfile(output, dtype="<f4").size, 16 * 24 * 3)
            with self.assertRaises(SystemExit):
                cli.main(["run", "--weights", str(weights), "--input", str(source), "--output", str(output), "--width", "24", "--height", "16"])

    def test_weights_cli_packages_logical_weights_for_mlx(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            weights = root / "weights.safetensors"
            write_logical_safetensors(weights, synthetic_weights())
            package = root / "Model.dlssmodel"
            self.assertEqual(weights_cli.main(["mlx", str(weights), str(package)]), 0)
            self.assertTrue((package / "manifest.json").exists())
            self.assertTrue((package / "weights.safetensors").exists())

    def test_sha256_reports_unknown_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "blob.bin"
            path.write_bytes(b"not a dll")
            self.assertEqual(weights_cli.main(["sha256", str(path)]), 0)


class VersionAndInspectTests(unittest.TestCase):
    def test_pe_file_version_reads_the_utf16_string(self):
        from mlxdlss.tools.extract_dlssnr_weights import pe_file_version

        blob = b"junk" + "FileVersion".encode("utf-16-le") + b"\x00\x00\x00\x00" + "310,8,0,0".encode("utf-16-le") + b"\x00\x00tail"
        self.assertEqual(pe_file_version(blob), "310,8,0,0")
        self.assertIsNone(pe_file_version(b"no version info here"))

    def test_inspect_lists_packed_tensors(self):
        import io
        from contextlib import redirect_stdout

        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as directory:
            packed = pathlib.Path(directory) / "packed.safetensors"
            save_file({"block1.layer0.weight": np.zeros(64, dtype=np.uint8)}, str(packed), metadata={"format": "dlssnr-WEIGHTS_HT-packed-u8-v2"})
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(weights_cli.main(["inspect", str(packed)]), 0)
            self.assertIn("block1.layer0.weight", buffer.getvalue())
            self.assertIn("tensors 1", buffer.getvalue())
