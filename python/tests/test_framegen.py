import os
import pathlib
import unittest

import numpy as np
import torch

from mlxdlss.framegen import FRAMEGEN_TENSORS, FrameGenerator, box2, photometric_error
from mlxdlss.tools import extract_dlssg_weights as fgx


def synthetic_framegen_weights(seed: int = 0, scale: float = 0.05) -> dict[str, torch.Tensor]:
    """Random tensors with the real shapes (block0: 16/32/64 channels, block1: 18/16/32)."""
    g = torch.Generator().manual_seed(seed)
    shapes = {}
    for name, cout, cin in (("block0.stem0", 32, 16), ("block0.stem1", 32, 32), ("block0.stem2", 64, 32), ("block1.stem0", 16, 18), ("block1.stem1", 32, 16)):
        shapes[name] = (cout, cin)
    for i in range(8):
        shapes[f"block0.res{i}"] = (64, 64)
        shapes[f"block1.res{i}"] = (32, 32)
    for i in range(3):
        shapes[f"block0.bot0.head{i}"] = (32, 64)
        shapes[f"block1.bot0.head{i}"] = (16, 32)
    shapes["block0.bot1"] = (8, 32)
    shapes["block1.bot1"] = (8, 16)
    weights = {}
    for name, (cout, cin) in shapes.items():
        weights[f"{name}.weight"] = (torch.randn(cout, cin, 3, 3, generator=g) * scale / np.sqrt(9 * cin)).half()
        weights[f"{name}.bias"] = (torch.randn(cout, generator=g) * scale).half()
    return weights


class FrameGenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weights = synthetic_framegen_weights()
        cls.fg = FrameGenerator(cls.weights, device="cpu")

    def test_tensor_list_matches_synthetic_shapes(self):
        self.assertEqual(sorted(FRAMEGEN_TENSORS), sorted(self.weights))

    def test_missing_tensor_is_rejected(self):
        weights = dict(self.weights)
        weights.pop("block1.bot1.bias")
        with self.assertRaises(ValueError):
            FrameGenerator(weights, device="cpu")

    def test_box2_pads_to_tile_multiple(self):
        x = torch.rand(1, 3, 540, 960)
        y = box2(x)
        self.assertEqual(tuple(y.shape), (1, 3, 272, 480))
        self.assertTrue(torch.allclose(y[:, :, :270], torch.nn.functional.avg_pool2d(x, 2)))
        self.assertEqual(float(y[:, :, 270:].abs().max()), 0.0)

    def test_photometric_error_is_blurred_mean_abs_difference(self):
        a = torch.zeros(1, 3, 8, 8)
        b = torch.zeros(1, 3, 8, 8)
        b[0, :, 4, 4] = 1.0
        e = photometric_error(a, b)[0, 0]
        self.assertAlmostEqual(float(e[4, 4]), 0.75 * 0.75, places=6)
        self.assertAlmostEqual(float(e[4, 5]), 0.75 * 0.125, places=6)
        self.assertAlmostEqual(float(e[5, 5]), 0.125 * 0.125, places=6)
        self.assertEqual(float(e[0, 0]), 0.0)

    def test_generate_shapes_and_range(self):
        a = (np.random.default_rng(1).random((96, 160, 3)) * 255).astype(np.uint8)
        b = np.roll(a, 3, axis=1)
        frames = self.fg.generate(a, b, factor=4)
        self.assertEqual(len(frames), 3)
        for f in frames:
            self.assertEqual(f.shape, (96, 160, 3))
            self.assertEqual(f.dtype, np.uint8)
        with self.assertRaises(ValueError):
            self.fg.generate(a, b, factor=1)

    def test_constant_frames_are_invariant(self):
        a = np.full((64, 96, 3), (30, 120, 210), np.uint8)
        out = self.fg.generate(a, a, factor=2)[0]
        # warping and blending a constant image gives the constant back whatever the flows are
        self.assertTrue(np.array_equal(out, a))

    def test_deterministic(self):
        a = (np.random.default_rng(3).random((64, 96, 3)) * 255).astype(np.uint8)
        b = (np.random.default_rng(4).random((64, 96, 3)) * 255).astype(np.uint8)
        self.assertTrue(np.array_equal(self.fg.generate(a, b)[0], self.fg.generate(a, b)[0]))

    def test_mismatched_frames_are_rejected(self):
        with self.assertRaises(ValueError):
            self.fg.interpolate(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 16, 3), np.uint8))


class ExtractorLayoutTests(unittest.TestCase):
    def test_custom_layout_roundtrip(self):
        cout, cin = 16, 8
        dense = np.random.default_rng(0).standard_normal((cout, cin, 3, 3)).astype(np.float16)
        packed = dense.reshape(cout // 8, 8, cin // 8, 8, 3, 3).transpose(0, 4, 5, 2, 1, 3).reshape(cout // 8, 9, cin // 8, 8, 8)
        self.assertTrue(np.array_equal(fgx.custom_dense(np.ascontiguousarray(packed).ravel(), cout, cin), dense))

    def test_plain_layout_roundtrip(self):
        cout, cin = 16, 18
        dense = np.random.default_rng(0).standard_normal((cout, cin, 3, 3)).astype(np.float16)
        packed = dense.reshape(cout, cin, 9).transpose(0, 2, 1)
        self.assertTrue(np.array_equal(fgx.plain_dense(np.ascontiguousarray(packed).ravel(), cout, cin), dense))

    def test_unet_layout_roundtrip(self):
        cout, cin, k = 16, 32, 1
        K = k * k * cin
        dense = np.random.default_rng(0).standard_normal((cout, cin, k, k)).astype(np.float16)
        bias = np.random.default_rng(1).standard_normal(cout).astype(np.float16)
        blob = np.zeros(2 * K * cout + (cout // 8) * 128, np.float16)
        KB = K // 16
        for n in range(cout):
            for kk in range(K):
                ci, tap = kk % cin, kk // cin
                idx = (n // 8) * KB * 128 + (kk // 16) * 128 + (n % 8) * 16 + fgx.FRAG[kk % 16]
                blob[idx] = dense[n, ci, tap // k, tap % k]
            blob[2 * K * cout + (n // 8) * 128 + n % 8] = bias[n]
        w, b = fgx.unet_dense(blob, cout, cin, k)
        self.assertTrue(np.array_equal(w, dense))
        self.assertTrue(np.array_equal(b, bias))

    def test_spec_covers_every_framegen_tensor(self):
        names = set()
        for name, (kind, *_rest) in fgx.SPEC_310_7_0.items():
            if kind == "custom3":
                names.update(f"{name}.head{i}.{p}" for i in range(3) for p in ("weight", "bias"))
            else:
                names.update(f"{name}.{p}" for p in ("weight", "bias"))
        self.assertTrue(set(FRAMEGEN_TENSORS) <= names)


@unittest.skipUnless(os.environ.get("MLXDLSS_FG_WEIGHTS"), "set MLXDLSS_FG_WEIGHTS to a dense frame generation safetensors")
class FrameGenGoldenTests(unittest.TestCase):
    def test_moving_edge_is_placed_between_frames(self):
        fg = FrameGenerator.from_safetensors(pathlib.Path(os.environ["MLXDLSS_FG_WEIGHTS"]), device="cpu")
        h, w = 128, 192
        a = np.zeros((h, w, 3), np.uint8)
        b = np.zeros((h, w, 3), np.uint8)
        a[:, 60:100] = 200
        b[:, 76:116] = 200
        mid = fg.generate(a, b, factor=2)[0]
        column = mid[h // 2, :, 0].astype(int)
        left = int(np.argmax(column > 100))
        self.assertTrue(64 <= left <= 72, f"edge at {left}, expected near 68")
