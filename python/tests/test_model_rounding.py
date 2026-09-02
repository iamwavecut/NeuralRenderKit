import unittest

import torch

from neuralrenderkit.model import e4m3_round_trip, e4m3_round_trip_bitwise


class RoundingTests(unittest.TestCase):
    def test_bitwise_rounding_matches_float8_cast(self):
        generator = torch.Generator().manual_seed(0)
        octaves = torch.arange(-6, 9, dtype=torch.float32)
        mantissa = torch.arange(0, 8, dtype=torch.float32)
        ties = (2.0 ** octaves[:, None]) * (1 + (mantissa[None, :] + 0.5) / 8)
        values = torch.cat([
            torch.rand(1 << 16, generator=generator) * 1000 - 500,
            torch.rand(1 << 14, generator=generator) * 0.02 - 0.01,
            ties.flatten(), -ties.flatten(),
            torch.tensor([0.0, 2**-6, 2**-9, 2**-10, 448.0, 449.0, 1e9, -1e9]),
        ])
        cast = values.clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float32)
        torch.testing.assert_close(e4m3_round_trip_bitwise(values), cast, rtol=0, atol=0)
        torch.testing.assert_close(e4m3_round_trip(values), cast, rtol=0, atol=0)

    def test_half_input_keeps_dtype(self):
        values = torch.tensor([1.0625, -3.3, 0.001], dtype=torch.float16)
        rounded = e4m3_round_trip_bitwise(values)
        self.assertEqual(rounded.dtype, torch.float16)
        torch.testing.assert_close(rounded.float(), values.float().to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
