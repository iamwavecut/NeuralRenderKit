import pathlib
import sys
import unittest

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "Tools"
sys.path.insert(0, str(TOOLS))

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

if torch is not None:
    import neural_rendering_reference as reference


@unittest.skipIf(torch is None, "torch is optional")
class AttentionBiasLayoutTests(unittest.TestCase):
    def test_recovery_is_a_permutation_with_known_bit_positions(self):
        stored = torch.arange(2 * 4096, dtype=torch.float32).reshape(2, 64, 64)

        recovered = reference.recover_attention_bias_layout(stored)

        self.assertEqual(tuple(recovered.shape), (2, 64, 64))
        self.assertEqual(sorted(recovered[0].flatten().tolist()), list(range(4096)))
        expectations = {
            (0, 1): 1 << 0,
            (16, 0): 1 << 1,
            (0, 16): 1 << 2,
            (0, 2): 1 << 3,
            (0, 8): 1 << 4,
            (1, 0): 1 << 5,
            (2, 0): 1 << 6,
            (8, 0): 1 << 7,
            (0, 4): 1 << 8,
            (0, 32): 1 << 9,
            (4, 0): 1 << 10,
            (32, 0): 1 << 11,
            (63, 63): 4095,
        }
        for (query, key), source in expectations.items():
            self.assertEqual(int(recovered[0, query, key]), source, (query, key))
            self.assertEqual(int(recovered[1, query, key]), 4096 + source, (query, key))

    def test_recovery_rejects_other_shapes(self):
        with self.assertRaises(ValueError):
            reference.recover_attention_bias_layout(torch.zeros(1, 32, 32))

    def test_single_head_and_split_blocks_are_swizzled(self):
        for block in (0, 1, 2, 3, 4, 66, 67, 68, 69, 70):
            self.assertTrue(reference.uses_fragment_swizzle(block, 1), block)
        for block in (23, 27, 30, 40, 45, 47):
            self.assertTrue(reference.uses_fragment_swizzle(block, 16), block)
        for block, heads in ((5, 2), (9, 4), (15, 8), (49, 8), (57, 4), (63, 2)):
            self.assertFalse(reference.uses_fragment_swizzle(block, heads), block)

    def test_zero_bias_window_block_is_invariant_under_recovery(self):
        zeros = torch.zeros(1, 64, 64)
        self.assertTrue(torch.equal(reference.recover_attention_bias_layout(zeros), zeros))


if __name__ == "__main__":
    unittest.main()
