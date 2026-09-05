"""Chunked evaluation (window attention and feed-forward branches in bounded pieces) must match the unchunked graph."""
import unittest

import torch

from mlxdlss import model as model_module
from mlxdlss.model import NeuralRenderingModel

from .synthetic import synthetic_weights


class ChunkedEvaluationTests(unittest.TestCase):
    def test_chunked_forward_matches_whole_frame(self):
        torch.manual_seed(1)
        network = NeuralRenderingModel(synthetic_weights(seed=3))
        network.eval()
        value = torch.rand(1, 96, 128, 16)
        saved = model_module.CHUNK_TOKENS
        try:
            model_module.CHUNK_TOKENS = 0
            with torch.no_grad():
                whole = network(value)
            model_module.CHUNK_TOKENS = 200   # a few windows per chunk, ragged last chunk
            with torch.no_grad():
                chunked = network(value)
        finally:
            model_module.CHUNK_TOKENS = saved
        self.assertEqual(whole.shape, chunked.shape)
        # Every token and window sees the same operations, but a BLAS may round a
        # GEMM differently for a different row count (macOS runners do, Linux and
        # Windows do not); the E4M3 publish then flips a few values by one quantum.
        difference = (whole - chunked).abs()
        self.assertLess(difference.max().item(), 2e-3, f"chunked evaluation differs by {difference.max().item()}")
        self.assertLess(difference.mean().item(), 2e-6, f"chunked evaluation differs on average by {difference.mean().item()}")

    def test_per_token_handles_ragged_chunks(self):
        saved = model_module.CHUNK_TOKENS
        try:
            model_module.CHUNK_TOKENS = 7
            value = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).reshape(2, 5, 3, 4)
            out = model_module._per_token(lambda t: t * 2 + 1, value)
        finally:
            model_module.CHUNK_TOKENS = saved
        self.assertTrue(torch.equal(out, value * 2 + 1))
