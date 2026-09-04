"""Chunked evaluation (window attention and feed-forward branches in bounded pieces) must match the unchunked graph."""
import unittest

import torch

from neuralrenderkit import model as model_module
from neuralrenderkit.model import NeuralRenderingModel

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
        difference = (whole - chunked).abs().max().item()
        self.assertLess(difference, 1e-6, f"chunked evaluation differs by {difference}")

    def test_per_token_handles_ragged_chunks(self):
        saved = model_module.CHUNK_TOKENS
        try:
            model_module.CHUNK_TOKENS = 7
            value = torch.arange(2 * 5 * 3 * 4, dtype=torch.float32).reshape(2, 5, 3, 4)
            out = model_module._per_token(lambda t: t * 2 + 1, value)
        finally:
            model_module.CHUNK_TOKENS = saved
        self.assertTrue(torch.equal(out, value * 2 + 1))
