import unittest

import numpy as np

from neuralrenderkit.composition import blur, compose_detail, compose_head, gaussian_kernel, resample


class CompositionTests(unittest.TestCase):
    def test_compose_head_adds_quarter_of_half_rounded_residual(self):
        head = np.array([[[0.5, -0.5, 0.0, 9.0]]], dtype=np.float32)
        color = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        np.testing.assert_array_equal(compose_head(head, color), [[[0.625, 0.375, 0.5]]])
        # the residual is half-rounded before scaling: 0.4 -> 0.39990234375
        head = np.array([[[0.4, 0.4, 0.4, 0.0]]], dtype=np.float32)
        np.testing.assert_allclose(compose_head(head, color), [[[0.5 + 0.39990234375 * 0.25] * 3]], rtol=0, atol=1e-7)

    def test_compose_head_blends_with_mask_red_and_intensity(self):
        head = np.array([[[0.5, 0.5, 0.5, 0.0]]], dtype=np.float32)
        color = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        mask = np.array([[[0.5, 0, 0]]], dtype=np.float32)
        np.testing.assert_array_equal(compose_head(head, color, control_mask=mask), [[[0.5625, 0.5625, 0.5625]]])
        np.testing.assert_allclose(compose_head(head, color, intensity=0), color)

    def test_gaussian_kernel_is_normalised_with_three_sigma_extent(self):
        kernel = gaussian_kernel(4.0)
        self.assertEqual(len(kernel), 25)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=6)

    def test_blur_preserves_constants_with_edge_extension(self):
        plane = np.full((6, 7), 0.3, dtype=np.float32)
        np.testing.assert_allclose(blur(plane, gaussian_kernel(2.0)), plane, atol=1e-6)

    def test_unit_strengths_return_output_unchanged(self):
        source = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
        output = np.clip(source + 0.1, 0, 1)
        self.assertIs(compose_detail(source, output), output)

    def test_zero_strengths_return_source(self):
        source = np.random.default_rng(1).random((8, 8, 3)).astype(np.float32)
        output = np.clip(source + 0.1, 0, 1)
        np.testing.assert_allclose(compose_detail(source, output, detail_strength=0, colour_strength=0), source, atol=1e-6)

    def test_integer_downscale_is_exact_box_average(self):
        image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
        small = resample(image, 2, 2)
        np.testing.assert_allclose(small[0, 0], image[:2, :2].mean(axis=(0, 1)))

    def test_lanczos_upscale_keeps_constant_images(self):
        image = np.full((4, 4, 3), 0.25, dtype=np.float32)
        np.testing.assert_allclose(resample(image, 8, 6), 0.25, atol=1e-6)
