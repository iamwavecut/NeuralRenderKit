import unittest

import numpy as np

from neuralrenderkit.features import NetworkGeometry, deterministic_noise, make_features, scaled_color


class FeatureTests(unittest.TestCase):
    def test_matches_swift_preprocessor_pins(self):
        color = np.array([[[0.25, 0.5, 0.75], [1, 0, 0.5]]], dtype=np.float32)
        expected = np.array([
            -0.2196044921875, 1.0283203125, 0.1273193359375, 1, -0.03125, 0, 0.03125, -0.03125, 0, 0.03125, 0, 1, 1, -1, -1, 0,
            0.170166015625, 1.9375, 0.325439453125, 1, 0.0625, -0.0625, 0, 0.0625, -0.0625, 0, 0, 1, 1, -1, -1, 0,
        ], dtype=np.float32).reshape(1, 2, 16)
        np.testing.assert_array_equal(make_features(color), expected)

    def test_vendor_geometry_extends_colour_and_regenerates_noise(self):
        color = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)
        geometry = NetworkGeometry(output_width=2, output_height=1, network_width=5, network_height=1)
        features = make_features(color, geometry=geometry)
        self.assertEqual(features.shape, (1, 5, 16))
        # mirror without repeating the edge: x = 2, 3, 4 -> source 0, 0, 0 (2*2-2-x clamped at 0)
        np.testing.assert_array_equal(features[0, 2:, 4:7], np.repeat(features[0, :1, 4:7], 3, axis=0))
        np.testing.assert_array_equal(features[0, :, 0:3], deterministic_noise(1, 5)[0])

    def test_vendor_aligned_geometry_rounds_to_320_and_multiples_of_64(self):
        self.assertEqual(NetworkGeometry.vendor_aligned(256, 256).network_width, 320)
        self.assertEqual(NetworkGeometry.vendor_aligned(1152, 1216).network_height, 1216)
        self.assertEqual(NetworkGeometry.vendor_aligned(1000, 700).network_width, 1024)
        self.assertEqual(NetworkGeometry.vendor_aligned(1000, 700).network_height, 704)

    def test_mirror_indices_never_repeat_the_edge(self):
        np.testing.assert_array_equal(NetworkGeometry.extended_indices(8, 5), [0, 1, 2, 3, 4, 3, 2, 1])

    def test_noise_depends_on_frame_index(self):
        first = deterministic_noise(4, 4, 0)
        second = deterministic_noise(4, 4, 1)
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(first, deterministic_noise(4, 4, 0))

    def test_control_mask_scales_tone_and_structure_per_pixel(self):
        color = np.zeros((1, 2, 3), dtype=np.float32)
        mask = np.array([[[1, 0.5, 0.25], [0, 0, 0]]], dtype=np.float32)
        features = make_features(color, control_mask=mask)
        self.assertEqual(features[0, 0, 11], 0.5)
        self.assertEqual(features[0, 0, 12], 0.25)
        self.assertEqual(features[0, 1, 11], 0)
        self.assertEqual(features[0, 0, 13], 0)
        self.assertEqual(features[0, 0, 14], 0)

    def test_scaled_colour_uses_half_rounding_chain(self):
        self.assertEqual(scaled_color(np.float32(0.25)), np.float32(-0.03125))
        self.assertEqual(scaled_color(np.float32(1.0)), np.float32(0.0625))
