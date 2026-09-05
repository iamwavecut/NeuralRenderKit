import unittest

import numpy as np

from mlxdlss import NeuralRenderingPipeline
from mlxdlss.features import NetworkGeometry, deterministic_noise, make_features, scaled_color
from mlxdlss.temporal import (
    TemporalOptions, TemporalSession, compose_temporal, extend_features, make_temporal_features, normalize_pixel_motion, sample_history,
)

from .synthetic import synthetic_weights


class TemporalMathTests(unittest.TestCase):
    def test_pixel_motion_uses_signed_scale_and_effective_extent(self):
        pixel = np.array([[[2, -4], [-1, 3]]], dtype=np.float32)
        motion = normalize_pixel_motion(pixel, scale_x=-2, scale_y=0.5, effective_width=8, effective_height=4)
        np.testing.assert_array_equal(motion, [[[-0.5, -0.5], [0.25, 0.375]]])

    def test_pixel_motion_adds_previous_minus_current_jitter(self):
        pixel = np.zeros((1, 1, 2), dtype=np.float32)
        motion = normalize_pixel_motion(pixel, scale_x=1, scale_y=1, effective_width=4, effective_height=2, jitter_dx=1, jitter_dy=-1)
        np.testing.assert_array_equal(motion, [[[0.25, -0.5]]])

    def test_temporal_compose_blends_prediction_with_reprojected_history(self):
        color = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
        head = np.array([[[1, -1, 0, 0]]], dtype=np.float32)
        features = np.zeros((1, 1, 16), dtype=np.float32); features[..., 7] = 0.0625; features[..., 8] = -0.0625
        np.testing.assert_allclose(compose_temporal(head, color, features, blend_scale=0.5), [[[0.5875, 0.1125, 0.575]]], atol=1e-6)

    def test_observed_depth_guide_keeps_current_pixel_motion(self):
        current = np.full((3, 3, 3), 0.5, dtype=np.float32)
        history = np.full((3, 3, 3), 0.5, dtype=np.float32); history[1, 2] = [1, 0.5, 0]
        motion = np.zeros((3, 3, 2), dtype=np.float32); motion[2, 2, 0] = 1 / 3
        depth = np.array([[0.9, 0.9, 0.9], [0.9, 0.8, 0.9], [0.9, 0.9, 0.2]], dtype=np.float32)[..., None]
        observed = make_temporal_features(current, history, motion, frame_index=1, depth=depth)
        np.testing.assert_array_equal(observed[1, 1, 7:10], [0, 0, 0])
        closest = make_temporal_features(current, history, motion, frame_index=1, depth=depth, depth_guide="closest")
        self.assertGreater(closest[1, 1, 7], 0)

    def test_five_tap_catmull_rom_changes_fractional_history_sample(self):
        current = np.full((1, 4, 3), 0.5, dtype=np.float32)
        history = np.array([[[0, 0.5, 0.5], [0, 0.5, 0.5], [1, 0.5, 0.5], [0, 0.5, 0.5]]], dtype=np.float32)
        motion = np.zeros((1, 4, 2), dtype=np.float32); motion[..., 0] = 0.125
        features = make_temporal_features(current, history, motion, frame_index=1)
        np.testing.assert_array_equal(features[0, 1, 7:10], [0.0078125, 0, 0])

    def test_integer_shift_reprojects_history_exactly(self):
        rng = np.random.default_rng(0)
        history = rng.random((12, 16, 3)).astype(np.float32)
        current = rng.random((12, 16, 3)).astype(np.float32)
        motion = np.zeros((12, 16, 2), dtype=np.float32); motion[..., 0] = 3 / 16; motion[..., 1] = -2 / 12
        features = make_temporal_features(current, history, motion, frame_index=2)
        np.testing.assert_allclose(features[2:10, 0:13, 7:10], scaled_color(history[0:8, 3:16]), atol=1e-6)

    def test_extend_features_mirrors_and_regenerates_noise(self):
        color = np.random.default_rng(1).random((5, 6, 3)).astype(np.float32)
        features = make_features(color, frame_index=3)
        geometry = NetworkGeometry(output_width=6, output_height=5, network_width=8, network_height=7)
        extended = extend_features(features, geometry, 3)
        self.assertEqual(extended.shape, (7, 8, 16))
        np.testing.assert_array_equal(extended[:5, :6], features)
        np.testing.assert_array_equal(extended[..., 0:3], deterministic_noise(7, 8, 3))
        np.testing.assert_array_equal(extended[6, :6, 4:7], features[2, :, 4:7])   # row 6 mirrors to 2*5-2-6 = 2


class TemporalSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = NeuralRenderingPipeline(synthetic_weights(), device="cpu")

    def test_session_uses_history_and_resets_on_scene_cut(self):
        calls = []
        def motion(current, previous):
            calls.append(1); return np.zeros((*current.shape[:2], 2), dtype=np.float32)
        session = TemporalSession(self.pipeline, options=TemporalOptions(scene_cut_threshold=0.3), motion=motion)
        rng = np.random.default_rng(2)
        frame = rng.random((24, 32, 3)).astype(np.float32) * 0.2
        first = session.process(frame)
        second = session.process(frame)
        self.assertEqual(session.frame_index, 2); self.assertEqual(len(calls), 1)
        self.assertEqual(first.shape, (24, 32, 3)); self.assertTrue(np.isfinite(second).all())
        session.process(np.clip(frame + 0.7, 0, 1))   # a scene cut
        self.assertEqual(session.scene_cuts, 1); self.assertEqual(session.frame_index, 1); self.assertEqual(len(calls), 1)

    def test_zero_motion_and_engine_motion_paths(self):
        session = TemporalSession(self.pipeline, motion="zero")
        frame = np.random.default_rng(3).random((16, 16, 3)).astype(np.float32)
        session.process(frame)
        engine = np.zeros((16, 16, 2), dtype=np.float32)
        output = session.process(frame, motion=engine)
        self.assertEqual(output.shape, (16, 16, 3))
        with self.assertRaises(ValueError):
            TemporalSession(self.pipeline, motion="magic")

    def test_flow_estimator_recovers_an_integer_shift(self):
        try:
            from mlxdlss.temporal import FlowMotionEstimator
            estimator = FlowMotionEstimator()
        except RuntimeError:
            self.skipTest("OpenCV not installed")
        rng = np.random.default_rng(4)
        base = rng.random((80, 112, 3)).astype(np.float32)
        from mlxdlss.composition import blur, gaussian_kernel
        base = np.stack([blur(base[..., c], gaussian_kernel(1.5)) for c in range(3)], axis=-1)
        previous = base[8:72, 8:104]; current = base[10:74, 5:101]   # content moved by (dx=-3, dy=+2) in pixels
        motion = estimator(current, previous)
        inner = motion[12:-12, 12:-12]
        self.assertAlmostEqual(float(inner[..., 0].mean() * 96), -3, delta=0.6)
        self.assertAlmostEqual(float(inner[..., 1].mean() * 64), 2, delta=0.6)
