import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Tools"))

import neural_rendering_reference


class NeuralRenderingReferenceTests(unittest.TestCase):
    def test_quadratic_gate_matches_recovered_formula(self):
        values = torch.tensor([-5.0, -4.0, -1.0, 0.0, 1.0, 4.0, 5.0])

        actual = neural_rendering_reference.quadratic_gate(values)

        expected = torch.tensor(
            [
                0.0,
                0.0,
                0.5029296875,
                0.89453125,
                1.2861328125,
                1.7890625,
                1.7890625,
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_e4m3_round_trip_matches_vendor_landmarks_and_ties(self):
        values = torch.tensor(
            [0.0, 2**-9, 7 * 2**-9, 2**-6, 1.0625, 1.1875, 448.0, 500.0, -1.0]
        )

    def test_e4m3_round_trip_trace_uses_tensor_only_equivalent(self):
        class RoundTrip(torch.nn.Module):
            def forward(self, value):
                return neural_rendering_reference.e4m3_round_trip(value)

        values = torch.tensor(
            [0.0, 2**-9, 7 * 2**-9, 2**-6, 1.0625, 1.1875, 448.0, 500.0, -1.0]
        )
        traced = torch.jit.trace(RoundTrip(), values, strict=True)

        torch.testing.assert_close(
            traced(values),
            neural_rendering_reference.e4m3_round_trip(values),
            rtol=0,
            atol=0,
        )
        self.assertNotIn("float8", str(traced.inlined_graph))

        actual = neural_rendering_reference.e4m3_round_trip(values)

        torch.testing.assert_close(
            actual,
            torch.tensor([0.0, 2**-9, 7 * 2**-9, 2**-6, 1.0, 1.25, 448.0, 448.0, -1.0]),
            rtol=0,
            atol=0,
        )

    def test_residual_adds_unscaled_branch_and_cosine_scaled_skip(self):
        skip = torch.tensor([1.0, 2.0])
        branch = torch.tensor([10.0, 20.0])

        actual = neural_rendering_reference.cosine_residual(
            skip,
            branch,
            torch.tensor([0.5, -0.25]),
        )

        torch.testing.assert_close(
            actual,
            torch.tensor([10.5, 19.5]),
            rtol=0,
            atol=0,
        )

    def test_cosine_attention_matches_hand_derived_identity_case(self):
        value = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        qkv_weight = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )

        actual = neural_rendering_reference.cosine_attention(
            value,
            qkv_weight=qkv_weight,
            attention_scale=torch.tensor([1.0]),
            attention_bias=torch.zeros(1, 2, 2),
            projection_weight=torch.eye(2),
            head_count=1,
        )

        expected = torch.tensor([[[0.75, 0.28125], [0.28125, 0.75]]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_vendor_cosine_normalize32_matches_half_fragment_landmarks(self):
        values = torch.stack(
            (
                torch.ones(32),
                torch.nn.functional.one_hot(torch.tensor(0), 32).float(),
            )
        )

        actual = neural_rendering_reference.vendor_cosine_normalize(values)

        torch.testing.assert_close(
            actual[0],
            torch.full((32,), 0.1767578125),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual[1], values[1], rtol=0, atol=0)

    def test_vendor_cosine_normalize_keeps_coreml_trace_path_tensor_only(self):
        class Normalize(torch.nn.Module):
            def forward(self, value):
                return neural_rendering_reference.vendor_cosine_publish(
                    value,
                    torch.ones(1),
                )

        traced = torch.jit.trace(
            Normalize(),
            torch.ones(1, 1, 1, 32),
            strict=True,
        )

        self.assertEqual(tuple(traced(torch.ones(1, 1, 1, 32)).shape), (1, 1, 1, 32))
        self.assertNotIn("aten::clamp_min", str(traced.inlined_graph))

    def test_vendor_softmax_matches_recovered_half_bit_affine_landmarks(self):
        logits = torch.full((1, 64), -10.0)
        logits[0, 0] = 0

        actual = neural_rendering_reference.vendor_approximate_softmax(logits)

        self.assertEqual(actual[0, 0], 0.875)
        torch.testing.assert_close(
            actual[0, 1:],
            torch.full((63,), 0.001953125),
            rtol=0,
            atol=0,
        )

    def test_vendor_softmax_supports_recovered_ninety_six_token_launch(self):
        logits = torch.full((1, 96), -10.0)
        logits[0, 0] = 0

        actual = neural_rendering_reference.vendor_approximate_softmax(logits)

        self.assertEqual(tuple(actual.shape), (1, 96))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(actual[0, 0], 0.75)
        torch.testing.assert_close(actual.sum(), torch.tensor(1.0), rtol=0, atol=0.02)

    def test_vendor_softmax_keeps_coreml_trace_path_tensor_only(self):
        class Softmax(torch.nn.Module):
            def forward(self, value):
                return neural_rendering_reference.vendor_approximate_softmax(value)

        traced = torch.jit.trace(Softmax(), torch.zeros(1, 64), strict=True)

        self.assertEqual(tuple(traced(torch.zeros(1, 64)).shape), (1, 64))

    def test_window_partition_uses_nhwc_spatial_order(self):
        value = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)

        windows = neural_rendering_reference.partition_windows(value, 2)

        self.assertEqual(tuple(windows.shape), (4, 4, 1))
        torch.testing.assert_close(
            windows[:, :, 0],
            torch.tensor(
                [
                    [0.0, 1.0, 4.0, 5.0],
                    [2.0, 3.0, 6.0, 7.0],
                    [8.0, 9.0, 12.0, 13.0],
                    [10.0, 11.0, 14.0, 15.0],
                ]
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            neural_rendering_reference.reverse_windows(
                windows,
                batch_count=1,
                height=4,
                width=4,
                window_size=2,
            ),
            value,
            rtol=0,
            atol=0,
        )

    def test_recovered_window_origins_follow_vendor_quadrant_cycle(self):
        expected = [(0, 0), (-4, -4), (0, -4), (-4, 0)]

        self.assertEqual(
            [
                neural_rendering_reference.recovered_window_origin(index)
                for index in range(9, 15)
            ],
            expected + expected[:2],
        )
        self.assertEqual(
            neural_rendering_reference.recovered_window_origin(56),
            (0, -4),
        )
        self.assertEqual(
            neural_rendering_reference.recovered_window_origin(70),
            (-4, -4),
        )

    def test_learned_upsample_interpolates_and_clamps_edges(self):
        value = torch.tensor([0.0, 2.0, 4.0, 6.0]).reshape(1, 2, 2, 1)

        actual = neural_rendering_reference.learned_upsample2(
            value,
            interpolation=torch.tensor([0.5]),
        )

        expected = torch.tensor(
            [
                0.0,
                1.0,
                2.0,
                2.0,
                2.0,
                3.0,
                4.0,
                4.0,
                4.0,
                5.0,
                6.0,
                6.0,
                4.0,
                5.0,
                6.0,
                6.0,
            ]
        ).reshape(1, 4, 4, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_decoder_input_doubles_crops_and_adds_sine_scaled_split_skip(self):
        latent = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 2, 2, 1)
        skip = torch.full((1, 3, 3, 1), 10.0)

        actual = neural_rendering_reference.decoder_input_merge(
            latent,
            skip=skip,
            skip_sine=torch.tensor([0.25]),
        )

        expected = torch.tensor([2.5, 2.5, 3.5, 2.5, 2.5, 3.5, 4.5, 4.5, 5.5]).reshape(
            1, 3, 3, 1
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_window_block_runs_feed_forward_before_attention_residual(self):
        value = torch.arange(1, 33, dtype=torch.float32).reshape(1, 4, 4, 2)

        actual = neural_rendering_reference.window_block(
            value,
            expansion_weight=torch.zeros(2, 4),
            feed_forward_projection_weight=torch.zeros(4, 2),
            feed_forward_cosine=torch.tensor([0.5, 0.5]),
            qkv_weight=torch.zeros(2, 6),
            attention_scale=torch.ones(1),
            attention_bias=torch.zeros(1, 4, 4),
            attention_projection_weight=torch.zeros(2, 2),
            attention_cosine=torch.tensor([0.25, 0.25]),
            head_count=1,
            window_size=2,
        )

        torch.testing.assert_close(actual, value * 0.125, rtol=0, atol=1e-6)

    def test_upsample_window_owns_merged_latent_and_skip_residual(self):
        prefix = "block66.layer0"
        weight0 = torch.zeros(64, 32)
        weight0[0, 0] = 1
        weights = {
            f"{prefix}.weight0": weight0,
            f"{prefix}.weight1": torch.zeros(32, 128),
            f"{prefix}.weight2": torch.zeros(128, 32),
            f"{prefix}.ffn_cos_skip": torch.ones(32),
            f"{prefix}.sin": torch.zeros(32),
            f"{prefix}.qkv_weight": torch.zeros(32, 96),
            f"{prefix}.attn_scale": torch.ones(1),
            f"{prefix}.attn_bias": torch.zeros(1, 64, 64),
            f"{prefix}.projection_weight": torch.zeros(32, 32),
            f"{prefix}.attn_cos_skip": torch.ones(32),
        }
        model = neural_rendering_reference.NeuralRenderingModel(weights)
        latent = torch.zeros(1, 1, 1, 64)
        latent[..., 0] = 1
        skip = torch.zeros(1, 2, 2, 32)

        actual = model._upsample_window(latent, skip, 66, head_count=1)

        expected = torch.zeros_like(actual)
        expected[..., 0] = 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shifted_window_block_zero_pads_and_crops_to_input_shape(self):
        value = torch.arange(1, 33, dtype=torch.float32).reshape(1, 4, 4, 2)

        actual = neural_rendering_reference.window_block(
            value,
            expansion_weight=torch.zeros(2, 4),
            feed_forward_projection_weight=torch.zeros(4, 2),
            feed_forward_cosine=torch.tensor([0.5, 0.5]),
            qkv_weight=torch.zeros(2, 6),
            attention_scale=torch.ones(1),
            attention_bias=torch.zeros(1, 4, 4),
            attention_projection_weight=torch.zeros(2, 2),
            attention_cosine=torch.tensor([0.25, 0.25]),
            head_count=1,
            window_size=2,
            window_origin=(-1, -1),
        )

        self.assertEqual(tuple(actual.shape), tuple(value.shape))
        torch.testing.assert_close(actual, value * 0.125, rtol=0, atol=1e-6)

    def test_branched_feed_forward_sums_k_tiles_before_activation(self):
        value = torch.zeros(1, 1, 1, 64)
        value[..., :32] = torch.arange(1, 33, dtype=torch.float32)
        expansion = torch.zeros(2, 4, 2, 32, 32)
        expansion[0, 0, 0] = torch.eye(32)
        branch_projection = torch.zeros(2, 4, 32, 32)
        branch_projection[0, 0] = torch.eye(32)

        actual = neural_rendering_reference.branched_feed_forward(
            value,
            expansion_weight=expansion,
            branch_projection_weight=branch_projection,
            output_projection_weight=torch.eye(64),
        )

        expected = torch.zeros_like(value)
        # The gated expansion and the per-head branch sum are published as E4M3
        # before their projections (vendor block-5 FFN captures).
        expected[..., :32] = neural_rendering_reference.e4m3_round_trip(
            neural_rendering_reference.quadratic_gate_activation(value[..., :32])
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_split_window_applies_group_mlp_with_quadratic_gate(self):
        # k / 8 for k < 16 is exactly representable in E4M3.
        value = (torch.arange(256, dtype=torch.float32) % 16 / 8).reshape(1, 2, 2, 64)
        expand = torch.zeros(1, 64, 256)
        expand[0, :, :64] = torch.eye(64)
        project = torch.zeros(1, 256, 64)
        project[0, :64] = torch.eye(64)

        actual = neural_rendering_reference.split_window_block(
            value,
            first_projection_weight=torch.eye(64),
            expand_weight=expand,
            project_weight=project,
            feed_forward_projection_weight=torch.eye(64),
            feed_forward_cosine=torch.zeros(64),
            qkv_weight=torch.zeros(64, 192),
            attention_scale=torch.ones(1),
            attention_bias=torch.zeros(1, 4, 4),
            attention_projection_weight=torch.zeros(64, 64),
            attention_cosine=torch.ones(64),
            head_count=1,
            window_size=2,
        )

        # The group MLP output is published as E4M3 before weight3 (vendor
        # ffwd_512_chained captures); an identity projection exposes it.
        expected = neural_rendering_reference.e4m3_round_trip(
            neural_rendering_reference.quadratic_gate_activation(value)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_global_block_attends_across_flattened_tokens(self):
        value = torch.arange(1, 33, dtype=torch.float32).reshape(1, 4, 4, 2)

        actual = neural_rendering_reference.global_block(
            value,
            expansion_weight=torch.zeros(2, 4),
            feed_forward_projection_weight=torch.zeros(4, 2),
            feed_forward_cosine=torch.tensor([0.5, 0.5]),
            qkv_weight=torch.zeros(2, 6),
            attention_scale=torch.ones(1),
            attention_projection_weight=torch.zeros(2, 2),
            attention_cosine=torch.tensor([0.25, 0.25]),
            head_count=1,
        )

        torch.testing.assert_close(actual, value * 0.125, rtol=0, atol=1e-6)

    def test_global_block_scales_cosine_logits_by_square_root_head_width(self):
        value = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        qkv_weight = torch.cat((torch.eye(2), torch.eye(2), torch.eye(2)), dim=1)

        actual = neural_rendering_reference.global_block(
            value,
            expansion_weight=torch.zeros(2, 2),
            feed_forward_projection_weight=torch.zeros(2, 2),
            feed_forward_cosine=torch.ones(2),
            qkv_weight=qkv_weight,
            attention_scale=torch.ones(1),
            attention_projection_weight=torch.eye(2),
            attention_cosine=torch.zeros(2),
            head_count=1,
        )

        expected = torch.tensor([[[[0.8125, 0.203125], [0.203125, 0.8125]]]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_global_block_applies_experimental_logit_cap_only_when_requested(self):
        value = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        qkv_weight = torch.cat((torch.eye(2), torch.eye(2), torch.eye(2)), dim=1)

        def run(scale, logit_cap):
            return neural_rendering_reference.global_block(
                value,
                expansion_weight=torch.zeros(2, 2),
                feed_forward_projection_weight=torch.zeros(2, 2),
                feed_forward_cosine=torch.ones(2),
                qkv_weight=qkv_weight,
                attention_scale=torch.full((1,), scale),
                attention_projection_weight=torch.eye(2),
                attention_cosine=torch.zeros(2),
                head_count=1,
                logit_cap=logit_cap,
            )

        # Scale 10 puts the self-logit at 10*sqrt(2) (published as 14); the
        # opt-in cap clamps it to 3, matching an uncapped block whose logits
        # are exactly 3.
        cap = neural_rendering_reference.EXPERIMENTAL_GLOBAL_ATTENTION_LOGIT_CAP
        self.assertEqual(cap, 3.0)
        capped = run(10.0, cap)
        uncapped = run(10.0, None)
        reference = run(3.0 / math.sqrt(2.0), None)
        torch.testing.assert_close(capped, reference, rtol=0, atol=0)
        self.assertFalse(torch.equal(capped, uncapped))

    def test_global_block_clamps_logits_symmetrically(self):
        # Anti-parallel tokens: the cross logit is -scale*sqrt(2) and the self
        # logit +scale*sqrt(2); the vit_1d kernels clamp both sides to +-3.
        value = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]])
        qkv_weight = torch.cat((torch.eye(2), torch.eye(2), torch.eye(2)), dim=1)

        def run(scale, logit_cap):
            return neural_rendering_reference.global_block(
                value,
                expansion_weight=torch.zeros(2, 2),
                feed_forward_projection_weight=torch.zeros(2, 2),
                feed_forward_cosine=torch.ones(2),
                qkv_weight=qkv_weight,
                attention_scale=torch.full((1,), scale),
                attention_projection_weight=torch.eye(2),
                attention_cosine=torch.zeros(2),
                head_count=1,
                logit_cap=logit_cap,
            )

        capped = run(10.0, 3.0)
        reference = run(3.0 / math.sqrt(2.0), None)
        torch.testing.assert_close(capped, reference, rtol=0, atol=0)

    def test_downsample_averages_two_by_two_before_projection(self):
        value = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4, 1)

        actual = neural_rendering_reference.downsample(
            value,
            weight=torch.tensor([[2.0, -1.0]]),
        )

        expected = torch.tensor(
            [5.0, -2.5, 9.0, -4.5, 21.0, -10.5, 25.0, -12.5]
        ).reshape(1, 2, 2, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_spatial_end_padding_matches_split_transition_extent(self):
        value = torch.ones(1, 6, 6, 1)

        padded = neural_rendering_reference.pad_spatial_end(value, 8)

        self.assertEqual(tuple(padded.shape), (1, 8, 8, 1))
        torch.testing.assert_close(padded[:, :6, :6], value, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(padded[:, 6:, :]).item(), 0)
        self.assertEqual(torch.count_nonzero(padded[:, :, 6:]).item(), 0)

    @unittest.skipUnless(
        os.environ.get("MLXDLSS_LOGICAL_WEIGHTS")
        and os.environ.get("MLXDLSS_NEURAL_RENDERING_PACKAGE"),
        "set external neural-rendering paths to run full parity",
    )
    def test_external_full_model_matches_mlx_head(self):
        model = neural_rendering_reference.load_model(
            pathlib.Path(os.environ["MLXDLSS_LOGICAL_WEIGHTS"])
        ).eval()
        input_values = (
            np.sin(np.arange(128 * 128 * 16, dtype=np.float32) * 0.001)
            .reshape(1, 128, 128, 16)
            .astype("<f4")
        )
        with torch.inference_mode():
            expected = model(torch.from_numpy(input_values.copy())).numpy()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            input_path = root / "input.f32"
            output_path = root / "output.f32"
            input_values.tofile(input_path)
            executable = REPOSITORY_ROOT / ".build/debug/mlxdlss"
            subprocess.run(
                [
                    str(executable),
                    "run",
                    os.environ["MLXDLSS_NEURAL_RENDERING_PACKAGE"],
                    "--input",
                    str(input_path),
                    "--input-format",
                    "model",
                    "--output",
                    str(output_path),
                    "--height",
                    "128",
                    "--width",
                    "128",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            actual = np.fromfile(output_path, dtype="<f4").reshape(1, 128, 128, 4)

        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
