import pathlib
import sys
import tempfile
import unittest

import numpy
from safetensors.numpy import safe_open, save_file


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Tools"))

import unpack_dlssnr_weights


class UnpackDLSSNRWeightsTests(unittest.TestCase):
    def test_qmma_e4m3_b_fragment_layout_becomes_k_by_n_matrix(self):
        packed = numpy.arange(32 * 64, dtype=numpy.uint16).astype(numpy.uint8)

        matrix = unpack_dlssnr_weights.unpack_qmma_e4m3_matrix(
            packed, input_features=32, output_features=64
        )

        self.assertEqual(matrix.shape, (32, 64))
        self.assertEqual(matrix.dtype, numpy.uint8)
        self.assertEqual(matrix[0, 0], packed[0])
        self.assertEqual(matrix[1, 0], packed[1])
        self.assertEqual(matrix[2, 0], packed[16])
        self.assertEqual(matrix[4, 0], packed[32])
        self.assertEqual(matrix[8, 0], packed[2])
        self.assertEqual(matrix[16, 0], packed[4])
        self.assertEqual(matrix[0, 1], packed[64])
        self.assertEqual(matrix[0, 8], packed[8])
        self.assertEqual(matrix[0, 16], packed[0x200])
        self.assertEqual(matrix[31, 63], packed[-1])

    def test_qmma_layout_supports_one_32_column_tile(self):
        packed = numpy.arange(32 * 32, dtype=numpy.uint16).astype(numpy.uint8)

        matrix = unpack_dlssnr_weights.unpack_qmma_e4m3_matrix(
            packed, input_features=32, output_features=32
        )

        self.assertEqual(matrix.shape, (32, 32))
        self.assertEqual(matrix[0, 0], packed[0])
        self.assertEqual(matrix[2, 0], packed[16])
        self.assertEqual(matrix[8, 0], packed[2])
        self.assertEqual(matrix[0, 8], packed[8])
        self.assertEqual(matrix[0, 16], packed[0x200])
        self.assertEqual(matrix[31, 31], packed[-1])

    def test_hmma_f16_b_fragment_layout_becomes_k_by_n_matrix(self):
        packed = numpy.arange(16 * 32, dtype=numpy.float16)

        matrix = unpack_dlssnr_weights.unpack_hmma_f16_matrix(
            packed.tobytes(), input_features=16, output_features=32
        )

        self.assertEqual(matrix.shape, (16, 32))
        self.assertEqual(matrix.dtype, numpy.float16)
        self.assertEqual(matrix[0, 0], packed[0])
        self.assertEqual(matrix[1, 0], packed[1])
        self.assertEqual(matrix[8, 0], packed[2])
        self.assertEqual(matrix[0, 8], packed[4])
        self.assertEqual(matrix[2, 0], packed[8])
        self.assertEqual(matrix[0, 1], packed[32])
        self.assertEqual(matrix[0, 16], packed[256])
        self.assertEqual(matrix[15, 31], packed[-1])

    def test_attention_bias_decodes_accumulator_tiles_and_token_order(self):
        packed = numpy.arange(2 * 64 * 64, dtype=numpy.float16)

        bias = unpack_dlssnr_weights.unpack_attention_bias(
            packed,
            head_count=2,
        )

        self.assertEqual(bias.shape, (2, 64, 64))
        self.assertTrue(bias.flags.c_contiguous)
        self.assertEqual(bias[0, 0, 0], packed[0])
        self.assertEqual(bias[0, 0, 1], packed[1])
        self.assertEqual(bias[0, 16, 0], packed[2])
        self.assertEqual(bias[0, 0, 4], packed[256])
        self.assertEqual(bias[1, 0, 0], packed[4096])

    def test_head_interleaved_qkv_columns_become_semantic_q_k_v_groups(self):
        weight = numpy.arange(4 * 12, dtype=numpy.float16).reshape(4, 12)

        semantic = unpack_dlssnr_weights._semantic_qkv_weight(
            weight,
            head_count=2,
        )

        numpy.testing.assert_array_equal(
            semantic[0],
            weight[0, [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]],
        )
        self.assertTrue(semantic.flags.c_contiguous)

    def test_nvidia_e4m3_finite_encoding_decodes_exact_landmarks(self):
        encoded = numpy.array(
            [0x00, 0x01, 0x07, 0x08, 0x38, 0x3C, 0x7E, 0x7F, 0x80, 0xB8, 0xFE, 0xFF],
            dtype=numpy.uint8,
        )

        decoded = unpack_dlssnr_weights.decode_e4m3(encoded)

        numpy.testing.assert_array_equal(
            decoded[:7],
            numpy.array(
                [0.0, 2**-9, 7 * 2**-9, 2**-6, 1.0, 1.5, 448.0],
                dtype=numpy.float32,
            ),
        )
        self.assertTrue(numpy.isnan(decoded[7]))
        self.assertTrue(numpy.signbit(decoded[8]))
        self.assertEqual(decoded[9], -1.0)
        self.assertEqual(decoded[10], -448.0)
        self.assertTrue(numpy.isnan(decoded[11]))

    def test_e4m3_decode_is_shape_independent_for_large_matrices(self):
        encoded = numpy.full((128, 640), 0x8D, dtype=numpy.uint8)

        decoded = unpack_dlssnr_weights.decode_e4m3(encoded)

        self.assertEqual(decoded[0, 0], -0.025390625)
        numpy.testing.assert_array_equal(
            decoded[:32, 512:544],
            unpack_dlssnr_weights.decode_e4m3(encoded[:32, 512:544]),
        )

    def test_split_swin_contract_payload_is_split_and_decoded(self):
        matrix_bytes = numpy.zeros(512 * 512, dtype=numpy.uint8)
        matrix_bytes[0] = 0x38
        skip = numpy.linspace(0.0, 1.0, 512, dtype=numpy.float16)
        payload = numpy.concatenate(
            (matrix_bytes, numpy.frombuffer(skip.tobytes(), dtype=numpy.uint8))
        )

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block23.layer1.layer", payload
        )

        self.assertEqual(
            set(tensors),
            {"block23.layer1.weight3", "block23.layer1.ffn_cos_skip"},
        )
        self.assertEqual(tensors["block23.layer1.weight3"].shape, (512, 512))
        self.assertEqual(tensors["block23.layer1.weight3"].dtype, numpy.float16)
        self.assertEqual(tensors["block23.layer1.weight3"][0, 0], 1.0)
        numpy.testing.assert_array_equal(tensors["block23.layer1.ffn_cos_skip"], skip)

    def test_split_swin_ffn_payload_decodes_group_mlp(self):
        payload = numpy.zeros(512 * 1024, dtype=numpy.uint8)
        payload[0] = 0x38
        payload[512 * 512] = 0x40
        payload[512 * 512 + 8 * 64 * 256] = 0x44

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block23.layer0.layer", payload
        )

        self.assertEqual(
            set(tensors),
            {
                "block23.layer0.first_projection_weight",
                "block23.layer0.group_expand_weight",
                "block23.layer0.group_project_weight",
            },
        )
        self.assertEqual(
            tensors["block23.layer0.first_projection_weight"].shape,
            (512, 512),
        )
        self.assertEqual(
            tensors["block23.layer0.group_expand_weight"].shape,
            (8, 64, 256),
        )
        self.assertEqual(
            tensors["block23.layer0.group_project_weight"].shape,
            (8, 256, 64),
        )
        self.assertEqual(tensors["block23.layer0.first_projection_weight"][0, 0], 1.0)
        self.assertEqual(tensors["block23.layer0.group_expand_weight"][0, 0, 0], 2.0)
        self.assertEqual(tensors["block23.layer0.group_project_weight"][0, 0, 0], 3.0)

    def test_split_swin_downsample_payload_exposes_layer4_weight(self):
        matrix_bytes = 512 * 1024
        payload = numpy.zeros(matrix_bytes + 16, dtype=numpy.uint8)
        payload[0] = 0x38

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block30.layer4.layer", payload
        )

        self.assertEqual(set(tensors), {"block30.layer4.weight"})
        self.assertEqual(tensors["block30.layer4.weight"].shape, (512, 1024))
        self.assertEqual(tensors["block30.layer4.weight"][0, 0], 1.0)

    def test_decoder_input_payload_exposes_projection_and_interpolation_sin(self):
        matrix_bytes = 1024 * 512
        interpolation = numpy.linspace(0.0, 1.0, 512, dtype=numpy.float16)
        payload = numpy.concatenate(
            (
                numpy.zeros(matrix_bytes, dtype=numpy.uint8),
                numpy.frombuffer(interpolation.tobytes(), dtype=numpy.uint8),
            )
        )

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block39.layer0.layer", payload
        )

        self.assertEqual(tensors["block39.layer0.conv_weight"].shape, (1024, 512))
        numpy.testing.assert_array_equal(
            tensors["block39.layer0.inp_upsample_sin"], interpolation
        )

    def test_known_payload_with_wrong_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "payload size mismatch"):
            unpack_dlssnr_weights.unpack_known_tensor(
                "block31.layer0.layer", numpy.zeros(16, dtype=numpy.uint8)
            )

    def test_fused_32_channel_swin_payload_is_split_into_logical_parameters(self):
        payload = numpy.zeros(0x50C0, dtype=numpy.uint8)
        payload[0] = 0x38
        payload[0x2010:0x2050] = numpy.frombuffer(
            numpy.ones(32, dtype=numpy.float16).tobytes(), dtype=numpy.uint8
        )
        payload[0x4C60:0x4C64] = numpy.frombuffer(
            numpy.array([0.5], dtype=numpy.float32).tobytes(), dtype=numpy.uint8
        )
        payload[0x5070:0x50B0] = numpy.frombuffer(
            numpy.full(32, 0.75, dtype=numpy.float16).tobytes(),
            dtype=numpy.uint8,
        )

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block1.layer0.layer", payload
        )

        self.assertEqual(tensors["block1.layer0.weight1"].shape, (32, 128))
        self.assertEqual(tensors["block1.layer0.weight2"].shape, (128, 32))
        self.assertEqual(tensors["block1.layer0.weight1"][0, 0], 1.0)
        self.assertEqual(tensors["block1.layer0.qkv_weight"].shape, (32, 96))
        self.assertEqual(tensors["block1.layer0.attn_bias"].shape, (1, 64, 64))
        numpy.testing.assert_array_equal(
            tensors["block1.layer0.ffn_cos_skip"],
            numpy.ones(32, dtype=numpy.float16),
        )
        numpy.testing.assert_array_equal(
            tensors["block1.layer0.attn_scale"],
            numpy.array([0.5], dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            tensors["block1.layer0.attn_cos_skip"],
            numpy.full(32, 0.75, dtype=numpy.float16),
        )

    def test_fused_64_channel_ffn_decodes_twenty_four_branch_matrices(self):
        payload = numpy.zeros(0xF140, dtype=numpy.uint8)
        payload[0x0000:0x0400] = 0x38
        payload[0x1000:0x1400] = 0x40
        payload[0x4000:0x4400] = 0x30
        payload[0x5000:0x5400] = 0xB8
        payload[0x6000] = 0x38
        payload[0x70A0:0x74A0] = 0x38
        payload[0x74A0:0x78A0] = 0x30
        payload[0x78A0:0x7CA0] = 0xB8
        payload[0x7CA0:0x80A0] = 0x40

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block5.layer0.layer", payload
        )

        expand = tensors["block5.layer0.ffn_expand_weight"]
        branch_projection = tensors["block5.layer0.ffn_branch_projection_weight"]
        output_projection = tensors["block5.layer0.ffn_output_projection_weight"]
        self.assertEqual(expand.shape, (2, 4, 2, 32, 32))
        self.assertEqual(branch_projection.shape, (2, 4, 32, 32))
        self.assertEqual(output_projection.shape, (64, 64))
        self.assertEqual(expand[0, 0, 0, 0, 0], 1.0)
        self.assertEqual(expand[0, 0, 1, 0, 0], 2.0)
        self.assertEqual(branch_projection[0, 0, 0, 0], 0.5)
        self.assertEqual(branch_projection[1, 0, 0, 0], -1.0)
        self.assertEqual(output_projection[0, 0], 1.0)
        qkv = tensors["block5.layer0.qkv_weight"]
        self.assertEqual(qkv[0, 0], 1.0)
        self.assertEqual(qkv[0, 32], 2.0)
        self.assertEqual(qkv[0, 64], 0.5)
        self.assertEqual(qkv[0, 128], -1.0)
        self.assertNotIn("block5.layer0.weight1", tensors)
        self.assertNotIn("block5.layer0.weight2", tensors)

    def test_upsample_swin_payload_exposes_adapter_and_sin(self):
        payload = numpy.zeros(0x5900, dtype=numpy.uint8)
        payload[0] = 0x40
        payload[0x1000] = 0x30
        payload[0x2000] = 0x38
        payload[0x2860:0x28A0] = numpy.frombuffer(
            numpy.linspace(-1.0, 1.0, 32, dtype=numpy.float16).tobytes(),
            dtype=numpy.uint8,
        )

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block66.layer0.layer", payload
        )

        self.assertEqual(tensors["block66.layer0.weight0"].shape, (64, 32))
        self.assertEqual(tensors["block66.layer0.weight0"][0, 0], 1.0)
        self.assertEqual(tensors["block66.layer0.weight1"][0, 0], 2.0)
        self.assertEqual(tensors["block66.layer0.weight2"][0, 0], 0.5)
        numpy.testing.assert_array_equal(
            tensors["block66.layer0.sin"],
            numpy.linspace(-1.0, 1.0, 32, dtype=numpy.float16),
        )

    def test_downsample_swin_payload_exposes_weight0_projection(self):
        payload = numpy.zeros(0x58C0, dtype=numpy.uint8)
        payload[0x50B0] = 0x38

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block4.layer0.layer", payload
        )

        self.assertEqual(tensors["block4.layer0.weight0"].shape, (32, 64))
        self.assertEqual(tensors["block4.layer0.weight0"][0, 0], 1.0)

    def test_pre_swin_payload_places_input_adapter_before_attention(self):
        payload = numpy.zeros(0x54C0, dtype=numpy.uint8)
        payload[0] = 0x38
        payload[0x2010:0x2012] = numpy.frombuffer(
            numpy.array([1.0], dtype=numpy.float16).tobytes(),
            dtype=numpy.uint8,
        )
        payload[0x2410:0x2450] = numpy.frombuffer(
            numpy.full(32, 0.75, dtype=numpy.float16).tobytes(),
            dtype=numpy.uint8,
        )

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block0.layer0.layer", payload
        )

        self.assertEqual(tensors["block0.layer0.weight1"][0, 0], 1.0)
        self.assertEqual(tensors["block0.layer0.input_adapter_weight"].shape, (16, 32))
        self.assertEqual(tensors["block0.layer0.input_adapter_weight"][0, 0], 1.0)
        numpy.testing.assert_array_equal(
            tensors["block0.layer0.ffn_cos_skip"],
            numpy.full(32, 0.75, dtype=numpy.float16),
        )

    def test_post_swin_payload_decodes_two_four_channel_output_heads(self):
        payload = numpy.zeros(0x5530, dtype=numpy.uint8)
        payload[0x5130:0x5132] = numpy.frombuffer(
            numpy.array([1.0], dtype=numpy.float16).tobytes(),
            dtype=numpy.uint8,
        )
        payload[0x5330:0x5332] = numpy.frombuffer(
            numpy.array([2.0], dtype=numpy.float16).tobytes(),
            dtype=numpy.uint8,
        )

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block70.layer0.layer", payload
        )

        self.assertEqual(tensors["block70.layer0.out_gain"].shape, (16, 4))
        self.assertEqual(tensors["block70.layer0.out_conv_weight"].shape, (16, 4))
        self.assertEqual(tensors["block70.layer0.out_gain"][0, 0], 1.0)
        self.assertEqual(tensors["block70.layer0.out_conv_weight"][0, 0], 2.0)

    def test_post_swin_payload_splits_input_rotation_sine_and_cosine(self):
        payload = numpy.zeros(0x5530, dtype=numpy.uint8)
        sine = numpy.linspace(-0.5, 0.5, 32, dtype=numpy.float16)
        cosine = numpy.sqrt(1 - sine.astype(numpy.float32) ** 2).astype(numpy.float16)
        payload[0x2050:0x20D0] = numpy.frombuffer(
            numpy.concatenate((sine, cosine)).tobytes(), dtype=numpy.uint8
        )

        tensors = unpack_dlssnr_weights.unpack_known_tensor(
            "block70.layer0.layer", payload
        )

        numpy.testing.assert_array_equal(tensors["block70.layer0.inp_merge_sin"], sine)
        numpy.testing.assert_array_equal(
            tensors["block70.layer0.inp_merge_cos"], cosine
        )

    def test_conversion_metadata_reports_fully_logical_output(self):
        payload = numpy.zeros(0x5530, dtype=numpy.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "packed.safetensors"
            destination = root / "logical.safetensors"
            save_file(
                {"block70.layer0.layer": payload},
                source,
                metadata={"format": unpack_dlssnr_weights.PACKED_FORMAT},
            )

            result = unpack_dlssnr_weights.convert(source, destination)

            with safe_open(destination, framework="numpy") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["fully_logical"], "true")
            self.assertEqual(metadata["opaque_output_tensor_count"], "0")
            self.assertEqual(metadata["opaque_output_tensors"], "[]")
            self.assertEqual(result["opaqueOutputTensorCount"], 0)


if __name__ == "__main__":
    unittest.main()
