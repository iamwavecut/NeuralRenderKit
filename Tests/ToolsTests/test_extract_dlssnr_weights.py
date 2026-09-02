import hashlib
import pathlib
import json
import struct
import sys
import tempfile
import unittest

import numpy
from safetensors.numpy import load_file, safe_open


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Tools"))

import extract_dlssnr_weights


def serialized_weight(name, values):
    data = numpy.asarray(values, dtype="<f2").tobytes()
    name_bytes = name.encode("utf-8")
    trailer = struct.pack("<QQI", 0, 1, len(values))
    inner_size = 8 + 8 + 4 + len(data) + len(trailer)
    return (
        struct.pack("<Q", len(name_bytes))
        + name_bytes
        + struct.pack("<QQQI", inner_size, inner_size, len(data), 1)
        + data
        + trailer
    )


def serialized_map():
    body = b"".join(
        (
            serialized_weight("block0.layer0.layer", [1.0, -2.0, 0.5]),
            serialized_weight("block1.layer0.bias", [0.25, -0.125]),
        )
    )
    return struct.pack("<Q", len(body) + 8) + body


class ExtractDLSSNRWeightsTests(unittest.TestCase):
    def test_validated_resource_blob_becomes_byte_exact_packed_safetensors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            blob_path = root / "WEIGHTS_HT.bin"
            blob_path.write_bytes(serialized_map())
            destination = root / "dlssnr.safetensors"

            summary = extract_dlssnr_weights.extract_resource_blob(
                blob_path, destination
            )

            arrays = load_file(destination)
            self.assertEqual(sorted(arrays), [
                "block0.layer0.layer",
                "block1.layer0.bias",
            ])
            self.assertEqual(arrays["block0.layer0.layer"].dtype, numpy.uint8)
            numpy.testing.assert_array_equal(
                arrays["block0.layer0.layer"],
                numpy.frombuffer(
                    numpy.array([1.0, -2.0, 0.5], dtype="<f2").tobytes(),
                    dtype=numpy.uint8,
                ),
            )
            self.assertEqual(summary["tensorCount"], 2)
            self.assertEqual(summary["payloadByteCount"], 10)
            self.assertEqual(summary["storageDataType"], "uint8")
            self.assertEqual(
                summary["resourceSHA256"],
                hashlib.sha256(serialized_map()).hexdigest(),
            )
            with safe_open(destination, framework="numpy") as handle:
                metadata = handle.metadata()
            self.assertEqual(metadata["format"], "dlssnr-WEIGHTS_HT-packed-u8-v2")
            self.assertEqual(metadata["tensor_count"], "2")
            self.assertEqual(metadata["payload_byte_count"], "10")
            records = json.loads(metadata["source_tensor_records"])
            self.assertEqual(
                records["block0.layer0.layer"],
                {
                    "byteCount": 6,
                    "dimensions": [3],
                    "dtypeCode": 1,
                    "metadata0": 0,
                    "metadata1": 0,
                },
            )
            self.assertIn("opaque", metadata["payload_contract"])

    def test_corrupt_size_dtype_shape_and_duplicate_are_rejected(self):
        valid = serialized_map()

        corrupt_total = bytearray(valid)
        struct.pack_into("<Q", corrupt_total, 0, len(valid) - 1)
        with self.assertRaisesRegex(ValueError, "serialized map size mismatch"):
            extract_dlssnr_weights.parse_weight_map(bytes(corrupt_total))

        corrupt_dtype = bytearray(valid)
        name_length = struct.unpack_from("<Q", corrupt_dtype, 8)[0]
        dtype_offset = 8 + 8 + name_length + 8 + 8 + 8
        struct.pack_into("<I", corrupt_dtype, dtype_offset, 2)
        with self.assertRaisesRegex(ValueError, "unsupported dtype"):
            extract_dlssnr_weights.parse_weight_map(bytes(corrupt_dtype))

        corrupt_shape = bytearray(valid)
        first_record_start = 8
        name_length = struct.unpack_from("<Q", corrupt_shape, first_record_start)[0]
        outer_offset = first_record_start + 8 + name_length
        outer_size = struct.unpack_from("<Q", corrupt_shape, outer_offset)[0]
        element_count_offset = outer_offset + 8 + outer_size - 4
        struct.pack_into("<I", corrupt_shape, element_count_offset, 4)
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            extract_dlssnr_weights.parse_weight_map(bytes(corrupt_shape))

        duplicate_body = serialized_weight("same", [1]) * 2
        duplicate = struct.pack("<Q", len(duplicate_body) + 8) + duplicate_body
        with self.assertRaisesRegex(ValueError, "duplicate tensor: same"):
            extract_dlssnr_weights.parse_weight_map(duplicate)

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            blob_path = root / "WEIGHTS_HT.bin"
            blob_path.write_bytes(serialized_map())
            destination = root / "dlssnr.safetensors"
            destination.write_bytes(b"keep")

            with self.assertRaises(FileExistsError):
                extract_dlssnr_weights.extract_resource_blob(
                    blob_path, destination
                )

            self.assertEqual(destination.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
