import pathlib
import struct
import sys
import tempfile
import unittest


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Tools"))

import compare_f32


class CompareFloat32Tests(unittest.TestCase):
    def write_values(self, path, values):
        path.write_bytes(struct.pack(f"<{len(values)}f", *values))

    def test_reports_literal_absolute_error_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            reference = root / "reference.f32"
            candidate = root / "candidate.f32"
            self.write_values(reference, [1.0, -2.0, 4.0])
            self.write_values(candidate, [1.0, -1.5, 3.75])

            result = compare_f32.compare(reference, candidate)

            self.assertEqual(result["elementCount"], 3)
            self.assertEqual(result["maximumAbsoluteError"], 0.5)
            self.assertEqual(result["meanAbsoluteError"], 0.25)
            self.assertAlmostEqual(result["meanSquaredError"], 0.1041666667)
            self.assertAlmostEqual(result["rootMeanSquaredError"], 0.3227486122)
            self.assertAlmostEqual(
                result["unitRangePeakSignalToNoiseRatioDecibels"],
                9.82271233,
            )
            self.assertEqual(result["nonFiniteReferenceCount"], 0)
            self.assertEqual(result["nonFiniteCandidateCount"], 0)
            self.assertEqual(result["finitePairCount"], 3)
            self.assertTrue(result["allFinite"])

    def test_reports_non_finite_counts_for_each_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            reference = root / "reference.f32"
            candidate = root / "candidate.f32"
            self.write_values(reference, [float("nan"), 1.0, 2.0])
            self.write_values(candidate, [0.0, float("inf"), float("nan")])

            result = compare_f32.compare(reference, candidate)

            self.assertFalse(result["allFinite"])
            self.assertEqual(result["nonFiniteReferenceCount"], 1)
            self.assertEqual(result["nonFiniteCandidateCount"], 2)
            self.assertEqual(result["finitePairCount"], 0)
            self.assertIsNone(result["maximumAbsoluteError"])
            self.assertIsNone(result["meanAbsoluteError"])
            self.assertIsNone(result["meanSquaredError"])
            self.assertIsNone(result["rootMeanSquaredError"])
            self.assertIsNone(result["unitRangePeakSignalToNoiseRatioDecibels"])

    def test_rejects_different_byte_counts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            reference = root / "reference.f32"
            candidate = root / "candidate.f32"
            self.write_values(reference, [1.0])
            self.write_values(candidate, [1.0, 2.0])

            with self.assertRaisesRegex(ValueError, "byte counts differ"):
                compare_f32.compare(reference, candidate)


if __name__ == "__main__":
    unittest.main()
