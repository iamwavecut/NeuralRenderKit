import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Tools"))

import compare_neural_rendering_golden_bundle


class CompareNeuralRenderingGoldenBundleTests(unittest.TestCase):
    def test_validated_bundle_aggregates_literal_frame_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            candidate = root / "candidate"
            candidate.mkdir()
            files = {
                "color.f32": [0.0, 0.5, 1.0],
                "motion.f32": [0.0, 0.0],
                "depth.f32": [1.0],
                "reference.f32": [0.0, 0.5, 1.0],
            }
            for name, values in files.items():
                (root / name).write_bytes(struct.pack(f"<{len(values)}f", *values))
            (candidate / "000000.f32").write_bytes(
                struct.pack("<3f", 0.25, 0.5, 0.5)
            )
            manifest = {
                "schemaVersion": 1,
                "height": 1,
                "width": 1,
                "motionConvention": "pixel-current-to-previous",
                "motionScaleX": -1.0,
                "motionScaleY": 1.0,
                "motionWidth": 1,
                "motionHeight": 1,
                "depthInverted": False,
                "frames": [
                    {
                        "jitterDeltaPixels": [0.25, -0.5],
                        **{
                            role: self.file_record(root / name)
                            for role, name in {
                                "color": "color.f32",
                                "motion": "motion.f32",
                                "depth": "depth.f32",
                                "output": "reference.f32",
                            }.items()
                        },
                    }
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))

            result = compare_neural_rendering_golden_bundle.compare_bundle(
                root, candidate
            )

            self.assertEqual(result["frameCount"], 1)
            self.assertEqual(result["motionConvention"], "pixel-current-to-previous")
            self.assertEqual(
                result.get("jitterDeltaPixelsPerFrame"), [[0.25, -0.5]]
            )
            self.assertTrue(result["allFinite"])
            self.assertEqual(result["maximumAbsoluteError"], 0.5)
            self.assertEqual(result["meanAbsoluteError"], 0.25)
            self.assertAlmostEqual(result["meanSquaredError"], 0.1041666667)
            self.assertEqual(result["frames"][0]["index"], 0)

    def test_rejects_unsafe_paths_and_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            candidate = root / "candidate"
            candidate.mkdir()
            manifest = {
                "schemaVersion": 1,
                "height": 1,
                "width": 1,
                "motionConvention": "normalized-history-uv-offset",
                "depthInverted": False,
                "frames": [
                    {
                        "color": {"file": "../color.f32", "sha256": "0" * 64},
                        "motion": {"file": "motion.f32", "sha256": "0" * 64},
                        "depth": {"file": "depth.f32", "sha256": "0" * 64},
                        "output": {"file": "output.f32", "sha256": "0" * 64},
                    }
                ],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "unsafe color file"):
                compare_neural_rendering_golden_bundle.compare_bundle(root, candidate)

            manifest["frames"][0]["color"]["file"] = "color.f32"
            for name, count in {
                "color.f32": 3,
                "motion.f32": 2,
                "depth.f32": 1,
                "output.f32": 3,
            }.items():
                (root / name).write_bytes(struct.pack(f"<{count}f", *([0.0] * count)))
            (candidate / "000000.f32").write_bytes(struct.pack("<3f", 0, 0, 0))
            (root / "manifest.json").write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                compare_neural_rendering_golden_bundle.compare_bundle(root, candidate)

    def test_candidate_runner_emits_aligned_pixel_jitter_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            frames = []
            for index, jitter in enumerate(([0.0, 0.0], [0.25, -0.5])):
                records = {}
                for role, values in {
                    "color": [0.1, 0.2, 0.3],
                    "motion": [float(index), 0.0],
                    "depth": [1.0],
                    "output": [0.4, 0.5, 0.6],
                }.items():
                    path = root / f"{role}-{index}.f32"
                    path.write_bytes(struct.pack(f"<{len(values)}f", *values))
                    records[role] = self.file_record(path)
                frames.append({"jitterDeltaPixels": jitter, **records})
            manifest = {
                "schemaVersion": 1,
                "height": 1,
                "width": 1,
                "motionConvention": "pixel-current-to-previous",
                "motionScaleX": -2.0,
                "motionScaleY": 1.0,
                "motionWidth": 1,
                "motionHeight": 1,
                "depthInverted": False,
                "frames": frames,
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            output = root / "candidate"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "Tools/run_neural_rendering_golden_bundle.py"),
                    str(root),
                    "/external/NeuralRendering.nrkmodel",
                    str(output),
                    "--executable",
                    "/opt/nrk",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                json.loads(completed.stdout)["arguments"],
                [
                    "/opt/nrk",
                    "run-sequence",
                    "/external/NeuralRendering.nrkmodel",
                    "--input-format",
                    "rgb-temporal-reference",
                    "--motion-format",
                    "pixel",
                    "--motion-scale-x",
                    "-2.0",
                    "--motion-scale-y",
                    "1.0",
                    "--height",
                    "1",
                    "--width",
                    "1",
                    "--depth-inverted",
                    "false",
                    "--output-dir",
                    str(output),
                    "--input",
                    str(root / "color-0.f32"),
                    "--motion",
                    str(root / "motion-0.f32"),
                    "--depth",
                    str(root / "depth-0.f32"),
                    "--jitter-delta-x",
                    "0.0",
                    "--jitter-delta-y",
                    "0.0",
                    "--input",
                    str(root / "color-1.f32"),
                    "--motion",
                    str(root / "motion-1.f32"),
                    "--depth",
                    str(root / "depth-1.f32"),
                    "--jitter-delta-x",
                    "0.25",
                    "--jitter-delta-y",
                    "-0.5",
                ],
            )

    def test_manifest_generator_packages_numbered_raw_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for index in range(2):
                for role, values in {
                    "color": [0.1 + index, 0.2, 0.3],
                    "motion": [float(index), -float(index)],
                    "depth": [1.0 - index * 0.25],
                    "output": [0.4, 0.5 + index, 0.6],
                }.items():
                    (root / f"{role}-{index:06d}.f32").write_bytes(
                        struct.pack(f"<{len(values)}f", *values)
                    )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "Tools/make_neural_rendering_golden_bundle.py"),
                    str(root),
                    "--height",
                    "1",
                    "--width",
                    "1",
                    "--motion-convention",
                    "pixel-current-to-previous",
                    "--motion-scale-x",
                    "-2",
                    "--motion-scale-y",
                    "1",
                    "--motion-width",
                    "1",
                    "--motion-height",
                    "1",
                    "--depth-inverted",
                    "false",
                    "--jitter-delta",
                    "0,0",
                    "--jitter-delta",
                    "0.25,-0.5",
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertEqual(manifest["motionScaleX"], -2.0)
            self.assertEqual(manifest["depthInverted"], False)
            self.assertEqual(
                [frame["jitterDeltaPixels"] for frame in manifest["frames"]],
                [[0.0, 0.0], [0.25, -0.5]],
            )
            self.assertEqual(
                [frame["color"]["file"] for frame in manifest["frames"]],
                ["color-000000.f32", "color-000001.f32"],
            )
            for frame in manifest["frames"]:
                for role in ("color", "motion", "depth", "output"):
                    self.assertEqual(
                        frame[role]["sha256"],
                        hashlib.sha256((root / frame[role]["file"]).read_bytes()).hexdigest(),
                    )
            _, _, convention, _, jitter = (
                compare_neural_rendering_golden_bundle._validated_frames(root)
            )
            self.assertEqual(convention, "pixel-current-to-previous")
            self.assertEqual(jitter, [[0.0, 0.0], [0.25, -0.5]])

    def file_record(self, path):
        return {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


if __name__ == "__main__":
    unittest.main()
