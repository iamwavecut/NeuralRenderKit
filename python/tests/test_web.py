import io
import json
import pathlib
import shutil
import subprocess
import tempfile
import time
import unittest

import importlib.util

import numpy as np

if importlib.util.find_spec("pydantic") is None or importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest("the web front end needs the 'web' extra (pip install 'mlxdlss[web]')")

from mlxdlss.web.effects import FrameGen, NeuralRender, describe_effects, media_kind, parse_effects, validate_chain
from mlxdlss.web.jobs import JobQueue, JobStore
from mlxdlss.web.settings import Settings

from .synthetic import synthetic_weights, write_logical_safetensors
from .test_framegen import synthetic_framegen_weights

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class EffectTests(unittest.TestCase):
    def test_media_kind(self):
        self.assertEqual(media_kind("clip.MP4"), "video")
        self.assertEqual(media_kind("photo.jpeg"), "image")
        with self.assertRaises(ValueError):
            media_kind("notes.txt")

    def test_parse_and_validate(self):
        effects = parse_effects([{"kind": "nr", "profile": "standard"}, {"kind": "fg", "mode": "slowmo", "factor": 4, "audio": "stretch"}])
        self.assertIsInstance(effects[0], NeuralRender)
        self.assertIsInstance(effects[1], FrameGen)
        validate_chain(effects, "video")
        with self.assertRaises(ValueError):
            validate_chain(effects, "image")
        with self.assertRaises(ValueError):
            validate_chain([], "video")
        with self.assertRaises(ValueError):
            validate_chain([effects[1], FrameGen()], "video")
        with self.assertRaises(ValueError):
            parse_effects([{"kind": "fg", "mode": "fps", "audio": "stretch"}])
        with self.assertRaises(ValueError):
            parse_effects([{"kind": "nr", "profile": "no-such-profile"}])
        with self.assertRaises(ValueError):
            parse_effects([{"kind": "nr", "temporal": True, "processing_scale": 2}])
        with self.assertRaises(ValueError):
            parse_effects([{"kind": "fg", "factor": 5}])

    def test_describe(self):
        d = describe_effects(mlxdlss_available=False, fg_weights=True, nr_weights=False)
        kinds = {e["kind"]: e for e in d["effects"]}
        self.assertTrue(kinds["fg"]["available"]); self.assertFalse(kinds["nr"]["available"])
        self.assertEqual(kinds["fg"]["fields"]["factor"]["choices"], [2, 3, 4])


class JobQueueTests(unittest.TestCase):
    def test_store_queue_cancel_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            store = JobStore(root)
            seen = []

            def runner(job, folder, report, should_stop):
                for i in range(5):
                    if should_stop():
                        return []
                    report("step", i / 5, i, 5); time.sleep(0.05)
                out = folder / "result.png"; out.write_bytes(b"x")
                return [out]

            queue = JobQueue(store, runner, on_change=lambda j: seen.append(j.state))
            job = store.create("a.png", [{"kind": "nr"}], data=b"data")
            self.assertTrue((store.folder(job.id) / "input.png").exists())
            queue.submit(job)
            for _ in range(100):
                if store.get(job.id).state == "done":
                    break
                time.sleep(0.05)
            self.assertEqual(store.get(job.id).state, "done")
            self.assertEqual(store.get(job.id).outputs, ["result.png"])
            self.assertIn("running", seen)
            # cancellation of a running job
            job2 = store.create("b.png", [{"kind": "nr"}], data=b"data"); queue.submit(job2)
            time.sleep(0.08); queue.cancel(job2.id)
            for _ in range(100):
                if store.get(job2.id).state in ("cancelled", "done"):
                    break
                time.sleep(0.05)
            self.assertEqual(store.get(job2.id).state, "cancelled")
            # invalid chain is rejected before a folder exists
            with self.assertRaises(ValueError):
                store.create("c.png", [{"kind": "fg"}], data=b"x")
            # a fresh store reloads the finished jobs from disk
            again = JobStore(root)
            self.assertEqual({j.id for j in again.list()}, {job.id, job2.id})

    def test_runner_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(pathlib.Path(directory))

            def runner(job, folder, report, should_stop):
                raise RuntimeError("boom")

            queue = JobQueue(store, runner)
            job = store.create("a.png", [{"kind": "nr"}], data=b"data"); queue.submit(job)
            for _ in range(100):
                if store.get(job.id).state == "failed":
                    break
                time.sleep(0.05)
            self.assertEqual(store.get(job.id).state, "failed")
            self.assertIn("boom", store.get(job.id).error)
            self.assertTrue((store.folder(job.id) / "error.log").exists())


def _png_bytes(width: int = 48, height: int = 32) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    Image.fromarray((rng.random((height, width, 3)) * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


class ApiAndRunnerTests(unittest.TestCase):
    """The FastAPI routes on a WebState with synthetic weights (no NiceGUI server)."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        root = pathlib.Path(cls.directory)
        cls.nr_weights = root / "nr.safetensors"
        write_logical_safetensors(cls.nr_weights, synthetic_weights())
        cls.fg_weights = root / "fg.safetensors"
        from safetensors.torch import save_file

        save_file(synthetic_framegen_weights(), str(cls.fg_weights), metadata={"format": "dlssg-framegen-dense-v1"})
        settings = Settings(root=str(root / "web"), nr_weights=str(cls.nr_weights), fg_weights=str(cls.fg_weights), backend="torch", device="cpu")
        from mlxdlss.web.state import WebState
        from mlxdlss.web.api import build_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        cls.state = WebState(settings)
        app = FastAPI()
        app.include_router(build_router(cls.state))
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def _wait(self, job_id: str, timeout: float = 120.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.client.get(f"/api/jobs/{job_id}").json()
            if data["state"] in ("done", "failed", "cancelled"):
                return data
            time.sleep(0.1)
        self.fail("job did not finish")

    def test_effects_endpoint(self):
        data = self.client.get("/api/effects").json()
        self.assertEqual({e["kind"] for e in data["effects"]}, {"nr", "fg"})
        self.assertTrue(data["backends"]["torch"])

    def test_image_job_runs_neural_rendering(self):
        response = self.client.post("/api/jobs", files={"file": ("photo.png", _png_bytes(), "image/png")},
                                    data={"effects": json.dumps([{"kind": "nr", "profile": "standard"}])})
        self.assertEqual(response.status_code, 201, response.text)
        job = self._wait(response.json()["id"])
        self.assertEqual(job["state"], "done", job.get("error"))
        self.assertEqual(job["outputs"], ["result.png"])
        download = self.client.get(f"/api/jobs/{job['id']}/download/0")
        self.assertEqual(download.status_code, 200)
        self.assertTrue(download.content.startswith(b"\x89PNG"))
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/input").status_code, 200)

    def test_bad_requests(self):
        response = self.client.post("/api/jobs", files={"file": ("photo.png", _png_bytes(), "image/png")},
                                    data={"effects": json.dumps([{"kind": "fg"}])})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/jobs/nope").status_code, 404)
        self.assertEqual(self.client.post("/api/jobs/nope/cancel").status_code, 404)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
    def test_video_job_chains_frame_generation_after_neural_rendering(self):
        src = pathlib.Path(self.directory) / "src.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
        response = self.client.post("/api/jobs", files={"file": ("clip.mp4", src.read_bytes(), "video/mp4")},
                                    data={"effects": json.dumps([{"kind": "nr"}, {"kind": "fg", "factor": 2}])})
        self.assertEqual(response.status_code, 201, response.text)
        job = self._wait(response.json()["id"], timeout=600)
        self.assertEqual(job["state"], "done", job.get("error"))
        self.assertEqual(job["outputs"], ["result.mp4"])
        from mlxdlss.video import probe

        info = probe(self.state.store.folder(job["id"]) / "result.mp4")
        self.assertEqual(info.frame_count, 7)   # 4 frames -> 4 + 3 generated
        self.assertAlmostEqual(info.fps, 20.0, places=3)
        self.assertEqual(job["preview"], "preview.mp4")
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/preview").status_code, 200)
