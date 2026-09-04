"""Jobs: one input file, an ordered effect chain, a state machine and a single-worker queue.

Every job owns a folder ``<root>/outputs/<id>/`` with the input, the result(s), a
preview and ``job.json``; the queue runs one GPU job at a time in a worker thread
and the UI polls the in-memory state.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .effects import FrameGen, NeuralRender, media_kind, parse_effects, validate_chain

STATES = ("queued", "running", "done", "failed", "cancelled")


@dataclass
class Job:
    id: str
    created: float
    input_name: str
    kind: str                                   # image | video
    effects: list[dict]                         # serialised effect models, in application order
    state: str = "queued"
    progress: float = 0.0                       # 0..1 of the current stage
    stage: str = ""                             # human-readable stage ("neural rendering 12/240")
    frames_done: int = 0
    frames_total: int | None = None
    outputs: list[str] = field(default_factory=list)   # file names inside the job folder
    preview: str | None = None                  # preview file name inside the job folder
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    backend: str = ""

    @property
    def seconds(self) -> float | None:
        if self.started is None:
            return None
        return (self.finished or time.time()) - self.started

    def to_dict(self) -> dict:
        data = asdict(self)
        data["seconds"] = self.seconds
        return data


class JobStore:
    """Job folders under ``root`` plus an in-memory index; thread-safe."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        for path in sorted(self.root.glob("*/job.json")):
            try:
                data = json.loads(path.read_text())
                data.pop("seconds", None)
                job = Job(**data)
                if job.state in ("queued", "running"):
                    job.state, job.error = "failed", "interrupted by a restart"
                self._jobs[job.id] = job
            except (OSError, ValueError, TypeError):
                continue

    def folder(self, job_id: str) -> Path:
        return self.root / job_id

    def create(self, input_name: str, effects_raw, *, data: bytes | None = None, source: Path | None = None) -> Job:
        kind = media_kind(input_name)
        effects = parse_effects(effects_raw)
        validate_chain(effects, kind)
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        folder = self.folder(job_id)
        folder.mkdir(parents=True, exist_ok=False)
        target = folder / ("input" + Path(input_name).suffix.lower())
        if data is not None:
            target.write_bytes(data)
        elif source is not None:
            target.write_bytes(Path(source).read_bytes())
        else:
            raise ValueError("a job needs file data or a source path")
        job = Job(id=job_id, created=time.time(), input_name=input_name, kind=kind, effects=[e.model_dump() for e in effects])
        with self._lock:
            self._jobs[job_id] = job
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        (self.folder(job.id) / "job.json").write_text(json.dumps(job.to_dict(), indent=2))

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def input_path(self, job: Job) -> Path:
        return self.folder(job.id) / ("input" + Path(job.input_name).suffix.lower())

    def delete(self, job_id: str) -> None:
        import shutil

        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(self.folder(job_id), ignore_errors=True)


Runner = Callable[[Job, Path, Callable[[str, float, int, int | None], None], Callable[[], bool]], list[Path]]


class JobQueue:
    """One worker thread; ``runner(job, folder, report, should_stop) -> output paths``."""

    def __init__(self, store: JobStore, runner: Runner, *, on_change: Callable[[Job], None] | None = None):
        self.store = store
        self.runner = runner
        self.on_change = on_change
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="nrk-web-jobs", daemon=True)
        self._thread.start()

    def submit(self, job: Job) -> None:
        job.state = "queued"
        self.store.save(job)
        self._queue.put(job.id)

    def cancel(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        if job is None:
            return False
        with self._lock:
            self._cancel.add(job_id)
        if job.state == "queued":
            job.state = "cancelled"; job.finished = time.time()
            self.store.save(job); self._notify(job)
        return True

    def _notify(self, job: Job) -> None:
        if self.on_change is not None:
            try:
                self.on_change(job)
            except Exception:
                pass

    def _should_stop(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.store.get(job_id)
            if job is None or job.state != "queued":
                continue
            job.state, job.started = "running", time.time()
            self.store.save(job); self._notify(job)

            def report(stage: str, progress: float, done: int, total: int | None, job=job) -> None:
                job.stage, job.progress, job.frames_done, job.frames_total = stage, max(0.0, min(1.0, progress)), done, total
                self._notify(job)

            try:
                outputs = self.runner(job, self.store.folder(job.id), report, lambda: self._should_stop(job.id))
                if self._should_stop(job.id):
                    job.state = "cancelled"
                else:
                    job.outputs = [p.name for p in outputs]
                    job.state, job.progress = "done", 1.0
            except Exception as error:  # the job must never take the worker down
                job.state = "failed"
                job.error = f"{error}"
                (self.store.folder(job.id) / "error.log").write_text(traceback.format_exc())
            finally:
                job.finished = time.time()
                with self._lock:
                    self._cancel.discard(job.id)
                self.store.save(job); self._notify(job)
