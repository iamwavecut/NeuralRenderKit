"""The process-wide wiring: settings, job store, queue and the runner behind them."""
from __future__ import annotations

from pathlib import Path

from .jobs import JobQueue, JobStore
from .runners import JobRunner, ModelCache
from .settings import Settings


class WebState:
    def __init__(self, settings: Settings | None = None, *, root: Path | None = None, runner=None):
        self.settings = settings or Settings.load()
        if root is not None:
            self.settings.root = str(root)
        self.settings.outputs.mkdir(parents=True, exist_ok=True)
        self.cache = ModelCache()
        self.store = JobStore(self.settings.outputs)
        self.runner = runner or JobRunner(lambda: self.settings, self.cache)
        self.queue = JobQueue(self.store, self.runner)

    def update_settings(self, **changes) -> None:
        for key, value in changes.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self.settings.save()


STATE: WebState | None = None


def get_state() -> WebState:
    global STATE
    if STATE is None:
        STATE = WebState()
    return STATE
