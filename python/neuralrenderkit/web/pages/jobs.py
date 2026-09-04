"""Jobs page: one list, newest first, with status, progress and downloads."""
from __future__ import annotations

import time

from nicegui import ui

from ..state import get_state
from . import ds
from .common import layout


def _effects_summary(effects: list[dict]) -> str:
    parts = []
    for e in effects:
        if e.get("kind") == "nr":
            parts.append("Neural rendering · " + str(e.get("profile")) + (" · temporal" if e.get("temporal") else ""))
        else:
            parts.append(f"Frame generation ×{e.get('factor')} · " + ("slow motion" if e.get("mode") == "slowmo" else "higher frame rate"))
    return "  →  ".join(parts)


@ui.page("/jobs")
def jobs_page() -> None:
    state = get_state()
    with layout("Jobs", "Jobs run one at a time. Results stay in the output folder until you delete them."):
        with ds.card(tight=True):

            @ui.refreshable
            def rows() -> None:
                jobs = state.store.list()
                if not jobs:
                    ds.empty_state("inbox", "Nothing here yet. Start a job on the Image or Video page.")
                    return
                for job in jobs:
                    with ui.element("div").classes("nrk-list-row"):
                        ds.chip(job.state)
                        with ui.element("div").classes("min-w-0"):
                            ui.label(job.input_name).classes("nrk-list-title truncate")
                            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(job.created))
                            extra = f" · {job.seconds:.0f} s" if job.seconds else ""
                            ui.label(f"{when} · {_effects_summary(job.effects)}{extra}").classes("nrk-list-meta truncate")
                            if job.state == "running":
                                ui.label(job.stage or "").classes("nrk-list-meta nrk-mono")
                                with ui.element("div").classes("mt-2"):
                                    ds.progress().value = job.progress
                            if job.error:
                                ui.label(job.error).classes("nrk-bad nrk-small mt-1")
                        with ui.element("div").classes("nrk-actions"):
                            if job.state in ("queued", "running"):
                                ds.button("Cancel", kind="danger", on_click=lambda j=job: (state.queue.cancel(j.id), rows.refresh()))
                            for i, name in enumerate(job.outputs):
                                ds.icon_button("download", tooltip=f"Download {name}", on_click=lambda j=job, i=i: ui.navigate.to(f"/api/jobs/{j.id}/download/{i}", new_tab=True))
                            if job.preview:
                                ds.icon_button("play_circle", tooltip="Open the side-by-side preview", on_click=lambda j=job: ui.navigate.to(f"/api/jobs/{j.id}/preview", new_tab=True))
                            ds.icon_button("folder_open", tooltip="Show the job folder", on_click=lambda j=job: ui.notify(str(state.store.folder(j.id))))
                            if job.state in ("done", "failed", "cancelled"):
                                ds.icon_button("delete_outline", tooltip="Delete the job and its files", danger=True, on_click=lambda j=job: (state.store.delete(j.id), rows.refresh()))

            rows()
        ui.timer(1.0, rows.refresh)
