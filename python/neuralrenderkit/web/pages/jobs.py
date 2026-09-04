"""Jobs page: everything queued, running and finished, with progress and downloads."""
from __future__ import annotations

import time

from nicegui import ui

from ..state import get_state
from .common import layout


def _effects_summary(effects: list[dict]) -> str:
    parts = []
    for e in effects:
        if e.get("kind") == "nr":
            parts.append(f"NR {e.get('profile')}" + (" temporal" if e.get("temporal") else ""))
        else:
            parts.append(f"FG x{e.get('factor')} {e.get('mode')}")
    return " → ".join(parts)


@ui.page("/jobs")
def jobs_page() -> None:
    state = get_state()
    with layout("Jobs"):
        ui.label("Jobs").classes("text-2xl")
        container = ui.column().classes("w-full gap-2")

        @ui.refreshable
        def table() -> None:
            jobs = state.store.list()
            if not jobs:
                ui.label("No jobs yet — start one on the Image or Video page.").classes("opacity-70")
                return
            for job in jobs:
                with ui.card().classes("w-full"):
                    with ui.row().classes("w-full items-center justify-between gap-4"):
                        with ui.column().classes("gap-0"):
                            ui.label(f"{job.input_name} · {job.kind} · {_effects_summary(job.effects)}").classes("font-medium")
                            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(job.created))
                            extra = f" · {job.seconds:.0f} s" if job.seconds else ""
                            ui.label(f"{when} · {job.state}{extra}" + (f" · {job.stage}" if job.state == "running" else "")).classes("text-sm opacity-70")
                            if job.error:
                                ui.label(job.error).classes("text-sm text-negative")
                        with ui.row().classes("gap-2 items-center"):
                            if job.state in ("queued", "running"):
                                ui.button("Cancel", on_click=lambda j=job: (state.queue.cancel(j.id), table.refresh())).props("flat color=negative")
                            for i, name in enumerate(job.outputs):
                                ui.link(name, f"/api/jobs/{job.id}/download/{i}", new_tab=True)
                            if job.preview:
                                ui.link("preview", f"/api/jobs/{job.id}/preview", new_tab=True)
                            ui.button(icon="folder", on_click=lambda j=job: ui.notify(str(state.store.folder(j.id)))).props("flat dense").tooltip("show the job folder")
                            if job.state in ("done", "failed", "cancelled"):
                                ui.button(icon="delete", on_click=lambda j=job: (state.store.delete(j.id), table.refresh())).props("flat dense color=negative")
                    if job.state == "running":
                        ui.linear_progress(value=job.progress, show_value=False).classes("w-full")

        with container:
            table()
        ui.timer(1.0, table.refresh)
