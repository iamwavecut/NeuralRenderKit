"""Jobs page: everything queued, running and finished, with progress and downloads."""
from __future__ import annotations

import time

from nicegui import ui

from ..state import get_state
from .common import card, layout, page_heading, status_chip


def _effects_summary(effects: list[dict]) -> str:
    parts = []
    for e in effects:
        if e.get("kind") == "nr":
            parts.append(f"neural rendering · {e.get('profile')}" + (" · temporal" if e.get("temporal") else ""))
        else:
            parts.append(f"frame generation ×{e.get('factor')} · {e.get('mode')}")
    return "  →  ".join(parts)


@ui.page("/jobs")
def jobs_page() -> None:
    state = get_state()
    with layout("Jobs"):
        page_heading("Jobs", "One job runs at a time; results stay in the output folder until deleted.")
        container = ui.column().classes("w-full gap-3")

        @ui.refreshable
        def table() -> None:
            jobs = state.store.list()
            if not jobs:
                with card():
                    ui.label("No jobs yet — start one on the Image or Video page.").classes("nrk-muted")
                return
            for job in jobs:
                with card():
                    with ui.row().classes("w-full items-center justify-between gap-4"):
                        with ui.row().classes("items-center gap-4"):
                            status_chip(job.state)
                            with ui.column().classes("gap-0"):
                                ui.label(job.input_name).classes("font-medium")
                                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(job.created))
                                extra = f" · {job.seconds:.0f} s" if job.seconds else ""
                                ui.label(f"{when} · {job.kind} · {_effects_summary(job.effects)}{extra}").classes("text-sm nrk-muted")
                                if job.state == "running" and job.stage:
                                    ui.label(job.stage).classes("text-sm nrk-muted")
                                if job.error:
                                    ui.label(job.error).classes("text-sm").style("color: #b91c1c")
                        with ui.row().classes("gap-1 items-center"):
                            if job.state in ("queued", "running"):
                                ui.button("Cancel", on_click=lambda j=job: (state.queue.cancel(j.id), table.refresh())).props("flat dense no-caps color=negative")
                            for i, name in enumerate(job.outputs):
                                ui.button(name, icon="download", on_click=lambda j=job, i=i: ui.navigate.to(f"/api/jobs/{j.id}/download/{i}", new_tab=True)).props("flat dense no-caps")
                            if job.preview:
                                ui.button("preview", icon="play_circle", on_click=lambda j=job: ui.navigate.to(f"/api/jobs/{j.id}/preview", new_tab=True)).props("flat dense no-caps")
                            ui.button(icon="folder_open", on_click=lambda j=job: ui.notify(str(state.store.folder(j.id)))).props("flat round dense").tooltip("show the job folder")
                            if job.state in ("done", "failed", "cancelled"):
                                ui.button(icon="delete_outline", on_click=lambda j=job: (state.store.delete(j.id), table.refresh())).props("flat round dense color=negative").tooltip("delete the job and its files")
                    if job.state == "running":
                        ui.linear_progress(value=job.progress, show_value=False).props("rounded size=6px color=primary track-color=grey-4").classes("w-full")

        with container:
            table()
        ui.timer(1.0, table.refresh)
