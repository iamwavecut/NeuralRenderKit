"""Shared page chrome and the effect editor."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from nicegui import ui

from ..effects import PROFILE_NAMES
from ..state import get_state

NAV = [("Image", "/"), ("Video", "/video"), ("Jobs", "/jobs"), ("Settings", "/settings")]


@contextmanager
def layout(title: str):
    state = get_state()
    ui.page_title(f"{title} · NeuralRenderKit")
    with ui.header().classes("items-center justify-between px-4 py-2"):
        with ui.row().classes("items-center gap-6"):
            ui.label("NeuralRenderKit").classes("text-lg font-medium")
            with ui.row().classes("gap-4"):
                for name, path in NAV:
                    ui.link(name, path).classes("text-white no-underline" + (" font-bold" if name == title else " opacity-80"))
        with ui.row().classes("items-center gap-2 text-sm opacity-80"):
            ui.label(f"backend: {state.settings.backend}" + (" · Metal available" if state.settings.nrk_available() else ""))
            running = sum(1 for j in state.store.list() if j.state in ("queued", "running"))
            ui.label(f"· {running} active" if running else "· idle")
    with ui.column().classes("w-full max-w-5xl mx-auto p-4 gap-4"):
        yield


def effect_editor(kind: str) -> Callable[[], list[dict]]:
    """Cards for the effects available to ``kind`` (image | video); returns a getter for the chain."""
    state = get_state()
    settings = state.settings
    nr_enabled = ui.switch("Neural rendering", value=True)
    with ui.card().classes("w-full").bind_visibility_from(nr_enabled, "value"):
        if not settings.has_nr_weights():
            ui.label("Weights not configured — set them on the Settings page").classes("text-warning text-sm")
        with ui.row().classes("items-center gap-4 flex-wrap"):
            profile = ui.select(list(PROFILE_NAMES), value="standard", label="profile").classes("w-40")
            scale = ui.number("processing scale", value=1.0, min=1.0, max=4.0, step=0.5).classes("w-36")
            detail = ui.number("detail", value=1.0, min=0.0, max=4.0, step=0.1).classes("w-28")
            colour = ui.number("colour", value=1.0, min=0.0, max=4.0, step=0.1).classes("w-28")
            radius = ui.number("detail radius", value=4.0, min=1.0, max=16.0, step=0.5).classes("w-32")
            intensity = ui.number("intensity", value=1.0, min=0.0, max=2.0, step=0.1).classes("w-28")
            temporal = ui.switch("temporal (history)", value=False) if kind == "video" else None
    fg_enabled = fg_mode = fg_factor = fg_audio = order = None
    if kind == "video":
        fg_enabled = ui.switch("Frame generation", value=False)
        with ui.card().classes("w-full").bind_visibility_from(fg_enabled, "value"):
            if not settings.has_fg_weights():
                ui.label("Weights not configured — nrk-weights extract-fg, then set the path on the Settings page").classes("text-warning text-sm")
            with ui.row().classes("items-center gap-4 flex-wrap"):
                fg_mode = ui.select({"fps": "higher frame rate (x factor)", "slowmo": "slow motion (duration x factor)"}, value="fps", label="mode").classes("w-64")
                fg_factor = ui.select([2, 3, 4], value=2, label="factor").classes("w-24")
                fg_audio = ui.select({"copy": "copy audio", "stretch": "stretch audio (slow motion)", "none": "drop audio"}, value="copy", label="audio").classes("w-56")
            order = ui.select({"nr_first": "neural rendering, then frame generation", "fg_first": "frame generation, then neural rendering"},
                              value="nr_first", label="order when both are on").classes("w-96")

    def chain() -> list[dict]:
        effects: list[dict] = []
        if nr_enabled.value:
            nr = {"kind": "nr", "profile": profile.value, "processing_scale": float(scale.value or 1.0), "detail_strength": float(detail.value or 0),
                  "colour_strength": float(colour.value or 0), "detail_radius": float(radius.value or 4.0), "intensity": float(intensity.value or 0)}
            if temporal is not None:
                nr["temporal"] = bool(temporal.value)
            effects.append(nr)
        if fg_enabled is not None and fg_enabled.value:
            fg = {"kind": "fg", "mode": fg_mode.value, "factor": int(fg_factor.value), "audio": fg_audio.value}
            if order is not None and order.value == "fg_first":
                effects.insert(0, fg)
            else:
                effects.append(fg)
        return effects

    return chain


def job_status_card(job_id: str, *, on_done: Callable[[], None] | None = None) -> None:
    """A live card following one job until it finishes."""
    state = get_state()
    with ui.card().classes("w-full") as card:
        title = ui.label().classes("font-medium")
        stage = ui.label().classes("text-sm opacity-80")
        bar = ui.linear_progress(value=0.0, show_value=False).classes("w-full")
        with ui.row().classes("gap-2"):
            cancel = ui.button("Cancel", on_click=lambda: state.queue.cancel(job_id)).props("flat color=negative")
            ui.link("All jobs", "/jobs").classes("self-center text-sm")

    def refresh() -> None:
        job = state.store.get(job_id)
        if job is None:
            timer.deactivate(); return
        title.text = f"{job.input_name} · {job.state}"
        stage.text = job.stage + (f" · {job.seconds:.0f} s" if job.seconds else "")
        bar.value = job.progress
        if job.state in ("done", "failed", "cancelled"):
            cancel.disable(); timer.deactivate()
            if job.state == "failed":
                stage.text = f"failed: {job.error}"
            if on_done is not None:
                on_done()

    timer = ui.timer(0.5, refresh)
    return card
