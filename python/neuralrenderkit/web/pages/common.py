"""Shared page chrome (theme, header, layout), the effect editor and the job card."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from nicegui import ui

from ..effects import PROFILE_NAMES
from ..state import get_state

NAV = [("Image", "/", "image"), ("Video", "/video", "movie"), ("Jobs", "/jobs", "list_alt"), ("Settings", "/settings", "tune")]

CSS = """
:root { --nrk-bg: #f5f6f8; --nrk-card: #ffffff; --nrk-border: #e3e6ea; --nrk-text: #16191d; --nrk-muted: #6b7280; --nrk-accent: #0f766e; --nrk-accent-soft: #e6f4f2; }
body.body--dark { --nrk-bg: #0f1114; --nrk-card: #171a1f; --nrk-border: #272c34; --nrk-text: #e8eaed; --nrk-muted: #9aa3ad; --nrk-accent: #2dd4bf; --nrk-accent-soft: #123b37; }
body { background: var(--nrk-bg) !important; color: var(--nrk-text); font-family: -apple-system, "SF Pro Text", "Segoe UI", Inter, Roboto, sans-serif; }
.nrk-header { background: var(--nrk-card) !important; color: var(--nrk-text) !important; border-bottom: 1px solid var(--nrk-border); box-shadow: none !important; }
.nrk-nav a { color: var(--nrk-muted); text-decoration: none; padding: 6px 10px; border-radius: 8px; font-weight: 500; }
.nrk-nav a:hover { background: var(--nrk-accent-soft); color: var(--nrk-text); }
.nrk-nav a.active { color: var(--nrk-text); background: var(--nrk-accent-soft); }
.nrk-card { background: var(--nrk-card) !important; border: 1px solid var(--nrk-border); border-radius: 14px; box-shadow: none !important; padding: 18px 20px; }
.nrk-title { font-size: 26px; font-weight: 600; letter-spacing: -0.01em; }
.nrk-subtitle { color: var(--nrk-muted); font-size: 14px; }
.nrk-section { font-size: 13px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--nrk-muted); }
.nrk-muted { color: var(--nrk-muted); }
.nrk-drop .q-uploader { background: transparent; box-shadow: none; border: 1.5px dashed var(--nrk-border); border-radius: 12px; width: 100%; }
.nrk-drop .q-uploader__header { background: transparent; color: var(--nrk-text); }
.nrk-drop .q-uploader__list { background: transparent; }
.nrk-chip { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.nrk-chip-queued { background: #e5e7eb; color: #374151; } body.body--dark .nrk-chip-queued { background: #2a3038; color: #cbd5e1; }
.nrk-chip-running { background: #dbeafe; color: #1d4ed8; } body.body--dark .nrk-chip-running { background: #1e3a5f; color: #93c5fd; }
.nrk-chip-done { background: #dcfce7; color: #166534; } body.body--dark .nrk-chip-done { background: #14352a; color: #86efac; }
.nrk-chip-failed { background: #fee2e2; color: #991b1b; } body.body--dark .nrk-chip-failed { background: #3f1d1d; color: #fca5a5; }
.nrk-chip-cancelled { background: #f3f4f6; color: #6b7280; } body.body--dark .nrk-chip-cancelled { background: #23272e; color: #9aa3ad; }
.nrk-ok { color: #15803d; } body.body--dark .nrk-ok { color: #4ade80; }
.nrk-warn { color: #b45309; } body.body--dark .nrk-warn { color: #fbbf24; }
.q-field--outlined .q-field__control { border-radius: 10px; }
"""


@contextmanager
def layout(title: str):
    """Header with navigation, backend/queue status and the theme toggle; a centred content column."""
    state = get_state()
    settings = state.settings
    ui.page_title(f"{title} · NeuralRenderKit")
    ui.colors(primary=("#2dd4bf" if settings.theme_dark else "#0f766e"), secondary="#64748b", accent="#0f766e", positive="#15803d", negative="#b91c1c", warning="#b45309")
    ui.add_head_html(f"<style>{CSS}</style>")
    dark = ui.dark_mode(value=settings.theme_dark)

    def toggle_theme() -> None:
        dark.toggle()
        state.update_settings(theme_dark=bool(dark.value))
        theme_button.props(f"icon={'light_mode' if dark.value else 'dark_mode'}")

    with ui.header().classes("nrk-header items-center justify-between px-6 py-2"):
        with ui.row().classes("items-center gap-8"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("auto_awesome").classes("text-xl").style("color: var(--nrk-accent)")
                ui.label("NeuralRenderKit").classes("text-base font-semibold")
            with ui.row().classes("nrk-nav items-center gap-1"):
                for name, path, _icon in NAV:
                    ui.link(name, path).classes("active" if name == title else "")
        with ui.row().classes("items-center gap-3 text-sm"):
            running = sum(1 for j in state.store.list() if j.state in ("queued", "running"))
            ui.label(f"{running} active" if running else "idle").classes("nrk-muted")
            ui.label("·").classes("nrk-muted")
            backend = settings.backend + (" · Metal" if settings.nrk_available() else "")
            ui.label(backend).classes("nrk-muted")
            theme_button = ui.button(icon="light_mode" if settings.theme_dark else "dark_mode", on_click=toggle_theme).props("flat round dense").tooltip("toggle theme")
    with ui.column().classes("w-full max-w-5xl mx-auto px-6 py-6 gap-5"):
        yield


def page_heading(title: str, subtitle: str) -> None:
    with ui.column().classes("gap-1"):
        ui.label(title).classes("nrk-title")
        ui.label(subtitle).classes("nrk-subtitle")


def card():
    return ui.card().classes("nrk-card w-full gap-3")


def _slider(label: str, *, value: float, minimum: float, maximum: float, step: float) -> ui.slider:
    with ui.column().classes("gap-0 w-44"):
        with ui.row().classes("w-full justify-between items-baseline"):
            ui.label(label).classes("text-sm")
            number = ui.label(f"{value:g}").classes("text-sm nrk-muted")
        slider = ui.slider(min=minimum, max=maximum, step=step, value=value).props("dense")
        slider.on_value_change(lambda e: number.set_text(f"{e.value:g}"))
    return slider


def effect_editor(kind: str) -> Callable[[], list[dict]]:
    """Effect cards for ``kind`` (image | video); returns a getter for the ordered chain."""
    state = get_state()
    settings = state.settings
    with card():
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                nr_enabled = ui.switch(value=True).props("dense")
                with ui.column().classes("gap-0"):
                    ui.label("Neural rendering").classes("font-medium")
                    ui.label("detail, colour and tone from the recovered transformer").classes("text-sm nrk-muted")
            if not settings.has_nr_weights():
                ui.label("weights not configured — see Settings").classes("text-sm nrk-warn")
        with ui.column().classes("w-full gap-3").bind_visibility_from(nr_enabled, "value"):
            with ui.row().classes("items-center gap-3"):
                ui.label("profile").classes("text-sm")
                profile = ui.toggle(list(PROFILE_NAMES), value="standard").props("dense unelevated no-caps toggle-color=primary")
            with ui.row().classes("gap-6 flex-wrap"):
                scale = _slider("processing scale", value=1.0, minimum=1.0, maximum=4.0, step=0.5)
                detail = _slider("detail", value=1.0, minimum=0.0, maximum=4.0, step=0.1)
                colour = _slider("colour", value=1.0, minimum=0.0, maximum=4.0, step=0.1)
                radius = _slider("detail radius", value=4.0, minimum=1.0, maximum=16.0, step=0.5)
                intensity = _slider("intensity", value=1.0, minimum=0.0, maximum=2.0, step=0.1)
            temporal = None
            if kind == "video":
                temporal = ui.switch("temporal: reproject the previous output and blend (native scale)", value=False).props("dense")
    fg_enabled = fg_mode = fg_factor = fg_audio = order = None
    if kind == "video":
        with card():
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-3"):
                    fg_enabled = ui.switch(value=False).props("dense")
                    with ui.column().classes("gap-0"):
                        ui.label("Frame generation").classes("font-medium")
                        ui.label("DLSS frame generation between consecutive frames").classes("text-sm nrk-muted")
                if not settings.has_fg_weights():
                    ui.label("weights not configured — nrk-weights extract-fg, then Settings").classes("text-sm nrk-warn")
            with ui.column().classes("w-full gap-3").bind_visibility_from(fg_enabled, "value"):
                with ui.row().classes("items-center gap-3 flex-wrap"):
                    ui.label("mode").classes("text-sm")
                    fg_mode = ui.toggle({"fps": "higher frame rate", "slowmo": "slow motion"}, value="fps").props("dense unelevated no-caps toggle-color=primary")
                    ui.label("factor").classes("text-sm ml-4")
                    fg_factor = ui.toggle([2, 3, 4], value=2).props("dense unelevated no-caps toggle-color=primary")
                    ui.label("audio").classes("text-sm ml-4")
                    fg_audio = ui.toggle({"copy": "copy", "stretch": "stretch (slow motion)", "none": "drop"}, value="copy").props("dense unelevated no-caps toggle-color=primary")
                ui.label("higher frame rate keeps the duration and multiplies the rate; slow motion keeps the rate and stretches the clip "
                         "(audio stretched with pitch preserved)").classes("text-sm nrk-muted")
        with card():
            with ui.row().classes("items-center gap-3 flex-wrap"):
                ui.label("order when both are on").classes("text-sm")
                order = ui.toggle({"nr_first": "neural rendering → frame generation", "fg_first": "frame generation → neural rendering"},
                                  value="nr_first").props("dense unelevated no-caps toggle-color=primary")

    def chain() -> list[dict]:
        effects: list[dict] = []
        if nr_enabled.value:
            nr = {"kind": "nr", "profile": profile.value, "processing_scale": float(scale.value), "detail_strength": float(detail.value),
                  "colour_strength": float(colour.value), "detail_radius": float(radius.value), "intensity": float(intensity.value)}
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


def status_chip(state_name: str) -> ui.label:
    return ui.label(state_name).classes(f"nrk-chip nrk-chip-{state_name}")


def job_status_card(job_id: str, *, on_done: Callable[[], None] | None = None):
    """A live card following one job until it finishes."""
    state = get_state()
    with card() as box:
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                chip = status_chip("queued")
                title = ui.label().classes("font-medium")
            with ui.row().classes("items-center gap-2"):
                cancel = ui.button("Cancel", on_click=lambda: state.queue.cancel(job_id)).props("flat dense no-caps color=negative")
                ui.link("all jobs", "/jobs").classes("text-sm")
        stage = ui.label().classes("text-sm nrk-muted")
        bar = ui.linear_progress(value=0.0, show_value=False).props("rounded size=6px color=primary track-color=grey-4").classes("w-full")

    def refresh() -> None:
        job = state.store.get(job_id)
        if job is None:
            timer.deactivate(); return
        title.text = job.input_name
        chip.set_text(job.state); chip.classes(replace=f"nrk-chip nrk-chip-{job.state}")
        stage.text = (job.stage or "") + (f" · {job.seconds:.0f} s" if job.seconds else "")
        bar.value = job.progress
        if job.state in ("done", "failed", "cancelled"):
            cancel.disable(); timer.deactivate()
            if job.state == "failed":
                stage.text = f"failed: {job.error}"
            if on_done is not None:
                on_done()

    timer = ui.timer(0.5, refresh)
    return box
