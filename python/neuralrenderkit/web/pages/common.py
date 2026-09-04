"""Page frame (header, navigation, theme) and the effect editor shared by the Image and Video pages."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from nicegui import ui

from ..effects import PROFILE_NAMES
from ..state import get_state
from . import ds

NAV = [("Image", "/"), ("Video", "/video"), ("Jobs", "/jobs"), ("Settings", "/settings")]


@contextmanager
def layout(title: str, lead: str | None = None):
    state = get_state()
    settings = state.settings
    ui.page_title(f"{title} · NeuralRenderKit")
    ui.colors(primary="#4cc2f0" if settings.theme_dark else "#0b6e99", positive="#1f8a4c", negative="#c0392b", warning="#b7791f")
    ds.install()
    dark = ui.dark_mode()
    if settings.theme_dark:
        dark.enable()
    else:
        dark.disable()

    def toggle_theme() -> None:
        dark.toggle()
        state.update_settings(theme_dark=bool(dark.value))
        theme_button.props(f"icon={'light_mode' if dark.value else 'dark_mode'}")

    with ui.header().classes("nrk-header"):
        with ui.element("div").classes("nrk-header-inner"):
            with ui.element("div").classes("flex items-center"):
                with ui.element("a").classes("nrk-brand").props("href=/"):
                    ui.element("span").classes("nrk-glyph").props("innerHTML=N")
                    ui.label("NeuralRenderKit")
                with ui.element("nav").classes("nrk-nav"):
                    for name, path in NAV:
                        ui.link(name, path).classes("active" if name == title else "")
            with ui.element("div").classes("nrk-status"):
                queue_label = ui.label()
                backend = settings.backend + (" · Metal" if settings.nrk_available() else "")
                ui.label(backend)
                theme_button = ui.button(icon="light_mode" if settings.theme_dark else "dark_mode", on_click=toggle_theme).props("flat round dense").classes("nrk-iconbtn")
                theme_button.tooltip("Switch theme")
        rail = ui.element("div").classes("nrk-rail")

    def refresh_status() -> None:
        active = sum(1 for j in state.store.list() if j.state in ("queued", "running"))
        queue_label.set_text(f"{active} running" if active else "idle")
        rail.classes(add="active" if active else "", remove="" if active else "active")

    refresh_status()
    ui.timer(1.0, refresh_status)
    with ui.element("main").classes("nrk-page"):
        if lead is not None:
            ds.page_head(title, lead)
        yield


def effect_editor(kind: str) -> Callable[[], list[dict]]:
    """Effect cards for ``kind`` (image | video); returns a getter for the ordered chain."""
    settings = get_state().settings
    with ds.card("Neural rendering") as box:
        with box.meta:
            nr_enabled = ui.switch(value=True).props("dense color=primary")
        if not settings.has_nr_weights():
            ui.label("Weights are not configured yet — add them in Settings.").classes("nrk-warn nrk-small")
        with ui.element("div").classes("nrk-stack").bind_visibility_from(nr_enabled, "value"):
            profile = ds.segmented_row("Profile", list(PROFILE_NAMES), value="standard")
            scale = ds.slider_row("Processing scale", value=1.0, minimum=1.0, maximum=4.0, step=0.5, hint="Runs the network on the frame resampled by this factor; 2 is the photoreal setting.")
            detail = ds.slider_row("Detail", value=1.0, minimum=0.0, maximum=4.0, step=0.1)
            colour = ds.slider_row("Colour", value=1.0, minimum=0.0, maximum=4.0, step=0.1)
            radius = ds.slider_row("Detail radius", value=4.0, minimum=1.0, maximum=16.0, step=0.5)
            intensity = ds.slider_row("Intensity", value=1.0, minimum=0.0, maximum=2.0, step=0.1)
            temporal = None
            if kind == "video":
                temporal = ds.switch_row("Temporal", "Reproject the previous output into the next frame and blend it (native scale only).", value=False)
    fg_enabled = fg_mode = fg_factor = fg_audio = order = None
    if kind == "video":
        with ds.card("Frame generation") as box:
            with box.meta:
                fg_enabled = ui.switch(value=False).props("dense color=primary")
            if not settings.has_fg_weights():
                ui.label("Weights are not configured yet — run nrk-weights extract-fg and add the file in Settings.").classes("nrk-warn nrk-small")
            with ui.element("div").classes("nrk-stack").bind_visibility_from(fg_enabled, "value"):
                fg_mode = ds.segmented_row("Mode", {"fps": "Higher frame rate", "slowmo": "Slow motion"}, value="fps",
                                           hint="Higher frame rate keeps the duration; slow motion keeps the rate and stretches the clip.")
                fg_factor = ds.segmented_row("Factor", {2: "×2", 3: "×3", 4: "×4"}, value=2)
                fg_audio = ds.segmented_row("Audio", {"copy": "Copy", "stretch": "Stretch", "none": "Drop"}, value="copy",
                                            hint="Stretch keeps the pitch and only applies to slow motion.")
        with ds.card("Order"):
            order = ds.segmented_row("When both are on", {"nr_first": "Render → generate", "fg_first": "Generate → render"}, value="nr_first",
                                     hint="Generating first sends every frame, including generated ones, through the renderer.")

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


def job_status_card(job_id: str, *, on_done: Callable[[], None] | None = None) -> None:
    """A live card following one job until it finishes."""
    state = get_state()
    with ds.card("Job") as box:
        with box.meta:
            status = ds.chip("queued")
        name = ui.label().classes("font-medium")
        stage = ui.label().classes("nrk-muted nrk-small")
        bar = ds.progress()
        with ui.element("div").classes("flex items-center gap-2"):
            cancel = ds.button("Cancel", kind="danger", on_click=lambda: state.queue.cancel(job_id))
            ds.button("All jobs", kind="ghost", on_click=lambda: ui.navigate.to("/jobs"))

    def refresh() -> None:
        job = state.store.get(job_id)
        if job is None:
            timer.deactivate(); return
        name.set_text(job.input_name)
        status.set_text(job.state); status.classes(replace=f"nrk-chip nrk-chip-{job.state}")
        stage.set_text((job.stage or "waiting for the queue") + (f" · {job.seconds:.0f} s" if job.seconds else ""))
        bar.value = job.progress
        if job.state in ("done", "failed", "cancelled"):
            cancel.disable(); timer.deactivate()
            if job.state == "failed":
                stage.set_text(job.error or "failed"); stage.classes(add="nrk-bad")
            if on_done is not None:
                on_done()

    timer = ui.timer(0.5, refresh)
