"""Settings page: weights, backend, device, folders."""
from __future__ import annotations

from pathlib import Path

from nicegui import ui

from ..state import get_state
from .common import card, layout, page_heading


@ui.page("/settings")
def settings_page() -> None:
    state = get_state()
    s = state.settings
    with layout("Settings"):
        page_heading("Settings", "Weights are extracted from your own DLSS library with nrk-weights; nothing is downloaded.")

        def status(path: str) -> tuple[str, str]:
            if not path:
                return "not set", "nrk-muted"
            return ("found", "nrk-ok") if Path(path).expanduser().exists() else ("missing", "nrk-warn")

        def path_field(label: str, value: str):
            with ui.row().classes("w-full items-center gap-3"):
                field = ui.input(label, value=value).props("outlined dense").classes("flex-1")
                text, cls = status(value)
                mark = ui.label(text).classes(f"text-sm w-16 {cls}")
            return field, mark

        with card():
            ui.label("Neural rendering").classes("nrk-section")
            nr_weights, nr_mark = path_field("logical safetensors (PyTorch) — nrk-weights decode", s.nr_weights)
            nr_model, model_mark = path_field(".nrkmodel package (Metal, macOS) — nrk-weights mlx", s.nr_model)
        with card():
            ui.label("Frame generation").classes("nrk-section")
            fg_weights, fg_mark = path_field("dense safetensors — nrk-weights extract-fg libnvidia-ngx-dlssg.so", s.fg_weights)
        with card():
            ui.label("Execution").classes("nrk-section")
            with ui.row().classes("gap-4 flex-wrap items-center"):
                backend = ui.select({"auto": "auto (Metal when available)", "torch": "PyTorch", "nrk": "Metal (nrk)"}, value=s.backend, label="backend").props("outlined dense").classes("w-64")
                device = ui.select(["auto", "cpu", "mps", "cuda"], value=s.device, label="torch device").props("outlined dense").classes("w-36")
                precision = ui.select(["reference", "fast"], value=s.precision, label="precision").props("outlined dense").classes("w-36")
            nrk_binary, _ = path_field("nrk binary (optional; default: PATH or the repository build)", s.nrk_binary)
            ui.label(f"nrk available: {'yes' if s.nrk_available() else 'no'} · neural rendering runs on {s.resolved_backend('nr')}, frame generation on {s.resolved_backend('fg')}").classes("text-sm nrk-muted")
        with card():
            ui.label("Folders").classes("nrk-section")
            root = ui.input("root (settings, uploads, outputs)", value=s.root).props("outlined dense").classes("w-full")

        def save() -> None:
            state.update_settings(nr_weights=nr_weights.value.strip(), nr_model=nr_model.value.strip(), fg_weights=fg_weights.value.strip(),
                                  backend=backend.value, device=device.value, precision=precision.value, nrk_binary=nrk_binary.value.strip(),
                                  root=root.value.strip() or s.root)
            for field, mark in ((nr_weights, nr_mark), (nr_model, model_mark), (fg_weights, fg_mark)):
                text, cls = status(field.value); mark.set_text(text); mark.classes(replace=f"text-sm w-16 {cls}")
            ui.notify("saved", type="positive")

        ui.button("Save", icon="check", on_click=save).props("unelevated no-caps color=primary").classes("self-start px-6")
