"""Settings page: weights, execution, folders — a preferences pane with a description per row."""
from __future__ import annotations

from pathlib import Path

from nicegui import ui

from ..state import get_state
from . import ds
from .common import layout


@ui.page("/settings")
def settings_page() -> None:
    state = get_state()
    s = state.settings
    with layout("Settings", "Weights are extracted from your own DLSS library with nrk-weights; nothing is downloaded."):
        marks: list[tuple[ui.input, ui.label]] = []

        def status(path: str) -> tuple[str, str]:
            if not path:
                return "not set", "nrk-muted"
            return ("found", "nrk-ok") if Path(path).expanduser().exists() else ("missing", "nrk-warn")

        def setting(name: str, description: str):
            row = ui.element("div").classes("nrk-setting")
            with row:
                with ui.element("div"):
                    ui.label(name).classes("nrk-setting-name")
                    ui.label(description).classes("nrk-setting-desc")
                control = ui.element("div").classes("min-w-0 flex items-center gap-3")
            return control

        def path_setting(name: str, description: str, value: str) -> ui.input:
            with setting(name, description):
                field = ui.input(value=value).props("outlined dense").classes("flex-1")
                text, cls = status(value)
                marks.append((field, ui.label(text).classes(f"nrk-small w-16 {cls}")))
            return field

        with ds.card("Neural rendering", tight=True):
            nr_weights = path_setting("PyTorch weights", "Logical safetensors from nrk-weights decode.", s.nr_weights)
            nr_model = path_setting("Metal package", ".nrkmodel from nrk-weights mlx (macOS).", s.nr_model)
        with ds.card("Frame generation", tight=True):
            fg_weights = path_setting("Weights", "Dense safetensors from nrk-weights extract-fg.", s.fg_weights)
        with ds.card("Execution", tight=True):
            with setting("Backend", "Auto picks Metal when the nrk binary is available."):
                backend = ui.toggle({"auto": "Auto", "torch": "PyTorch", "nrk": "Metal"}, value=s.backend).props("unelevated no-caps dense toggle-color=primary").classes("nrk-seg")
            with setting("PyTorch device", "Auto picks CUDA or MPS when present."):
                device = ui.toggle(["auto", "cpu", "mps", "cuda"], value=s.device).props("unelevated no-caps dense toggle-color=primary").classes("nrk-seg")
            with setting("Precision", "Reference keeps float32; fast uses float16 on GPUs."):
                precision = ui.toggle({"reference": "Reference", "fast": "Fast"}, value=s.precision).props("unelevated no-caps dense toggle-color=primary").classes("nrk-seg")
            nrk_binary = path_setting("nrk binary", "Leave empty to use PATH or the repository build.", s.nrk_binary)
            with setting("Now", ""):
                ui.label(f"nrk {'available' if s.nrk_available() else 'not found'} · neural rendering on {s.resolved_backend('nr')} · frame generation on {s.resolved_backend('fg')}").classes("nrk-muted nrk-small")
        with ds.card("Folders", tight=True):
            with setting("Root", "Settings, uploads and job outputs."):
                root = ui.input(value=s.root).props("outlined dense").classes("flex-1")

        def save() -> None:
            state.update_settings(nr_weights=nr_weights.value.strip(), nr_model=nr_model.value.strip(), fg_weights=fg_weights.value.strip(),
                                  backend=backend.value, device=device.value, precision=precision.value, nrk_binary=nrk_binary.value.strip(),
                                  root=root.value.strip() or s.root)
            for field, mark in marks:
                text, cls = status(field.value); mark.set_text(text); mark.classes(replace=f"nrk-small w-16 {cls}")
            ui.notify("Settings saved", type="positive")

        ds.button("Save settings", kind="primary", icon="check", on_click=save)
