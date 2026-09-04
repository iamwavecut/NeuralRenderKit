"""Settings page: weights, backend, device, folders."""
from __future__ import annotations

from pathlib import Path

from nicegui import ui

from ..state import get_state
from .common import layout


@ui.page("/settings")
def settings_page() -> None:
    state = get_state()
    s = state.settings
    with layout("Settings"):
        ui.label("Settings").classes("text-2xl")
        ui.label("Weights are extracted from your own DLSS library with nrk-weights; nothing is downloaded.").classes("opacity-70")

        def status(path: str) -> str:
            return "✓ found" if path and Path(path).expanduser().exists() else ("· not set" if not path else "✗ missing")

        with ui.card().classes("w-full gap-2"):
            ui.label("Neural rendering").classes("font-medium")
            nr_weights = ui.input("logical safetensors (torch backend) — nrk-weights decode", value=s.nr_weights).classes("w-full")
            nr_status = ui.label(status(s.nr_weights)).classes("text-sm opacity-70")
            nr_model = ui.input(".nrkmodel package (Metal backend, macOS) — nrk-weights mlx", value=s.nr_model).classes("w-full")
            model_status = ui.label(status(s.nr_model)).classes("text-sm opacity-70")
        with ui.card().classes("w-full gap-2"):
            ui.label("Frame generation").classes("font-medium")
            fg_weights = ui.input("dense safetensors — nrk-weights extract-fg libnvidia-ngx-dlssg.so", value=s.fg_weights).classes("w-full")
            fg_status = ui.label(status(s.fg_weights)).classes("text-sm opacity-70")
        with ui.card().classes("w-full gap-2"):
            ui.label("Execution").classes("font-medium")
            with ui.row().classes("gap-4 flex-wrap"):
                backend = ui.select({"auto": "auto (Metal when available)", "torch": "PyTorch", "nrk": "Metal (nrk)"}, value=s.backend, label="backend").classes("w-60")
                device = ui.select(["auto", "cpu", "mps", "cuda"], value=s.device, label="torch device").classes("w-36")
                precision = ui.select(["reference", "fast"], value=s.precision, label="precision").classes("w-36")
                nrk_binary = ui.input("nrk binary (optional)", value=s.nrk_binary).classes("w-72")
            ui.label(f"nrk available: {'yes' if s.nrk_available() else 'no'} · neural rendering runs on {s.resolved_backend('nr')}, frame generation on {s.resolved_backend('fg')}").classes("text-sm opacity-70")
        with ui.card().classes("w-full gap-2"):
            ui.label("Folders").classes("font-medium")
            root = ui.input("root (settings, uploads, outputs)", value=s.root).classes("w-full")
            dark = ui.switch("dark theme (applies on restart)", value=s.theme_dark)

        def save() -> None:
            state.update_settings(nr_weights=nr_weights.value.strip(), nr_model=nr_model.value.strip(), fg_weights=fg_weights.value.strip(),
                                  backend=backend.value, device=device.value, precision=precision.value, nrk_binary=nrk_binary.value.strip(),
                                  root=root.value.strip() or s.root, theme_dark=bool(dark.value))
            nr_status.text = status(nr_weights.value); model_status.text = status(nr_model.value); fg_status.text = status(fg_weights.value)
            ui.notify("saved")

        ui.button("Save", on_click=save).props("color=primary")
