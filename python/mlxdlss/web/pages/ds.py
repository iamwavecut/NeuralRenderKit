"""The front end's design system: tokens, stylesheet and the components every page is built from.

Tokens live in CSS custom properties on ``body`` (dark overrides on ``body.body--dark``, on the same
element so derived tokens recompute);
components are thin NiceGUI wrappers that apply the classes, so a page never
touches raw Quasar styling.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from nicegui import events, ui

STYLESHEET = """
body {
  --n0: #ffffff; --n50: #f4f5f7; --n100: #e9ecf0; --n200: #dde1e6; --n300: #c5cbd3; --n500: #6b7480; --n700: #3a414b; --n900: #111418;
  --bg: var(--n50); --card: var(--n0); --line: var(--n200); --line-strong: var(--n300); --text: var(--n900); --muted: var(--n500); --soft: var(--n100);
  --accent: #0b6e99; --accent-ink: #ffffff; --accent-soft: #e3f1f8;
  --ok: #1f8a4c; --ok-soft: #e4f4ea; --warn: #b7791f; --warn-soft: #fbf1dc; --bad: #c0392b; --bad-soft: #fbe6e3; --run: #0b6e99; --run-soft: #e3f1f8;
  --r-sm: 8px; --r-md: 12px; --r-lg: 16px; --shadow: 0 1px 2px rgba(17,20,24,.06);
  --font: -apple-system, "SF Pro Text", "Helvetica Neue", Inter, Segoe UI, Roboto, sans-serif;
  --mono: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
}
body.body--dark {
  --n0: #16191e; --n50: #0e1013; --n100: #1d2127; --n200: #262b33; --n300: #343b45; --n500: #8a94a3; --n700: #c4cad3; --n900: #e6e8eb;
  --accent: #4cc2f0; --accent-ink: #0b1a22; --accent-soft: #12303d;
  --ok: #4ade80; --ok-soft: #12301f; --warn: #fbbf24; --warn-soft: #3a2c10; --bad: #f87171; --bad-soft: #3d1a1a; --run: #4cc2f0; --run-soft: #12303d;
  --shadow: none;
}
html, body { background: var(--bg) !important; color: var(--text); font-family: var(--font); font-size: 14px; line-height: 1.45; -webkit-font-smoothing: antialiased; }
.nicegui-content { padding: 0 !important; }
*:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }

/* header */
.mlxdlss-header { background: var(--card) !important; color: var(--text) !important; border-bottom: 1px solid var(--line); box-shadow: none !important; padding: 0 !important; }
.mlxdlss-header-inner { max-width: 1120px; margin: 0 auto; padding: 0 28px; height: 56px; display: flex; align-items: center; justify-content: space-between; width: 100%; }
.mlxdlss-brand { display: flex; align-items: center; gap: 10px; font-weight: 600; letter-spacing: -0.01em; font-size: 15px; }
.mlxdlss-brand .mlxdlss-glyph { width: 22px; height: 22px; border-radius: 6px; background: var(--accent); display: grid; place-items: center; color: var(--accent-ink); font-size: 12px; font-weight: 700; }
.mlxdlss-nav { display: flex; align-items: stretch; gap: 4px; height: 56px; margin-left: 36px; }
.mlxdlss-nav a { display: flex; align-items: center; padding: 0 12px; color: var(--muted); text-decoration: none; font-weight: 500; border-bottom: 2px solid transparent; transition: color .15s; }
.mlxdlss-nav a:hover { color: var(--text); }
.mlxdlss-nav a.active { color: var(--text); border-bottom-color: var(--accent); }
.mlxdlss-status { display: flex; align-items: center; gap: 12px; color: var(--muted); font-size: 13px; }
.mlxdlss-rail { position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; overflow: hidden; pointer-events: none; }
.mlxdlss-rail.active::before { content: ""; position: absolute; inset: 0; width: 30%; background: var(--accent); animation: mlxdlss-rail 1.4s ease-in-out infinite; }
@keyframes mlxdlss-rail { 0% { transform: translateX(-100%); } 100% { transform: translateX(340%); } }

/* page */
.mlxdlss-page { max-width: 1120px; margin: 0 auto; padding: 32px 28px 64px; width: 100%; box-sizing: border-box; }
.mlxdlss-page-head { margin-bottom: 24px; }
.mlxdlss-h1 { font-size: 24px; font-weight: 600; letter-spacing: -0.015em; line-height: 1.2; margin: 0; }
.mlxdlss-lead { color: var(--muted); margin: 6px 0 0; font-size: 14px; max-width: 64ch; }
.mlxdlss-grid { display: grid; grid-template-columns: minmax(0, 7fr) minmax(0, 5fr); gap: 24px; align-items: start; }
@media (max-width: 960px) { .mlxdlss-grid { grid-template-columns: 1fr; } }
.mlxdlss-stack { display: flex; flex-direction: column; gap: 16px; min-width: 0; }

/* card */
.mlxdlss-card { background: var(--card); border: 1px solid var(--line); border-radius: var(--r-lg); box-shadow: var(--shadow); overflow: hidden; }
.mlxdlss-card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 20px; border-bottom: 1px solid var(--line); min-height: 48px; }
.mlxdlss-card-title { font-size: 13px; font-weight: 600; letter-spacing: 0.02em; }
.mlxdlss-card-body { padding: 20px; display: flex; flex-direction: column; gap: 16px; }
.mlxdlss-card-body.tight { padding: 0; gap: 0; }
.mlxdlss-card-head:has(+ .mlxdlss-card-body.hidden) { border-bottom: 0; }   /* a switched-off effect card is just its header */

/* text */
.mlxdlss-muted { color: var(--muted); }
.mlxdlss-small { font-size: 13px; }
.mlxdlss-mono { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 12.5px; }
.mlxdlss-ok { color: var(--ok); } .mlxdlss-warn { color: var(--warn); } .mlxdlss-bad { color: var(--bad); }

/* dropzone */
.mlxdlss-drop { position: relative; border: 1.5px dashed var(--line-strong); border-radius: var(--r-md); background: var(--bg); min-height: 176px; display: flex; align-items: center; justify-content: center; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
.mlxdlss-drop:hover, .mlxdlss-drop:has(.q-uploader--dnd) { border-color: var(--accent); background: var(--accent-soft); }
.mlxdlss-drop-inner { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 24px; pointer-events: none; }
.mlxdlss-drop-inner .q-icon { font-size: 30px; color: var(--accent); }
.mlxdlss-drop-title { font-weight: 500; }
.mlxdlss-drop-hint { color: var(--muted); font-size: 13px; }
.mlxdlss-drop .q-uploader { position: absolute; inset: 0; width: 100%; max-height: none; opacity: 0; box-shadow: none; }
.mlxdlss-drop .q-uploader__list { display: none; }
.mlxdlss-file { display: flex; align-items: center; gap: 12px; padding: 12px 14px; border: 1px solid var(--line); border-radius: var(--r-md); background: var(--card); }
.mlxdlss-file img { width: 64px; height: 44px; object-fit: cover; border-radius: 6px; background: var(--soft); }
.mlxdlss-file .mlxdlss-file-name { font-weight: 500; }
.mlxdlss-preview { width: 100%; border-radius: var(--r-md); overflow: hidden; background: var(--soft); }

/* controls */
.mlxdlss-row { display: grid; grid-template-columns: 148px minmax(0, 1fr) 56px; align-items: center; column-gap: 16px; row-gap: 6px; min-height: 32px; }
.mlxdlss-row-label { color: var(--text); font-size: 13px; }
.mlxdlss-row-control { min-width: 0; display: flex; align-items: center; gap: 12px; }
.mlxdlss-row-value { text-align: right; color: var(--muted); }
.mlxdlss-row-hint { color: var(--muted); font-size: 12.5px; margin: -6px 0 0 164px; }
.mlxdlss-card-body { container-type: inline-size; }
@container (max-width: 440px) {
  .mlxdlss-row { grid-template-columns: minmax(0, 1fr) 56px; }
  .mlxdlss-row-label { grid-column: 1 / -1; }
  .mlxdlss-row-hint { margin-left: 0; }
}
.mlxdlss-seg.q-btn-group { background: var(--soft); border-radius: 10px; padding: 3px; gap: 2px; box-shadow: none; display: inline-flex; flex-wrap: wrap; grid-column: 2 / -1; }
.mlxdlss-seg .q-btn { border-radius: 8px !important; padding: 0 12px; min-height: 28px; font-weight: 500; font-size: 13px; text-transform: none; color: var(--muted); background: transparent !important; }
.mlxdlss-seg .q-btn.bg-primary, .mlxdlss-seg .q-btn[aria-pressed="true"] { background: var(--card) !important; color: var(--text) !important; box-shadow: 0 1px 2px rgba(17,20,24,.12); }
.mlxdlss-seg .q-btn .q-focus-helper { display: none; }
.mlxdlss-slider .q-slider__track-container { padding: 0; }
.mlxdlss-slider .q-slider__track { background: var(--n200) !important; }
.mlxdlss-slider .q-slider__selection { background: var(--accent) !important; }
.mlxdlss-slider .q-slider__thumb { color: var(--accent) !important; }
.mlxdlss-slider .q-slider__thumb-shape path { fill: var(--card); stroke: var(--accent); stroke-width: 2.5; }
.mlxdlss-switch-row { display: flex; align-items: flex-start; gap: 12px; }
.mlxdlss-switch-row .q-toggle { margin: -4px 0 0 -6px; }
.mlxdlss-switch-title { font-weight: 600; }
.mlxdlss-switch-desc { color: var(--muted); font-size: 13px; }
.q-toggle__inner--truthy .q-toggle__track { background: var(--accent) !important; opacity: .35 !important; }
.q-toggle__inner--truthy .q-toggle__thumb:after { background: var(--accent) !important; }
.q-field--outlined .q-field__control { border-radius: var(--r-sm); background: var(--card); }
.q-field--outlined .q-field__control:before { border-color: var(--line-strong); }
.q-field__label, .q-field__native { color: var(--text) !important; }
.q-field--outlined .q-field__control:hover:before { border-color: var(--accent); }

/* buttons */
.mlxdlss-btn { border-radius: 10px !important; text-transform: none !important; font-weight: 600 !important; letter-spacing: 0; padding: 0 16px !important; min-height: 36px !important; font-size: 14px !important; }
.mlxdlss-btn-primary { background: var(--accent) !important; color: var(--accent-ink) !important; }
.mlxdlss-btn-primary:hover { filter: brightness(1.08); }
.mlxdlss-btn-secondary { background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--line-strong) !important; }
.mlxdlss-btn-ghost { background: transparent !important; color: var(--muted) !important; }
.mlxdlss-btn-ghost:hover { color: var(--text) !important; background: var(--soft) !important; }
.mlxdlss-btn-danger { background: transparent !important; color: var(--bad) !important; }
.mlxdlss-btn-danger:hover { background: var(--bad-soft) !important; }
.mlxdlss-btn-lg { min-height: 42px !important; padding: 0 22px !important; font-size: 15px !important; }
.mlxdlss-iconbtn { color: var(--muted) !important; }
.mlxdlss-iconbtn:hover { color: var(--text) !important; }
.q-btn:before { box-shadow: none !important; }

/* chips, progress, lists */
.mlxdlss-chip { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: 0.01em; white-space: nowrap; }
.mlxdlss-chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.mlxdlss-chip-queued { background: var(--soft); color: var(--muted); }
.mlxdlss-chip-running { background: var(--run-soft); color: var(--run); }
.mlxdlss-chip-done { background: var(--ok-soft); color: var(--ok); }
.mlxdlss-chip-failed { background: var(--bad-soft); color: var(--bad); }
.mlxdlss-chip-cancelled { background: var(--soft); color: var(--muted); }
.mlxdlss-progress .q-linear-progress { border-radius: 999px; height: 6px; background: var(--n200) !important; }
.mlxdlss-progress .q-linear-progress__track { opacity: 0 !important; }
.mlxdlss-progress .q-linear-progress__model { background: var(--accent) !important; }
.mlxdlss-list-row { display: grid; grid-template-columns: 96px minmax(0, 1fr) auto; align-items: center; gap: 16px; padding: 14px 20px; border-top: 1px solid var(--line); }
.mlxdlss-list-row > .mlxdlss-chip { justify-self: start; }
.mlxdlss-list-row:first-child { border-top: 0; }
.mlxdlss-list-title { font-weight: 500; }
.mlxdlss-list-meta { color: var(--muted); font-size: 13px; margin-top: 2px; }
.mlxdlss-actions { display: flex; align-items: center; gap: 2px; }
.mlxdlss-empty { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 48px 20px; text-align: center; color: var(--muted); }
.mlxdlss-empty .q-icon { font-size: 32px; color: var(--n300); }

/* settings rows */
.mlxdlss-setting { display: grid; grid-template-columns: 260px minmax(0, 1fr); gap: 24px; align-items: start; padding: 16px 20px; border-top: 1px solid var(--line); }
.mlxdlss-setting:first-child { border-top: 0; }
.mlxdlss-setting-name { font-weight: 500; }
.mlxdlss-setting-desc { color: var(--muted); font-size: 13px; margin-top: 2px; }
@media (max-width: 760px) { .mlxdlss-setting { grid-template-columns: 1fr; gap: 8px; } }

/* before/after */
.mlxdlss-wipe { position: relative; user-select: none; border-radius: var(--r-md); overflow: hidden; background: var(--soft); }
.mlxdlss-wipe img { display: block; width: 100%; }
.mlxdlss-wipe .after { position: absolute; inset: 0; }
.mlxdlss-wipe .divider { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.25); transform: translateX(-1px); }
.mlxdlss-wipe .tag { position: absolute; top: 10px; padding: 2px 8px; border-radius: 999px; background: rgba(17,20,24,.65); color: #fff; font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.mlxdlss-wipe .tag.left { left: 10px; } .mlxdlss-wipe .tag.right { right: 10px; }
.mlxdlss-kv { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 13px; }
.mlxdlss-kv b { color: var(--text); font-weight: 500; }
"""


def install() -> None:
    ui.add_head_html(f"<style>{STYLESHEET}</style>")


# ---------------------------------------------------------------- layout blocks
@contextmanager
def card(title: str | None = None, *, tight: bool = False):
    """A bordered card; ``title`` adds a header row whose right side is the ``card_meta`` slot."""
    with ui.element("section").classes("mlxdlss-card w-full") as box:
        if title is not None:
            with ui.element("div").classes("mlxdlss-card-head"):
                ui.label(title).classes("mlxdlss-card-title")
                box.meta = ui.element("div").classes("flex items-center gap-2")  # type: ignore[attr-defined]
        box.body = ui.element("div").classes("mlxdlss-card-body" + (" tight" if tight else ""))
        with box.body:
            yield box


def page_head(title: str, lead: str) -> None:
    with ui.element("div").classes("mlxdlss-page-head"):
        ui.html(f"<h1 class='mlxdlss-h1'>{title}</h1><p class='mlxdlss-lead'>{lead}</p>")


# ---------------------------------------------------------------- controls
def slider_row(label: str, *, value: float, minimum: float, maximum: float, step: float, hint: str | None = None) -> ui.slider:
    """label | slider | value, the value in monospace tabular figures."""
    with ui.element("div").classes("mlxdlss-row"):
        ui.label(label).classes("mlxdlss-row-label")
        with ui.element("div").classes("mlxdlss-row-control mlxdlss-slider"):
            slider = ui.slider(min=minimum, max=maximum, step=step, value=value).props("dense color=primary").classes("w-full")
        readout = ui.label(f"{value:g}").classes("mlxdlss-row-value mlxdlss-mono")
    slider.on_value_change(lambda e: readout.set_text(f"{e.value:g}"))
    if hint:
        ui.label(hint).classes("mlxdlss-row-hint")
    return slider


def segmented_row(label: str, options, *, value, hint: str | None = None) -> ui.toggle:
    with ui.element("div").classes("mlxdlss-row"):
        ui.label(label).classes("mlxdlss-row-label")
        with ui.element("div").classes("mlxdlss-row-control"):
            toggle = ui.toggle(options, value=value).props("unelevated no-caps dense toggle-color=primary").classes("mlxdlss-seg")
    if hint:
        ui.label(hint).classes("mlxdlss-row-hint")
    return toggle


def switch_row(title: str, description: str, *, value: bool) -> ui.switch:
    with ui.element("div").classes("mlxdlss-switch-row"):
        switch = ui.switch(value=value).props("dense color=primary")
        with ui.element("div"):
            ui.label(title).classes("mlxdlss-switch-title")
            ui.label(description).classes("mlxdlss-switch-desc")
    return switch


def button(text: str, *, kind: str = "primary", icon: str | None = None, large: bool = False, on_click: Callable | None = None) -> ui.button:
    b = ui.button(text, icon=icon, on_click=on_click).props("unelevated no-caps")
    b.classes(f"mlxdlss-btn mlxdlss-btn-{kind}" + (" mlxdlss-btn-lg" if large else ""))
    return b


def icon_button(icon: str, *, tooltip: str, on_click: Callable | None = None, danger: bool = False) -> ui.button:
    b = ui.button(icon=icon, on_click=on_click).props("flat round dense").classes("mlxdlss-iconbtn" if not danger else "mlxdlss-btn-danger")
    b.tooltip(tooltip)
    return b


def chip(state: str) -> ui.label:
    return ui.label(state).classes(f"mlxdlss-chip mlxdlss-chip-{state}")


def progress() -> ui.linear_progress:
    with ui.element("div").classes("mlxdlss-progress w-full"):
        return ui.linear_progress(value=0.0, show_value=False).props("rounded")


def dropzone(*, accept: str, title: str, hint: str, on_upload: Callable[[events.UploadEventArguments], object], max_size: int) -> ui.element:
    """A whole-area dropzone: click anywhere to browse, drag anywhere to drop."""
    with ui.element("div").classes("mlxdlss-drop w-full") as zone:
        with ui.element("div").classes("mlxdlss-drop-inner"):
            ui.icon("add_photo_alternate" if accept.startswith("image") else "movie")
            ui.label(title).classes("mlxdlss-drop-title")
            ui.label(hint).classes("mlxdlss-drop-hint")
        uploader = ui.upload(on_upload=on_upload, auto_upload=True, max_file_size=max_size).props(f"accept={accept} flat")
        uploader.on("click", lambda _e: uploader.run_method("pickFiles"))
    return zone


def file_row(name: str, meta: str, *, thumbnail: str | None = None) -> None:
    with ui.element("div").classes("mlxdlss-file"):
        if thumbnail:
            ui.html(f'<img src="{thumbnail}" alt="">')
        else:
            ui.icon("videocam").classes("text-2xl").style("color: var(--accent)")
        with ui.element("div").classes("min-w-0"):
            ui.label(name).classes("mlxdlss-file-name truncate")
            ui.label(meta).classes("mlxdlss-muted mlxdlss-small")


def empty_state(icon: str, text: str) -> None:
    with ui.element("div").classes("mlxdlss-empty"):
        ui.icon(icon)
        ui.label(text)


def result_meta(job) -> None:
    """Right-side header of a result card: time, backend, download."""
    ui.label(f"{job.seconds:.1f} s · {job.backend}").classes("mlxdlss-muted mlxdlss-small mlxdlss-mono")
    button("Download", kind="secondary", icon="download", on_click=lambda: ui.navigate.to(f"/api/jobs/{job.id}/download/0", new_tab=True))


def wipe_compare(before_url: str, after_url: str) -> None:
    """Before/after images with a wipe divider driven by the slider below."""
    with ui.element("div").classes("mlxdlss-wipe"):
        ui.image(before_url)
        after = ui.image(after_url).classes("after").style("clip-path: inset(0 0 0 50%)")
        divider = ui.element("div").classes("divider").style("left: 50%")
        ui.label("before").classes("tag left")
        ui.label("after").classes("tag right")
    with ui.element("div").classes("mlxdlss-slider w-full px-1"):
        slider = ui.slider(min=0, max=100, value=50).props("dense color=primary").classes("w-full")

    def move() -> None:
        after.style(f"clip-path: inset(0 0 0 {slider.value}%)"); after.update()
        divider.style(f"left: {slider.value}%"); divider.update()

    slider.on_value_change(lambda _e: move())
