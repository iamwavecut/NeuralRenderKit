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
.nrk-header { background: var(--card) !important; color: var(--text) !important; border-bottom: 1px solid var(--line); box-shadow: none !important; padding: 0 !important; }
.nrk-header-inner { max-width: 1120px; margin: 0 auto; padding: 0 28px; height: 56px; display: flex; align-items: center; justify-content: space-between; width: 100%; }
.nrk-brand { display: flex; align-items: center; gap: 10px; font-weight: 600; letter-spacing: -0.01em; font-size: 15px; }
.nrk-brand .nrk-glyph { width: 22px; height: 22px; border-radius: 6px; background: var(--accent); display: grid; place-items: center; color: var(--accent-ink); font-size: 12px; font-weight: 700; }
.nrk-nav { display: flex; align-items: stretch; gap: 4px; height: 56px; margin-left: 36px; }
.nrk-nav a { display: flex; align-items: center; padding: 0 12px; color: var(--muted); text-decoration: none; font-weight: 500; border-bottom: 2px solid transparent; transition: color .15s; }
.nrk-nav a:hover { color: var(--text); }
.nrk-nav a.active { color: var(--text); border-bottom-color: var(--accent); }
.nrk-status { display: flex; align-items: center; gap: 12px; color: var(--muted); font-size: 13px; }
.nrk-rail { position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; overflow: hidden; pointer-events: none; }
.nrk-rail.active::before { content: ""; position: absolute; inset: 0; width: 30%; background: var(--accent); animation: nrk-rail 1.4s ease-in-out infinite; }
@keyframes nrk-rail { 0% { transform: translateX(-100%); } 100% { transform: translateX(340%); } }

/* page */
.nrk-page { max-width: 1120px; margin: 0 auto; padding: 32px 28px 64px; width: 100%; box-sizing: border-box; }
.nrk-page-head { margin-bottom: 24px; }
.nrk-h1 { font-size: 24px; font-weight: 600; letter-spacing: -0.015em; line-height: 1.2; margin: 0; }
.nrk-lead { color: var(--muted); margin: 6px 0 0; font-size: 14px; max-width: 64ch; }
.nrk-grid { display: grid; grid-template-columns: minmax(0, 7fr) minmax(0, 5fr); gap: 24px; align-items: start; }
@media (max-width: 960px) { .nrk-grid { grid-template-columns: 1fr; } }
.nrk-stack { display: flex; flex-direction: column; gap: 16px; min-width: 0; }

/* card */
.nrk-card { background: var(--card); border: 1px solid var(--line); border-radius: var(--r-lg); box-shadow: var(--shadow); overflow: hidden; }
.nrk-card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 20px; border-bottom: 1px solid var(--line); min-height: 48px; }
.nrk-card-title { font-size: 13px; font-weight: 600; letter-spacing: 0.02em; }
.nrk-card-body { padding: 20px; display: flex; flex-direction: column; gap: 16px; }
.nrk-card-body.tight { padding: 0; gap: 0; }
.nrk-card-head:has(+ .nrk-card-body.hidden) { border-bottom: 0; }   /* a switched-off effect card is just its header */

/* text */
.nrk-muted { color: var(--muted); }
.nrk-small { font-size: 13px; }
.nrk-mono { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 12.5px; }
.nrk-ok { color: var(--ok); } .nrk-warn { color: var(--warn); } .nrk-bad { color: var(--bad); }

/* dropzone */
.nrk-drop { position: relative; border: 1.5px dashed var(--line-strong); border-radius: var(--r-md); background: var(--bg); min-height: 176px; display: flex; align-items: center; justify-content: center; text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
.nrk-drop:hover, .nrk-drop:has(.q-uploader--dnd) { border-color: var(--accent); background: var(--accent-soft); }
.nrk-drop-inner { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 24px; pointer-events: none; }
.nrk-drop-inner .q-icon { font-size: 30px; color: var(--accent); }
.nrk-drop-title { font-weight: 500; }
.nrk-drop-hint { color: var(--muted); font-size: 13px; }
.nrk-drop .q-uploader { position: absolute; inset: 0; width: 100%; max-height: none; opacity: 0; box-shadow: none; }
.nrk-drop .q-uploader__list { display: none; }
.nrk-file { display: flex; align-items: center; gap: 12px; padding: 12px 14px; border: 1px solid var(--line); border-radius: var(--r-md); background: var(--card); }
.nrk-file img { width: 64px; height: 44px; object-fit: cover; border-radius: 6px; background: var(--soft); }
.nrk-file .nrk-file-name { font-weight: 500; }
.nrk-preview { width: 100%; border-radius: var(--r-md); overflow: hidden; background: var(--soft); }

/* controls */
.nrk-row { display: grid; grid-template-columns: 148px minmax(0, 1fr) 56px; align-items: center; column-gap: 16px; row-gap: 6px; min-height: 32px; }
.nrk-row-label { color: var(--text); font-size: 13px; }
.nrk-row-control { min-width: 0; display: flex; align-items: center; gap: 12px; }
.nrk-row-value { text-align: right; color: var(--muted); }
.nrk-row-hint { color: var(--muted); font-size: 12.5px; margin: -6px 0 0 164px; }
.nrk-card-body { container-type: inline-size; }
@container (max-width: 440px) {
  .nrk-row { grid-template-columns: minmax(0, 1fr) 56px; }
  .nrk-row-label { grid-column: 1 / -1; }
  .nrk-row-hint { margin-left: 0; }
}
.nrk-seg.q-btn-group { background: var(--soft); border-radius: 10px; padding: 3px; gap: 2px; box-shadow: none; display: inline-flex; flex-wrap: wrap; grid-column: 2 / -1; }
.nrk-seg .q-btn { border-radius: 8px !important; padding: 0 12px; min-height: 28px; font-weight: 500; font-size: 13px; text-transform: none; color: var(--muted); background: transparent !important; }
.nrk-seg .q-btn.bg-primary, .nrk-seg .q-btn[aria-pressed="true"] { background: var(--card) !important; color: var(--text) !important; box-shadow: 0 1px 2px rgba(17,20,24,.12); }
.nrk-seg .q-btn .q-focus-helper { display: none; }
.nrk-slider .q-slider__track-container { padding: 0; }
.nrk-slider .q-slider__track { background: var(--n200) !important; }
.nrk-slider .q-slider__selection { background: var(--accent) !important; }
.nrk-slider .q-slider__thumb { color: var(--accent) !important; }
.nrk-slider .q-slider__thumb-shape path { fill: var(--card); stroke: var(--accent); stroke-width: 2.5; }
.nrk-switch-row { display: flex; align-items: flex-start; gap: 12px; }
.nrk-switch-row .q-toggle { margin: -4px 0 0 -6px; }
.nrk-switch-title { font-weight: 600; }
.nrk-switch-desc { color: var(--muted); font-size: 13px; }
.q-toggle__inner--truthy .q-toggle__track { background: var(--accent) !important; opacity: .35 !important; }
.q-toggle__inner--truthy .q-toggle__thumb:after { background: var(--accent) !important; }
.q-field--outlined .q-field__control { border-radius: var(--r-sm); background: var(--card); }
.q-field--outlined .q-field__control:before { border-color: var(--line-strong); }
.q-field__label, .q-field__native { color: var(--text) !important; }
.q-field--outlined .q-field__control:hover:before { border-color: var(--accent); }

/* buttons */
.nrk-btn { border-radius: 10px !important; text-transform: none !important; font-weight: 600 !important; letter-spacing: 0; padding: 0 16px !important; min-height: 36px !important; font-size: 14px !important; }
.nrk-btn-primary { background: var(--accent) !important; color: var(--accent-ink) !important; }
.nrk-btn-primary:hover { filter: brightness(1.08); }
.nrk-btn-secondary { background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--line-strong) !important; }
.nrk-btn-ghost { background: transparent !important; color: var(--muted) !important; }
.nrk-btn-ghost:hover { color: var(--text) !important; background: var(--soft) !important; }
.nrk-btn-danger { background: transparent !important; color: var(--bad) !important; }
.nrk-btn-danger:hover { background: var(--bad-soft) !important; }
.nrk-btn-lg { min-height: 42px !important; padding: 0 22px !important; font-size: 15px !important; }
.nrk-iconbtn { color: var(--muted) !important; }
.nrk-iconbtn:hover { color: var(--text) !important; }
.q-btn:before { box-shadow: none !important; }

/* chips, progress, lists */
.nrk-chip { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: 0.01em; white-space: nowrap; }
.nrk-chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.nrk-chip-queued { background: var(--soft); color: var(--muted); }
.nrk-chip-running { background: var(--run-soft); color: var(--run); }
.nrk-chip-done { background: var(--ok-soft); color: var(--ok); }
.nrk-chip-failed { background: var(--bad-soft); color: var(--bad); }
.nrk-chip-cancelled { background: var(--soft); color: var(--muted); }
.nrk-progress .q-linear-progress { border-radius: 999px; height: 6px; background: var(--n200) !important; }
.nrk-progress .q-linear-progress__track { opacity: 0 !important; }
.nrk-progress .q-linear-progress__model { background: var(--accent) !important; }
.nrk-list-row { display: grid; grid-template-columns: 96px minmax(0, 1fr) auto; align-items: center; gap: 16px; padding: 14px 20px; border-top: 1px solid var(--line); }
.nrk-list-row > .nrk-chip { justify-self: start; }
.nrk-list-row:first-child { border-top: 0; }
.nrk-list-title { font-weight: 500; }
.nrk-list-meta { color: var(--muted); font-size: 13px; margin-top: 2px; }
.nrk-actions { display: flex; align-items: center; gap: 2px; }
.nrk-empty { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 48px 20px; text-align: center; color: var(--muted); }
.nrk-empty .q-icon { font-size: 32px; color: var(--n300); }

/* settings rows */
.nrk-setting { display: grid; grid-template-columns: 260px minmax(0, 1fr); gap: 24px; align-items: start; padding: 16px 20px; border-top: 1px solid var(--line); }
.nrk-setting:first-child { border-top: 0; }
.nrk-setting-name { font-weight: 500; }
.nrk-setting-desc { color: var(--muted); font-size: 13px; margin-top: 2px; }
@media (max-width: 760px) { .nrk-setting { grid-template-columns: 1fr; gap: 8px; } }

/* before/after */
.nrk-wipe { position: relative; user-select: none; border-radius: var(--r-md); overflow: hidden; background: var(--soft); }
.nrk-wipe img { display: block; width: 100%; }
.nrk-wipe .after { position: absolute; inset: 0; }
.nrk-wipe .divider { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; box-shadow: 0 0 0 1px rgba(0,0,0,.25); transform: translateX(-1px); }
.nrk-wipe .tag { position: absolute; top: 10px; padding: 2px 8px; border-radius: 999px; background: rgba(17,20,24,.65); color: #fff; font-size: 11px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.nrk-wipe .tag.left { left: 10px; } .nrk-wipe .tag.right { right: 10px; }
.nrk-kv { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 13px; }
.nrk-kv b { color: var(--text); font-weight: 500; }
"""


def install() -> None:
    ui.add_head_html(f"<style>{STYLESHEET}</style>")


# ---------------------------------------------------------------- layout blocks
@contextmanager
def card(title: str | None = None, *, tight: bool = False):
    """A bordered card; ``title`` adds a header row whose right side is the ``card_meta`` slot."""
    with ui.element("section").classes("nrk-card w-full") as box:
        if title is not None:
            with ui.element("div").classes("nrk-card-head"):
                ui.label(title).classes("nrk-card-title")
                box.meta = ui.element("div").classes("flex items-center gap-2")  # type: ignore[attr-defined]
        box.body = ui.element("div").classes("nrk-card-body" + (" tight" if tight else ""))
        with box.body:
            yield box


def page_head(title: str, lead: str) -> None:
    with ui.element("div").classes("nrk-page-head"):
        ui.html(f"<h1 class='nrk-h1'>{title}</h1><p class='nrk-lead'>{lead}</p>")


# ---------------------------------------------------------------- controls
def slider_row(label: str, *, value: float, minimum: float, maximum: float, step: float, hint: str | None = None) -> ui.slider:
    """label | slider | value, the value in monospace tabular figures."""
    with ui.element("div").classes("nrk-row"):
        ui.label(label).classes("nrk-row-label")
        with ui.element("div").classes("nrk-row-control nrk-slider"):
            slider = ui.slider(min=minimum, max=maximum, step=step, value=value).props("dense color=primary").classes("w-full")
        readout = ui.label(f"{value:g}").classes("nrk-row-value nrk-mono")
    slider.on_value_change(lambda e: readout.set_text(f"{e.value:g}"))
    if hint:
        ui.label(hint).classes("nrk-row-hint")
    return slider


def segmented_row(label: str, options, *, value, hint: str | None = None) -> ui.toggle:
    with ui.element("div").classes("nrk-row"):
        ui.label(label).classes("nrk-row-label")
        with ui.element("div").classes("nrk-row-control"):
            toggle = ui.toggle(options, value=value).props("unelevated no-caps dense toggle-color=primary").classes("nrk-seg")
    if hint:
        ui.label(hint).classes("nrk-row-hint")
    return toggle


def switch_row(title: str, description: str, *, value: bool) -> ui.switch:
    with ui.element("div").classes("nrk-switch-row"):
        switch = ui.switch(value=value).props("dense color=primary")
        with ui.element("div"):
            ui.label(title).classes("nrk-switch-title")
            ui.label(description).classes("nrk-switch-desc")
    return switch


def button(text: str, *, kind: str = "primary", icon: str | None = None, large: bool = False, on_click: Callable | None = None) -> ui.button:
    b = ui.button(text, icon=icon, on_click=on_click).props("unelevated no-caps")
    b.classes(f"nrk-btn nrk-btn-{kind}" + (" nrk-btn-lg" if large else ""))
    return b


def icon_button(icon: str, *, tooltip: str, on_click: Callable | None = None, danger: bool = False) -> ui.button:
    b = ui.button(icon=icon, on_click=on_click).props("flat round dense").classes("nrk-iconbtn" if not danger else "nrk-btn-danger")
    b.tooltip(tooltip)
    return b


def chip(state: str) -> ui.label:
    return ui.label(state).classes(f"nrk-chip nrk-chip-{state}")


def progress() -> ui.linear_progress:
    with ui.element("div").classes("nrk-progress w-full"):
        return ui.linear_progress(value=0.0, show_value=False).props("rounded")


def dropzone(*, accept: str, title: str, hint: str, on_upload: Callable[[events.UploadEventArguments], object], max_size: int) -> ui.element:
    """A whole-area dropzone: click anywhere to browse, drag anywhere to drop."""
    with ui.element("div").classes("nrk-drop w-full") as zone:
        with ui.element("div").classes("nrk-drop-inner"):
            ui.icon("add_photo_alternate" if accept.startswith("image") else "movie")
            ui.label(title).classes("nrk-drop-title")
            ui.label(hint).classes("nrk-drop-hint")
        uploader = ui.upload(on_upload=on_upload, auto_upload=True, max_file_size=max_size).props(f"accept={accept} flat")
        uploader.on("click", lambda _e: uploader.run_method("pickFiles"))
    return zone


def file_row(name: str, meta: str, *, thumbnail: str | None = None) -> None:
    with ui.element("div").classes("nrk-file"):
        if thumbnail:
            ui.html(f'<img src="{thumbnail}" alt="">')
        else:
            ui.icon("videocam").classes("text-2xl").style("color: var(--accent)")
        with ui.element("div").classes("min-w-0"):
            ui.label(name).classes("nrk-file-name truncate")
            ui.label(meta).classes("nrk-muted nrk-small")


def empty_state(icon: str, text: str) -> None:
    with ui.element("div").classes("nrk-empty"):
        ui.icon(icon)
        ui.label(text)


def result_meta(job) -> None:
    """Right-side header of a result card: time, backend, download."""
    ui.label(f"{job.seconds:.1f} s · {job.backend}").classes("nrk-muted nrk-small nrk-mono")
    button("Download", kind="secondary", icon="download", on_click=lambda: ui.navigate.to(f"/api/jobs/{job.id}/download/0", new_tab=True))


def wipe_compare(before_url: str, after_url: str) -> None:
    """Before/after images with a wipe divider driven by the slider below."""
    with ui.element("div").classes("nrk-wipe"):
        ui.image(before_url)
        after = ui.image(after_url).classes("after").style("clip-path: inset(0 0 0 50%)")
        divider = ui.element("div").classes("divider").style("left: 50%")
        ui.label("before").classes("tag left")
        ui.label("after").classes("tag right")
    with ui.element("div").classes("nrk-slider w-full px-1"):
        slider = ui.slider(min=0, max=100, value=50).props("dense color=primary").classes("w-full")

    def move() -> None:
        after.style(f"clip-path: inset(0 0 0 {slider.value}%)"); after.update()
        divider.style(f"left: {slider.value}%"); divider.update()

    slider.on_value_change(lambda _e: move())
