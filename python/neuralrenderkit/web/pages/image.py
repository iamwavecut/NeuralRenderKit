"""Image page: drop a picture, choose the neural rendering controls, compare before/after."""
from __future__ import annotations

import base64

from nicegui import events, ui

from ..state import get_state
from .common import card, effect_editor, job_status_card, layout, page_heading


def before_after(before_url: str, after_url: str) -> None:
    """Two stacked images with a wipe slider between them."""
    with ui.element("div").classes("relative w-full select-none rounded-xl overflow-hidden"):
        ui.image(before_url).classes("w-full block")
        after = ui.image(after_url).classes("w-full block absolute top-0 left-0").style("clip-path: inset(0 0 0 50%)")
        line = ui.element("div").classes("absolute top-0 bottom-0 w-px bg-white opacity-90").style("left: 50%")
    slider = ui.slider(min=0, max=100, value=50).classes("w-full").props("color=primary")

    def move() -> None:
        after.style(f"clip-path: inset(0 0 0 {slider.value}%)"); after.update()
        line.style(f"left: {slider.value}%"); line.update()

    slider.on_value_change(lambda _e: move())
    with ui.row().classes("w-full justify-between text-xs nrk-muted"):
        ui.label("original")
        ui.label("neural rendering")


@ui.page("/")
def image_page() -> None:
    state = get_state()
    with layout("Image"):
        page_heading("Still image", "Drop a PNG or JPEG, set the controls and run; the result appears below with a wipe slider.")
        upload_state: dict = {}
        with card():
            ui.label("Input").classes("nrk-section")
            with ui.row().classes("w-full gap-5 items-start"):
                with ui.column().classes("nrk-drop flex-1"):
                    ui.upload(on_upload=lambda e: on_upload(e), auto_upload=True, label="drop an image here or click to choose",
                              max_file_size=200_000_000).props("accept=image/* flat bordered").classes("w-full")
                preview = ui.image().classes("max-h-48 rounded-lg").style("display: none")
                info = ui.label().classes("text-sm nrk-muted")

        async def on_upload(e: events.UploadEventArguments) -> None:
            data = await e.file.read()
            upload_state["name"], upload_state["data"] = e.file.name, data
            mime = e.file.content_type or "image/png"
            preview.set_source(f"data:{mime};base64,{base64.b64encode(data).decode()}")
            preview.style("display: block"); preview.update()
            info.text = f"{e.file.name} · {len(data) / 1e6:.1f} MB"

        chain = effect_editor("image")
        results = ui.column().classes("w-full gap-4")

        def run() -> None:
            if "data" not in upload_state:
                ui.notify("upload an image first", type="warning"); return
            try:
                job = state.store.create(upload_state["name"], chain(), data=upload_state["data"])
            except ValueError as error:
                ui.notify(str(error), type="negative"); return
            state.queue.submit(job)
            with results:
                holder = ui.column().classes("w-full")

                def show() -> None:
                    finished = state.store.get(job.id)
                    if finished is None or finished.state != "done":
                        return
                    with holder, card():
                        ui.label("Result").classes("nrk-section")
                        before_after(f"/api/jobs/{job.id}/input", f"/api/jobs/{job.id}/download/0")
                        with ui.row().classes("items-center gap-4"):
                            ui.button("Download", on_click=lambda: ui.navigate.to(f"/api/jobs/{job.id}/download/0", new_tab=True)).props("unelevated no-caps color=primary")
                            ui.label(f"{finished.seconds:.1f} s · {finished.backend}").classes("text-sm nrk-muted")

                job_status_card(job.id, on_done=show)

        ui.button("Run", icon="play_arrow", on_click=run).props("unelevated no-caps color=primary size=lg").classes("self-start px-6")
