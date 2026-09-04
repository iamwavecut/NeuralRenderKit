"""Image page: drop a picture, choose the neural rendering controls, compare before/after."""
from __future__ import annotations

import base64

from nicegui import events, ui

from ..state import get_state
from .common import effect_editor, job_status_card, layout


def before_after(before_url: str, after_url: str) -> None:
    """Two stacked images with a slider that wipes between them."""
    with ui.element("div").classes("relative w-full select-none") as box:
        ui.image(before_url).classes("w-full block")
        after = ui.image(after_url).classes("w-full block absolute top-0 left-0").style("clip-path: inset(0 0 0 50%)")
        ui.element("div").classes("absolute top-0 bottom-0 w-px bg-white opacity-80").style("left: 50%") .props("id=wipe")
    slider = ui.slider(min=0, max=100, value=50).classes("w-full")

    def move() -> None:
        after.style(f"clip-path: inset(0 0 0 {slider.value}%)")
        after.update()

    slider.on_value_change(lambda _e: move())
    with ui.row().classes("w-full justify-between text-xs opacity-70"):
        ui.label("original")
        ui.label("neural rendering")
    return box


@ui.page("/")
def image_page() -> None:
    state = get_state()
    with layout("Image"):
        ui.label("Still image").classes("text-2xl")
        ui.label("Drop a PNG/JPEG, pick the controls, run. The result appears below with a wipe slider.").classes("opacity-70")
        upload_state: dict = {}
        preview = ui.image().classes("max-h-64 rounded").style("display: none")

        async def on_upload(e: events.UploadEventArguments) -> None:
            data = await e.file.read()
            upload_state["name"], upload_state["data"] = e.file.name, data
            mime = e.file.content_type or "image/png"
            preview.set_source(f"data:{mime};base64,{base64.b64encode(data).decode()}")
            preview.style("display: block"); preview.update()
            ui.notify(f"{e.file.name}: {len(data) / 1e6:.1f} MB")

        ui.upload(on_upload=on_upload, auto_upload=True, label="image", max_file_size=200_000_000).props("accept=image/*").classes("w-full")
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
                    with holder:
                        before_after(f"/api/jobs/{job.id}/input", f"/api/jobs/{job.id}/download/0")
                        with ui.row().classes("gap-4"):
                            ui.link("download result", f"/api/jobs/{job.id}/download/0", new_tab=True)
                            ui.label(f"{finished.seconds:.1f} s on {finished.backend}").classes("opacity-70 self-center")

                job_status_card(job.id, on_done=show)

        ui.button("Run", on_click=run).props("color=primary")
