"""Image page: source and result on the left, controls on the right."""
from __future__ import annotations

import base64
import io

from nicegui import events, ui

from ..state import get_state
from . import ds
from .common import effect_editor, job_status_card, layout


@ui.page("/")
def image_page() -> None:
    state = get_state()
    with layout("Image", "Enhance a still with the neural renderer, then compare before and after."):
        upload_state: dict = {}
        with ui.element("div").classes("nrk-grid"):
            with ui.element("div").classes("nrk-stack"):
                with ds.card("Source"):
                    zone = ds.dropzone(accept="image/*", title="Drop an image here, or click to browse", hint="PNG, JPEG, TIFF, WebP · up to 200 MB",
                                       on_upload=lambda e: on_upload(e), max_size=200_000_000)
                    picked = ui.element("div").classes("w-full").style("display: none")
                results = ui.element("div").classes("nrk-stack")
            with ui.element("div").classes("nrk-stack"):
                chain = effect_editor("image")
                run_button = ds.button("Run", kind="primary", icon="play_arrow", large=True, on_click=lambda: run())

        async def on_upload(e: events.UploadEventArguments) -> None:
            data = await e.file.read()
            upload_state["name"], upload_state["data"] = e.file.name, data
            mime = e.file.content_type or "image/png"
            size = ""
            try:
                from PIL import Image

                with Image.open(io.BytesIO(data)) as im:
                    size = f"{im.width}×{im.height} · "
            except Exception:
                pass
            picked.clear()
            with picked:
                ds.file_row(e.file.name, f"{size}{len(data) / 1e6:.1f} MB", thumbnail=f"data:{mime};base64,{base64.b64encode(data).decode()}")
            picked.style("display: block"); picked.update()
            zone.style("min-height: 96px"); zone.update()

        def run() -> None:
            if "data" not in upload_state:
                ui.notify("Choose an image first.", type="warning"); return
            try:
                job = state.store.create(upload_state["name"], chain(), data=upload_state["data"])
            except ValueError as error:
                ui.notify(str(error), type="negative"); return
            state.queue.submit(job)
            with results:
                holder = ui.element("div").classes("nrk-stack w-full")

                def show() -> None:
                    finished = state.store.get(job.id)
                    if finished is None or finished.state != "done":
                        return
                    with holder:
                        with ds.card("Result") as box:
                            with box.meta:
                                ui.label(f"{finished.seconds:.1f} s · {finished.backend}").classes("nrk-muted nrk-small nrk-mono")
                                ds.button("Download", kind="secondary", icon="download", on_click=lambda: ui.navigate.to(f"/api/jobs/{job.id}/download/0", new_tab=True))
                            ds.wipe_compare(f"/api/jobs/{job.id}/input", f"/api/jobs/{job.id}/download/0")

                job_status_card(job.id, on_done=show)
