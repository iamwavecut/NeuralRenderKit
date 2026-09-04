"""Video page: source and preview on the left, the effect chain on the right."""
from __future__ import annotations

from nicegui import events, ui

from ..state import get_state
from . import ds
from .common import effect_editor, job_status_card, layout


@ui.page("/video")
def video_page() -> None:
    state = get_state()
    with layout("Video", "Run a clip through the neural renderer, the frame generator, or both, and preview the result next to the original."):
        upload_state: dict = {}
        with ui.element("div").classes("nrk-grid"):
            with ui.element("div").classes("nrk-stack"):
                with ds.card("Source"):
                    zone = ds.dropzone(accept="video/*", title="Drop a video here, or click to browse", hint="MP4, MOV, MKV, WebM · any length",
                                       on_upload=lambda e: on_upload(e), max_size=8_000_000_000)
                    picked = ui.element("div").classes("w-full").style("display: none")
                results = ui.element("div").classes("nrk-stack")
            with ui.element("div").classes("nrk-stack"):
                chain = effect_editor("video")
                ds.button("Run", kind="primary", icon="play_arrow", large=True, on_click=lambda: run())

        async def on_upload(e: events.UploadEventArguments) -> None:
            target = state.settings.uploads / e.file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            await e.file.save(target)
            upload_state["name"], upload_state["path"] = e.file.name, target
            meta = f"{target.stat().st_size / 1e6:.1f} MB"
            try:
                from ...video import probe

                p = probe(target)
                meta = f"{p.width}×{p.height} · {p.fps:.3g} fps · {p.frame_count} frames · {'audio' if p.has_audio else 'no audio'} · {meta}"
            except Exception as error:  # ffprobe missing or unreadable file
                meta = f"{meta} · {error}"
            picked.clear()
            with picked:
                ds.file_row(e.file.name, meta)
            picked.style("display: block"); picked.update()
            zone.style("min-height: 96px"); zone.update()

        def run() -> None:
            if "path" not in upload_state:
                ui.notify("Choose a video first.", type="warning"); return
            try:
                job = state.store.create(upload_state["name"], chain(), source=upload_state["path"])
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
                                ui.label(f"{finished.seconds:.0f} s · {finished.backend}").classes("nrk-muted nrk-small nrk-mono")
                                ds.button("Download", kind="secondary", icon="download", on_click=lambda: ui.navigate.to(f"/api/jobs/{job.id}/download/0", new_tab=True))
                            if finished.preview:
                                ui.video(f"/api/jobs/{job.id}/preview").classes("nrk-preview")
                                ui.label("Original on the left, result on the right; the first 12 seconds.").classes("nrk-muted nrk-small")

                job_status_card(job.id, on_done=show)
