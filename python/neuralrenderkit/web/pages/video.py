"""Video page: drop a clip, chain neural rendering and frame generation, watch the side-by-side preview."""
from __future__ import annotations

from nicegui import events, ui

from ..state import get_state
from .common import effect_editor, job_status_card, layout


@ui.page("/video")
def video_page() -> None:
    state = get_state()
    with layout("Video"):
        ui.label("Video").classes("text-2xl")
        ui.label("Drop a clip, choose the effects and their order, run. Frame generation doubles/triples/quadruples the frame "
                 "rate or slows the clip down; the preview shows original | result side by side.").classes("opacity-70")
        upload_state: dict = {}
        info = ui.label().classes("text-sm opacity-80")

        async def on_upload(e: events.UploadEventArguments) -> None:
            target = state.settings.uploads / e.file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            await e.file.save(target)
            upload_state["name"], upload_state["path"] = e.file.name, target
            try:
                from ...video import probe

                p = probe(target)
                info.text = f"{e.file.name}: {p.width}x{p.height}, {p.fps:.3f} fps, {p.frame_count} frames, audio {'yes' if p.has_audio else 'no'}"
            except Exception as error:  # ffprobe missing or unreadable file
                info.text = f"{e.file.name}: {error}"

        ui.upload(on_upload=on_upload, auto_upload=True, label="video", max_file_size=8_000_000_000).props("accept=video/*").classes("w-full")
        chain = effect_editor("video")
        results = ui.column().classes("w-full gap-4")

        def run() -> None:
            if "path" not in upload_state:
                ui.notify("upload a video first", type="warning"); return
            try:
                job = state.store.create(upload_state["name"], chain(), source=upload_state["path"])
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
                        if finished.preview:
                            ui.video(f"/api/jobs/{job.id}/preview").classes("w-full rounded")
                        with ui.row().classes("gap-4"):
                            ui.link("download result", f"/api/jobs/{job.id}/download/0", new_tab=True)
                            ui.label(f"{finished.seconds:.0f} s on {finished.backend}").classes("opacity-70 self-center")

                job_status_card(job.id, on_done=show)

        ui.button("Run", on_click=run).props("color=primary")
