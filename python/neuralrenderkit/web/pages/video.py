"""Video page: source and preview on the left, the effect chain on the right."""
from __future__ import annotations

from nicegui import events, ui

from ..state import get_state
from . import ds
from .common import effect_editor, job_status_card, layout


def result_card(job) -> None:
    """The converted clip; the side-by-side comparison with the original is a view, not the product."""
    result_url = f"/api/jobs/{job.id}/output/0"
    with ds.card("Result") as box:
        with box.meta:
            if job.preview:
                view = ui.toggle({"result": "Result", "compare": "Side by side"}, value="result").props("unelevated no-caps dense toggle-color=primary").classes("nrk-seg")
            ds.result_meta(job)
        player = ui.video(result_url).classes("nrk-preview")
        note = ui.label(f"{job.input_name} → {job.outputs[0]}").classes("nrk-muted nrk-small")
        if job.preview:
            def switch() -> None:
                compare = view.value == "compare"
                player.set_source(f"/api/jobs/{job.id}/preview" if compare else result_url)
                note.set_text("Original on the left, result on the right; the first 12 seconds." if compare else f"{job.input_name} → {job.outputs[0]}")

            view.on_value_change(lambda _e: switch())


@ui.page("/video")
def video_page(job: str | None = None) -> None:
    """``?job=ID`` opens a finished job's side-by-side preview."""
    state = get_state()
    with layout("Video", "Run a clip through the neural renderer, the frame generator, or both; the result plays here, with the original beside it on request."):
        upload_state: dict = {}
        with ui.element("div").classes("nrk-grid"):
            with ui.element("div").classes("nrk-stack"):
                with ds.card("Source"):
                    zone = ds.dropzone(accept="video/*", title="Drop a video here, or click to browse", hint="MP4, MOV, MKV, WebM · any length",
                                       on_upload=lambda e: on_upload(e), max_size=8_000_000_000)
                    picked = ui.element("div").classes("w-full").style("display: none")
                results = ui.element("div").classes("nrk-stack")
                opened = state.store.get(job) if job else None
                if opened is not None and opened.state == "done" and opened.outputs:
                    with results:
                        result_card(opened)
            with ui.element("div").classes("nrk-stack"):
                chain = effect_editor("video", opened.effects if opened is not None else None)
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
                        result_card(finished)

                job_status_card(job.id, on_done=show)
