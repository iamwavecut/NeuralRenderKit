"""HTTP API on the NiceGUI app: jobs, effects, previews, downloads.

    GET  /api/effects                what the backend offers right now
    POST /api/jobs                   multipart: file + effects (JSON string) -> job
    GET  /api/jobs, /api/jobs/{id}   states and progress
    POST /api/jobs/{id}/cancel
    GET  /api/jobs/{id}/preview      side-by-side preview (jpg or mp4)
    GET  /api/jobs/{id}/download/{n} the n-th output file
"""
from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .effects import describe_effects
from .state import WebState


def build_router(state: WebState) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/effects")
    def effects() -> dict:
        s = state.settings
        return describe_effects(nrk_available=s.nrk_available(), fg_weights=s.has_fg_weights(), nr_weights=s.has_nr_weights())

    @router.get("/jobs")
    def jobs() -> list[dict]:
        return [j.to_dict() for j in state.store.list()]

    @router.get("/jobs/{job_id}")
    def job(job_id: str) -> dict:
        j = state.store.get(job_id)
        if j is None:
            raise HTTPException(404, "no such job")
        return j.to_dict()

    @router.post("/jobs")
    async def create(file: UploadFile = File(...), effects: str = Form(...)) -> JSONResponse:
        try:
            chain = json.loads(effects)
            if not isinstance(chain, list):
                raise ValueError("effects must be a JSON list")
            data = await file.read()
            j = state.store.create(file.filename or "upload.bin", chain, data=data)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        state.queue.submit(j)
        return JSONResponse(j.to_dict(), status_code=201)

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        if not state.queue.cancel(job_id):
            raise HTTPException(404, "no such job")
        return state.store.get(job_id).to_dict()

    @router.delete("/jobs/{job_id}")
    def delete(job_id: str) -> dict:
        j = state.store.get(job_id)
        if j is None:
            raise HTTPException(404, "no such job")
        if j.state in ("queued", "running"):
            raise HTTPException(409, "cancel the job first")
        state.store.delete(job_id)
        return {"deleted": job_id}

    @router.get("/jobs/{job_id}/input")
    def input_file(job_id: str) -> FileResponse:
        j = state.store.get(job_id)
        if j is None:
            raise HTTPException(404, "no such job")
        return FileResponse(state.store.input_path(j))

    @router.get("/jobs/{job_id}/preview")
    def preview(job_id: str) -> FileResponse:
        j = state.store.get(job_id)
        if j is None or not j.preview:
            raise HTTPException(404, "no preview")
        return FileResponse(state.store.folder(job_id) / j.preview)

    @router.get("/jobs/{job_id}/download/{index}")
    def download(job_id: str, index: int) -> FileResponse:
        j = state.store.get(job_id)
        if j is None or index < 0 or index >= len(j.outputs):
            raise HTTPException(404, "no such output")
        name = j.outputs[index]
        return FileResponse(state.store.folder(job_id) / name, filename=f"{j.id}-{name}")

    return router
