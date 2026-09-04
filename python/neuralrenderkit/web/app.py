"""``nrk-web``: start the local front end.

    nrk-web [--port 8181] [--host 127.0.0.1] [--native] [--no-browser] [--root DIR]

The page opens in the default browser (``--native`` opens a window instead, which
needs ``pip install pywebview``). Weights are configured on the Settings page or
through ``NRK_NR_WEIGHTS``, ``NRK_NR_MODEL`` and ``NRK_FG_WEIGHTS``; jobs and
results live under ``~/NeuralRenderKit`` (``--root`` / ``NRK_WEB_ROOT``).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import state as state_module
from .settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nrk-web", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8181)
    parser.add_argument("--native", action="store_true", help="open a native window (pywebview) instead of the browser")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--root", type=Path, default=None, help="folder for settings, jobs and results (default ~/NeuralRenderKit)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("warning: ffmpeg/ffprobe not found in PATH; video jobs will fail until they are installed", file=sys.stderr)
    try:
        from nicegui import app, ui
    except ImportError:
        print("error: the web front end needs `pip install 'neuralrenderkit[web]'` (nicegui, pydantic)", file=sys.stderr)
        return 2
    settings = Settings.load(args.root)
    state_module.STATE = state_module.WebState(settings)
    from .api import build_router
    from .pages import register_pages

    app.include_router(build_router(state_module.STATE))
    register_pages()
    ui.run(
        host=args.host, port=args.port, title="NeuralRenderKit", dark=settings.theme_dark, reload=False,
        show=not args.no_browser and not args.native, native=args.native, favicon="🎞️", show_welcome_message=False,
        uvicorn_logging_level="warning",
    )
    return 0


if __name__ in {"__main__", "__mp_main__"}:
    sys.exit(main())
