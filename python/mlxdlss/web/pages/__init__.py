"""NiceGUI pages: Image, Video, Jobs, Settings."""
from __future__ import annotations


def register_pages() -> None:
    """Import the page modules so their ``@ui.page`` routes register with the app."""
    from . import image, jobs, settings, video  # noqa: F401
