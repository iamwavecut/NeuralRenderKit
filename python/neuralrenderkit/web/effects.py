"""Effect models: what a job applies, in order, and what the current backend can do."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from ..features import PROFILES

PROFILE_NAMES = tuple(PROFILES)


class NeuralRender(BaseModel):
    """The neural-rendering transformer (detail, colour, tone) on every frame."""

    kind: Literal["nr"] = "nr"
    profile: str = "standard"
    processing_scale: float = Field(1.0, ge=1.0, le=4.0)
    detail_strength: float = Field(1.0, ge=0.0, le=4.0)
    colour_strength: float = Field(1.0, ge=0.0, le=4.0)
    detail_radius: float = Field(4.0, ge=1.0, le=16.0)
    intensity: float = Field(1.0, ge=0.0, le=2.0)
    temporal: bool = False            # video only: reprojected history + learned blend

    def model_post_init(self, _context) -> None:
        if self.profile not in PROFILE_NAMES:
            raise ValueError(f"profile must be one of {PROFILE_NAMES}")
        if self.temporal and self.processing_scale != 1.0:
            raise ValueError("temporal mode runs at the native scale (processing_scale must be 1)")


class FrameGen(BaseModel):
    """DLSS frame generation between consecutive frames (video only)."""

    kind: Literal["fg"] = "fg"
    mode: Literal["fps", "slowmo"] = "fps"
    factor: Literal[2, 3, 4] = 2
    audio: Literal["copy", "stretch", "none"] = "copy"

    def model_post_init(self, _context) -> None:
        if self.audio == "stretch" and self.mode != "slowmo":
            raise ValueError("audio 'stretch' only applies to slowmo")


Effect = Annotated[Union[NeuralRender, FrameGen], Field(discriminator="kind")]
EffectList = TypeAdapter(list[Effect])

MediaKind = Literal["image", "video"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def media_kind(filename: str) -> MediaKind:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    raise ValueError(f"unsupported file type '{suffix}': images {sorted(IMAGE_SUFFIXES)}, videos {sorted(VIDEO_SUFFIXES)}")


def parse_effects(raw) -> list[NeuralRender | FrameGen]:
    """Validate a list of effect dicts (or models) into models; raises ValueError."""
    try:
        return EffectList.validate_python([e.model_dump() if isinstance(e, BaseModel) else e for e in raw])
    except ValidationError as error:
        raise ValueError("; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors())) from error


def validate_chain(effects: list[NeuralRender | FrameGen], kind: MediaKind) -> None:
    """The rules a job's effect chain must follow; raises ValueError."""
    if not effects:
        raise ValueError("choose at least one effect")
    if kind == "image" and any(isinstance(e, FrameGen) for e in effects):
        raise ValueError("frame generation needs a video; images take neural rendering only")
    if sum(isinstance(e, FrameGen) for e in effects) > 1:
        raise ValueError("at most one frame generation effect per job")
    if sum(isinstance(e, NeuralRender) for e in effects) > 1:
        raise ValueError("at most one neural rendering effect per job")


def describe_effects(*, nrk_available: bool, fg_weights: bool, nr_weights: bool) -> dict:
    """What the UI/API can offer right now (field ranges and availability)."""
    return {
        "effects": [
            {
                "kind": "nr", "name": "Neural rendering", "media": ["image", "video"], "available": nr_weights,
                "fields": {
                    "profile": {"type": "choice", "choices": list(PROFILE_NAMES), "default": "standard"},
                    "processing_scale": {"type": "number", "min": 1.0, "max": 4.0, "default": 1.0},
                    "detail_strength": {"type": "number", "min": 0.0, "max": 4.0, "default": 1.0},
                    "colour_strength": {"type": "number", "min": 0.0, "max": 4.0, "default": 1.0},
                    "detail_radius": {"type": "number", "min": 1.0, "max": 16.0, "default": 4.0},
                    "intensity": {"type": "number", "min": 0.0, "max": 2.0, "default": 1.0},
                    "temporal": {"type": "bool", "default": False, "media": ["video"]},
                },
            },
            {
                "kind": "fg", "name": "Frame generation", "media": ["video"], "available": fg_weights,
                "fields": {
                    "mode": {"type": "choice", "choices": ["fps", "slowmo"], "default": "fps"},
                    "factor": {"type": "choice", "choices": [2, 3, 4], "default": 2},
                    "audio": {"type": "choice", "choices": ["copy", "stretch", "none"], "default": "copy"},
                },
            },
        ],
        "backends": {"torch": True, "nrk": nrk_available},
    }
