"""NeuralRenderKit: PyTorch inference for the recovered neural-rendering transformer."""

from .features import PROFILES, AutomaticMask, NetworkGeometry, deterministic_noise, make_features
from .composition import compose_detail, compose_head, resample
from .pipeline import EnhanceResult, NeuralRenderingPipeline, NeuralRenderingSession, PreparedFrame, load_weights, resolve_device
from .temporal import BLEND_SCALE, TemporalOptions, TemporalSession, compose_temporal, make_temporal_features, normalize_pixel_motion
from .video import ConvertOptions, ConvertResult, VideoInfo, convert, probe

__version__ = "0.1.0"
__all__ = [
    "PROFILES", "AutomaticMask", "NetworkGeometry", "deterministic_noise", "make_features",
    "compose_detail", "compose_head", "resample",
    "EnhanceResult", "NeuralRenderingPipeline", "NeuralRenderingSession", "PreparedFrame", "load_weights", "resolve_device",
    "ConvertOptions", "ConvertResult", "VideoInfo", "convert", "probe",
    "BLEND_SCALE", "TemporalOptions", "TemporalSession", "compose_temporal", "make_temporal_features", "normalize_pixel_motion",
]
