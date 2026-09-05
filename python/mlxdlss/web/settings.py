"""User settings for the web front end: weight locations, backend, output folder."""
from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def default_root() -> Path:
    return Path(os.environ.get("MLXDLSS_WEB_ROOT", Path.home() / "MLX-DLSS"))


@dataclass
class Settings:
    root: str = field(default_factory=lambda: str(default_root()))
    nr_weights: str = ""          # logical safetensors (mlxdlss-weights decode) for the torch backend
    nr_model: str = ""            # MODEL.dlssmodel for the Swift/Metal backend
    fg_weights: str = ""          # dense frame generation safetensors (mlxdlss-weights extract-fg)
    backend: str = "auto"         # auto | torch | mlxdlss
    device: str = "auto"          # torch device
    precision: str = "reference"  # torch precision for neural rendering (reference | fast ...)
    mlxdlss_binary: str = ""          # explicit path to mlxdlss (default: PATH or the repository build)
    theme_dark: bool = False

    @property
    def root_path(self) -> Path:
        return Path(self.root).expanduser()

    @property
    def outputs(self) -> Path:
        return self.root_path / "outputs"

    @property
    def uploads(self) -> Path:
        return self.root_path / "uploads"

    def mlxdlss_available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        try:
            from ..mlxdlss_stream import find_mlxdlss

            find_mlxdlss(self.mlxdlss_binary or None)
            return True
        except RuntimeError:
            return False

    def resolved_backend(self, effect: str = "nr") -> str:
        """Backend for an effect: ``auto`` picks Metal when ``mlxdlss`` is available (and, for neural
        rendering, a .dlssmodel is configured), PyTorch otherwise."""
        if self.backend == "mlxdlss":
            return "mlxdlss"
        if self.backend == "auto" and self.mlxdlss_available() and (effect != "nr" or self.nr_model):
            return "mlxdlss"
        return "torch"

    def has_nr_weights(self) -> bool:
        target = self.nr_model if self.resolved_backend("nr") == "mlxdlss" else self.nr_weights
        return bool(target) and Path(target).expanduser().exists()

    def has_fg_weights(self) -> bool:
        return bool(self.fg_weights) and Path(self.fg_weights).expanduser().exists()

    # -- persistence -----------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.root_path / "settings.json"

    @classmethod
    def load(cls, root: str | Path | None = None) -> "Settings":
        settings = cls()
        if root is not None:
            settings.root = str(root)
        try:
            data = json.loads(settings.path.read_text())
            data.pop("root", None) if root is not None else None
            for f in fields(cls):
                if f.name in data:
                    setattr(settings, f.name, data[f.name])
        except (OSError, ValueError):
            pass
        for key, attribute in (("MLXDLSS_NR_WEIGHTS", "nr_weights"), ("MLXDLSS_NR_MODEL", "nr_model"), ("MLXDLSS_FG_WEIGHTS", "fg_weights"), ("MLXDLSS_BINARY", "mlxdlss_binary")):
            if os.environ.get(key) and not getattr(settings, attribute):
                setattr(settings, attribute, os.environ[key])
        return settings

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2))
