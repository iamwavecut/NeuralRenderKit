"""User settings for the web front end: weight locations, backend, output folder."""
from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def default_root() -> Path:
    return Path(os.environ.get("NRK_WEB_ROOT", Path.home() / "NeuralRenderKit"))


@dataclass
class Settings:
    root: str = field(default_factory=lambda: str(default_root()))
    nr_weights: str = ""          # logical safetensors (nrk-weights decode) for the torch backend
    nr_model: str = ""            # MODEL.nrkmodel for the Swift/Metal backend
    fg_weights: str = ""          # dense frame generation safetensors (nrk-weights extract-fg)
    backend: str = "auto"         # auto | torch | nrk
    device: str = "auto"          # torch device
    precision: str = "reference"  # torch precision for neural rendering (reference | fast ...)
    nrk_binary: str = ""          # explicit path to nrk (default: PATH or the repository build)
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

    def nrk_available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        try:
            from ..nrk_stream import find_nrk

            find_nrk(self.nrk_binary or None)
            return True
        except RuntimeError:
            return False

    def resolved_backend(self, effect: str = "nr") -> str:
        """Backend for an effect: ``auto`` picks Metal when ``nrk`` is available (and, for neural
        rendering, a .nrkmodel is configured), PyTorch otherwise."""
        if self.backend == "nrk":
            return "nrk"
        if self.backend == "auto" and self.nrk_available() and (effect != "nr" or self.nr_model):
            return "nrk"
        return "torch"

    def has_nr_weights(self) -> bool:
        target = self.nr_model if self.resolved_backend("nr") == "nrk" else self.nr_weights
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
        for key, attribute in (("NRK_NR_WEIGHTS", "nr_weights"), ("NRK_NR_MODEL", "nr_model"), ("NRK_FG_WEIGHTS", "fg_weights"), ("NRK_BINARY", "nrk_binary")):
            if os.environ.get(key) and not getattr(settings, attribute):
                setattr(settings, attribute, os.environ[key])
        return settings

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2))
