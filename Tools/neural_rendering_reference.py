"""Compatibility shim: the implementation lives in ``python/neuralrenderkit``.

Importing this module yields the package module itself (same object), so
callers that patch attributes or compare classes keep working.
"""
import pathlib
import sys

_PACKAGE_ROOT = str(pathlib.Path(__file__).resolve().parents[1] / "python")
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from neuralrenderkit.model import *  # noqa: E402,F401,F403

import neuralrenderkit.model as _implementation  # noqa: E402

sys.modules[__name__] = _implementation

