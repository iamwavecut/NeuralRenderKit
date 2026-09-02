"""Compatibility shim: the implementation lives in ``python/neuralrenderkit``.

Importing this module yields the package module itself (same object), so
callers that patch attributes or compare classes keep working.
"""
import pathlib
import sys

_PACKAGE_ROOT = str(pathlib.Path(__file__).resolve().parents[1] / "python")
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from neuralrenderkit.tools.extract_dlssnr_weights import *  # noqa: E402,F401,F403
from neuralrenderkit.tools.extract_dlssnr_weights import main  # noqa: E402,F401

import neuralrenderkit.tools.extract_dlssnr_weights as _implementation  # noqa: E402

sys.modules[__name__] = _implementation

if __name__ == "__main__":
    raise SystemExit(main())
