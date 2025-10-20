from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType

_CURRENT_MODULE = sys.modules.get(__name__)
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent.resolve()

log = logging.getLogger(__name__)

_removed_paths: list[tuple[int, str]] = []
for idx in range(len(sys.path) - 1, -1, -1):
    try:
        candidate = Path(sys.path[idx]).resolve()
    except Exception as exc:
        log.debug("fastapi.path_resolve_failed", idx=idx, error=str(exc), exc_info=True)
        continue
    if candidate == _PROJECT_ROOT:
        _removed_paths.append((idx, sys.path[idx]))
        sys.path.pop(idx)

_real_module: ModuleType | None = None
try:
    sys.modules.pop(__name__, None)
    _real_module = importlib.import_module(__name__)
except ModuleNotFoundError:
    _real_module = None
finally:
    for idx, path in reversed(_removed_paths):
        sys.path.insert(idx, path)

if _real_module is not None and getattr(_real_module, "__file__", None) != __file__:
    try:
        from starlette.datastructures import URL as _StarletteURL  # type: ignore
    except Exception:  # pragma: no cover - optional dependency resolution
        _StarletteURL = None  # type: ignore[assignment]

    if _StarletteURL is not None and not hasattr(_real_module, "URL"):
        _real_module.URL = _StarletteURL
        real_all = list(getattr(_real_module, "__all__", []))
        if "URL" not in real_all:
            real_all.append("URL")
            _real_module.__all__ = real_all

    sys.modules[__name__] = _real_module
    globals().update(_real_module.__dict__)
    __all__ = getattr(
        _real_module,
        "__all__",
        [name for name in globals() if not name.startswith("_")],
    )
else:  # pragma: no cover - fallback stub
    if _CURRENT_MODULE is not None:
        sys.modules[__name__] = _CURRENT_MODULE
    from ._stub import *  # noqa: F401, F403

    __all__ = [name for name in globals() if not name.startswith("_")]
