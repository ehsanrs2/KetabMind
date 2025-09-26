import sys
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if not hasattr(httpx, "BaseTransport"):

    class _CompatBaseTransport:  # pragma: no cover - compatibility shim
        def __enter__(self) -> "_CompatBaseTransport":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

        def close(self) -> None:
            return None

    httpx.BaseTransport = _CompatBaseTransport


if not hasattr(httpx._client, "UseClientDefault"):

    class _UseClientDefault:  # pragma: no cover - compatibility shim
        pass

    httpx._client.UseClientDefault = _UseClientDefault
    httpx._client.USE_CLIENT_DEFAULT = _UseClientDefault()


if "follow_redirects" not in httpx.Client.__init__.__code__.co_varnames:
    _orig_client_init = httpx.Client.__init__

    def _compat_client_init(self: httpx.Client, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("follow_redirects", None)
        kwargs.pop("allow_redirects", None)
        kwargs.setdefault("trust_env", False)
        _orig_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _compat_client_init
