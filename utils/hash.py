"""Hashing utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return a namespaced sha256 hash for a file."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
