from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

try:  # pragma: no cover - fallback for <3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_version() -> str:
    with (project_root() / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return data["tool"]["poetry"]["version"]


def ensure_config() -> Path:
    root = project_root()
    config_dir = root / "installer"
    config_dir.mkdir(exist_ok=True)

    system = platform.system().lower()
    if system.startswith("linux"):
        config_name = "linux.json"
    elif system.startswith("darwin"):
        config_name = "darwin.json"
    else:
        config_name = "windows.json"

    config_path = config_dir / config_name
    if not config_path.exists():
        config_path.write_text(
            "{\n"
            "  \"appId\": \"com.ketabmind.desktop\",\n"
            "  \"productName\": \"KetabMind Desktop\",\n"
            "  \"directories\": {\n"
            "    \"output\": \"dist\"\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
    return config_path


def test_build_desktop_creates_versioned_artifact():
    root = project_root()
    ensure_config()

    dist_dir = root / "dist"
    if dist_dir.exists():
        for existing in dist_dir.iterdir():
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
    dist_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env.setdefault("PYTHON_BIN", "python3")

    subprocess.run(
        ["bash", str(root / "scripts" / "build_desktop.sh")],
        check=True,
        cwd=root,
        env=env,
    )

    version = read_version()
    artifacts = list(dist_dir.glob(f"ketabmind-desktop-{version}*"))
    assert artifacts, "No desktop build artifacts with embedded version were created"
    assert all(artifact.is_file() for artifact in artifacts), "Artifacts must be regular files"

    for artifact in artifacts:
        artifact.unlink()
