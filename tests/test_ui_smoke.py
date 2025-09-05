from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx


def test_ui_smoke() -> None:
    ui_dir = Path("apps/ui")
    env = {**os.environ, "CI": "1"}
    subprocess.run(  # noqa: S603
        ["npm", "install", "--no-audit", "--no-fund"],  # noqa: S607
        cwd=ui_dir,
        env=env,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["npm", "run", "build"],  # noqa: S607
        cwd=ui_dir,
        env=env,
        check=True,
    )
    proc = subprocess.Popen(  # noqa: S603
        ["npm", "run", "start", "--", "-p", "3100"],  # noqa: S607
        cwd=ui_dir,
        env=env,
    )
    try:
        for _ in range(30):
            try:
                r = httpx.get("http://127.0.0.1:3100", timeout=1.0)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: S110
                time.sleep(1)
                continue
            time.sleep(1)
        else:
            raise AssertionError("UI did not start")
        assert "KetabMind" in r.text
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
