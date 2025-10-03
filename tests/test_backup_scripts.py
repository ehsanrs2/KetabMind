import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


def test_backup_creates_archive(tmp_path):
    storage_dir = tmp_path / "qdrant_storage"
    storage_dir.mkdir()
    (storage_dir / "data.bin").write_text("storage")

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "doc.txt").write_text("upload")

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "QDRANT_STORAGE_DIR": str(storage_dir),
            "UPLOAD_DIR": str(upload_dir),
            "QDRANT_BACKUP_DIR": str(backup_dir),
        }
    )

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "backup_qdrant.sh"
    assert script_path.is_file(), "Backup script is missing"

    bash_path = shutil.which("bash")
    if bash_path is None:
        pytest.skip("bash executable not available")

    result = subprocess.run(  # noqa: S603
        [bash_path, str(script_path)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    backups = list(backup_dir.glob("qdrant_backup_*.tar.gz"))
    assert backups, f"No backup created. stdout={result.stdout}, stderr={result.stderr}"
    backup_file = backups[0]

    with tarfile.open(backup_file, "r:gz") as archive:
        archive_names = archive.getnames()

    assert f"{storage_dir.name}/data.bin" in archive_names
    assert f"{upload_dir.name}/doc.txt" in archive_names
