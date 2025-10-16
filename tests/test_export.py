from __future__ import annotations

import gzip
import importlib
import io
import zipfile
from pathlib import Path

import pytest

from fastapi.testclient import TestClient
from tests.helpers import setup_api_stubs


@pytest.fixture()
def export_test_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, int]:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(tmp_path / "qdrant"))
    monkeypatch.setenv("QDRANT_COLLECTION", "export-tests")
    monkeypatch.setenv("EMBED_MODEL", "mock")

    setup_api_stubs(monkeypatch)

    import apps.api.db.repositories as repositories
    import apps.api.db.session as db_session
    import apps.api.main as api_main
    import core.config as config
    from apps.api.db.base import Base

    importlib.reload(config)
    importlib.reload(db_session)
    importlib.reload(api_main)

    Base.metadata.create_all(db_session._engine)

    with db_session.session_scope() as session:
        user_repo = repositories.UserRepository(session)
        user = user_repo.create(email="alice@example.com", name="Alice")
        session_repo = repositories.SessionRepository(session, user.id)
        chat_session = session_repo.create(title="Export Session")
        message_repo = repositories.MessageRepository(session, user.id)
        message_repo.create(
            session_id=chat_session.id,
            role="user",
            content="What is the summary?",
        )
        answer = message_repo.create(
            session_id=chat_session.id,
            role="assistant",
            content="- Point one\n- Point two",
            citations=["[book1:10-12]", "[book2:5-6]"],
            meta={"coverage": 0.75, "confidence": 0.5},
        )
        answer_id = answer.id

    client = TestClient(api_main.app)
    return client, answer_id


def test_export_generates_pdf_and_word(export_test_client: tuple[TestClient, int]) -> None:
    client, message_id = export_test_client

    pdf_response = client.post("/export", json={"message_id": message_id, "format": "pdf"})
    assert pdf_response.status_code == 200
    assert pdf_response.headers["content-type"] == "application/pdf"

    pdf_bytes = pdf_response.content
    if pdf_response.headers.get("content-encoding") == "gzip":
        pdf_bytes = gzip.decompress(pdf_bytes)
    assert b"Citations:" in pdf_bytes
    assert b"[book1:10-12]" in pdf_bytes
    assert b"Coverage: 0.75" in pdf_bytes
    assert b"Confidence: 0.5" in pdf_bytes

    docx_response = client.post("/export", json={"message_id": message_id, "format": "word"})
    assert docx_response.status_code == 200
    assert (
        docx_response.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    docx_bytes = docx_response.content
    if docx_response.headers.get("content-encoding") == "gzip":
        docx_bytes = gzip.decompress(docx_bytes)

    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    assert "Question:" in document_xml
    assert "What is the summary?" in document_xml
    assert "Point one" in document_xml
    assert "Point two" in document_xml
    assert "[book1:10-12]" in document_xml
    assert "Coverage: 0.75" in document_xml
    assert "Confidence: 0.5" in document_xml
