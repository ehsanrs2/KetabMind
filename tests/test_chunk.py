from core.chunk.sliding import chunk_text


def test_chunk_text() -> None:
    lines = ["word" for _ in range(50)]
    chunks = chunk_text(lines, size=10, overlap=5)
    assert len(chunks) > 0
