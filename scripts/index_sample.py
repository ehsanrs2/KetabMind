from __future__ import annotations

from pathlib import Path

from scripts.index_file import index_path


def main() -> None:
    sample = Path("docs/fixtures/sample.pdf")
    if not sample.exists():
        raise SystemExit("docs/fixtures/sample.pdf not found. Provide a sample PDF.")
    index_path(sample)


if __name__ == "__main__":
    main()
