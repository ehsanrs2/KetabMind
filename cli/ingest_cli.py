from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import click

from ingest.pipeline import ingest_file, write_records


def _load_metadata(value: str | None) -> Mapping[str, Any] | None:
    if value is None:
        return None

    candidate = Path(value)
    if candidate.exists():
        raw = candidate.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise click.BadParameter(f"Invalid JSON in metadata file: {candidate}") from exc
    else:
        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise click.BadParameter("--with-meta expects a JSON object or path to JSON file") from exc

    if not isinstance(data, Mapping):
        raise click.BadParameter("Metadata must decode to a JSON object")
    return data


@click.command(help="Process an input document and write normalised JSONL records.")
@click.argument("input_path", type=click.Path(path_type=Path, exists=True, readable=True))
@click.option("--output", "-o", required=True, type=click.Path(path_type=Path), help="Destination JSONL path")
@click.option(
    "--with-meta",
    "with_meta",
    type=str,
    default=None,
    help="Inline JSON object or path to metadata JSON file",
)
def run(input_path: Path, output: Path, with_meta: str | None) -> None:
    metadata = _load_metadata(with_meta)
    result = ingest_file(input_path, metadata=metadata)
    write_records(result.records, output)
    click.echo(
        json.dumps(
            {
                "book_id": result.book_id,
                "version": result.version,
                "file_hash": result.file_hash,
                "records": len(result.records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    run()
