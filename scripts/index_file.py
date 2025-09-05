from __future__ import annotations

import os
from pathlib import Path

from core.config import settings
from core.index import index_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Input file (.pdf or .txt)")
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", settings.qdrant_collection),
        help="Qdrant collection name",
    )
    args = parser.parse_args()
    in_path = Path(args.path)
    index_path(in_path, collection=args.collection)


if __name__ == "__main__":
    main()
