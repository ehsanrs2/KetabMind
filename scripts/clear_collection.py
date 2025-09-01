from __future__ import annotations

import argparse

from core.config import settings
from core.embed.adapter import get_embedder
from core.vector.qdrant_client import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("collection", help="Collection name to delete")
    args = parser.parse_args()

    embedder = get_embedder(settings.embed_model)
    store = VectorStore(
        mode=settings.qdrant_mode,
        location=settings.qdrant_location,
        url=settings.qdrant_url,
        collection=args.collection,
        vector_size=embedder.dim,
    )
    try:
        store.delete_collection()
        print(f"Deleted collection: {args.collection}")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to delete collection: {e}")


if __name__ == "__main__":
    main()
