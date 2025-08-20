import os

import typer

app = typer.Typer()


@app.command()  # type: ignore[misc]
def main(collection: str, mode: str = typer.Option("local", "--mode")) -> None:
    os.environ["QDRANT_MODE"] = mode
    from core.vector.qdrant import VectorStore

    store = VectorStore()
    store.collection = collection
    store.wipe_collection()


if __name__ == "__main__":
    app()
