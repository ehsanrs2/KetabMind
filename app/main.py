"""Expose the FastAPI app for ASGI servers."""

from apps.api.main import app as api_app

app = api_app

__all__ = ["app"]
