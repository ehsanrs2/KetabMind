import re
import sys
import types

from fastapi.testclient import TestClient


def _ensure_qdrant_stub() -> None:
    if "qdrant_client" in sys.modules:
        return

    client_module = types.ModuleType("qdrant_client")

    class _StubQdrantClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._points: list[object] = []

        def get_collection(self, *args: object, **kwargs: object) -> None:
            raise Exception("collection missing")

        def recreate_collection(self, *args: object, **kwargs: object) -> None:
            return None

        def scroll(self, *args: object, **kwargs: object) -> tuple[list[object], None]:
            return ([], None)

        def upsert(self, *args: object, **kwargs: object) -> None:
            self._points.extend(kwargs.get("points", []))

        def search(self, *args: object, **kwargs: object) -> list[object]:
            return []

        def delete_collection(self, *args: object, **kwargs: object) -> None:
            self._points.clear()

    client_module.QdrantClient = _StubQdrantClient

    http_module = types.ModuleType("qdrant_client.http")
    models_module = types.ModuleType("qdrant_client.http.models")

    class _VectorParams:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    class _CollectionParams:
        def __init__(self, vectors: object | None = None, *args: object, **kwargs: object) -> None:
            self.vectors = vectors

    class _Filter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    class _FieldCondition:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    class _MatchValue:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    class _PointStruct:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    models_module.VectorParams = _VectorParams
    models_module.CollectionParams = _CollectionParams
    models_module.Filter = _Filter
    models_module.FieldCondition = _FieldCondition
    models_module.MatchValue = _MatchValue
    models_module.PointStruct = _PointStruct
    models_module.Distance = types.SimpleNamespace(COSINE="COSINE")

    http_module.models = models_module

    sys.modules["qdrant_client"] = client_module
    sys.modules["qdrant_client.http"] = http_module
    sys.modules["qdrant_client.http.models"] = models_module


_ensure_qdrant_stub()

from app.main import app  # noqa: E402


def _metric_value(metrics_text: str, metric: str, labels: dict[str, str]) -> float:
    pattern = rf"{metric}\{{([^}}]+)\}} ([0-9.eE+-]+)"
    for match in re.finditer(pattern, metrics_text):
        raw_labels = match.group(1)
        parsed: dict[str, str] = {}
        for item in raw_labels.split(","):
            key, _, value = item.partition("=")
            parsed[key.strip()] = value.strip().strip('"')
        if all(parsed.get(k) == v for k, v in labels.items()):
            return float(match.group(2))
    return 0.0


def test_metrics_increment_after_requests() -> None:
    client = TestClient(app)

    baseline_metrics = client.get("/metrics").text()
    baseline_health = _metric_value(
        baseline_metrics,
        "api_requests_total",
        {"path": "/health", "method": "GET", "status": "200"},
    )
    baseline_ready = _metric_value(
        baseline_metrics,
        "api_requests_total",
        {"path": "/ready", "method": "GET", "status": "200"},
    )
    baseline_latency_health = _metric_value(
        baseline_metrics,
        "api_request_latency_seconds_count",
        {"path": "/health", "method": "GET"},
    )
    baseline_latency_ready = _metric_value(
        baseline_metrics,
        "api_request_latency_seconds_count",
        {"path": "/ready", "method": "GET"},
    )

    client.get("/health")
    client.get("/ready")

    metrics = client.get("/metrics")
    assert metrics.status_code == 200

    updated_health = _metric_value(
        metrics.text(),
        "api_requests_total",
        {"path": "/health", "method": "GET", "status": "200"},
    )
    updated_ready = _metric_value(
        metrics.text(),
        "api_requests_total",
        {"path": "/ready", "method": "GET", "status": "200"},
    )

    assert updated_health >= baseline_health + 1
    assert updated_ready >= baseline_ready + 1

    latency_health = _metric_value(
        metrics.text(),
        "api_request_latency_seconds_count",
        {"path": "/health", "method": "GET"},
    )
    latency_ready = _metric_value(
        metrics.text(),
        "api_request_latency_seconds_count",
        {"path": "/ready", "method": "GET"},
    )

    assert latency_health >= baseline_latency_health + 1
    assert latency_ready >= baseline_latency_ready + 1
