import pytest
from fastapi.testclient import TestClient

from ws_server import app, create_jwt


@pytest.mark.asyncio
async def test_websocket_echo_and_ping(monkeypatch):
    token = create_jwt({"sub": "tester"})
    monkeypatch.setattr("ws_server.PING_INTERVAL_SECONDS", 0.01)

    with TestClient(app) as client:
        with client.websocket_connect(f"/ws?token={token}") as websocket:
            payload = {"type": "delta", "delta": "سلام"}
            websocket.send_json(payload)

            messages = [websocket.receive_json() for _ in range(2)]

    assert {**payload, "echoed": True} in messages
    assert {"type": "ping"} in messages
