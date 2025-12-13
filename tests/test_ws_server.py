import pytest
from fastapi.testclient import TestClient

from ws_server import app, create_jwt


@pytest.mark.asyncio
async def test_websocket_echo_with_valid_jwt():
    token = create_jwt({"sub": "tester"})

    with TestClient(app) as client:
        with client.websocket_connect(f"/ws?token={token}") as websocket:
            payload = {"type": "delta", "delta": "سلام"}
            websocket.send_json(payload)
            message = websocket.receive_json()

    assert message == {**payload, "echoed": True}
