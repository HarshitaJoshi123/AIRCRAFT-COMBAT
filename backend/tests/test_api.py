import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_loaded" in body
    assert "status" in body


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_simulation_lifecycle_via_rest(client):
    r = client.post("/api/simulation/reset")
    assert r.status_code == 200
    assert r.json()["status"] == "IDLE"

    r = client.post("/api/simulation/start")
    assert r.status_code == 200
    assert r.json()["status"] == "ACTIVE"

    r = client.post("/api/simulation/start")
    assert r.status_code == 409  # already active

    r = client.post("/api/simulation/pause")
    assert r.status_code == 200
    assert r.json()["status"] == "PAUSED"

    r = client.post("/api/simulation/resume")
    assert r.status_code == 200
    assert r.json()["status"] == "ACTIVE"

    r = client.post("/api/simulation/stop")
    assert r.status_code == 200


def test_websocket_simulation_connects(client):
    with client.websocket_connect("/ws/simulation") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "status"
