import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_db_path, monkeypatch):
    monkeypatch.setenv("ARBITRATION_DB_PATH", tmp_db_path)
    from api.main import app

    return TestClient(app)


def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["provider_mode"] == "mock"


def test_arbitrate_rejects_empty_output(client):
    resp = client.post("/v1/arbitrate", json={"output": "   "})
    assert resp.status_code == 422


def test_arbitrate_and_fetch_round_trip(client):
    resp = client.post("/v1/arbitrate", json={"output": "The Eiffel Tower is in London.", "prompt": "Where is it?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "verdict" in body and "critic_runs" in body

    fetch = client.get(f"/v1/arbitrations/{body['id']}")
    assert fetch.status_code == 200
    assert fetch.json()["id"] == body["id"]


def test_arbitrate_batch(client):
    resp = client.post(
        "/v1/arbitrate/batch",
        json={"items": [{"output": "a clean fact"}, {"output": "another clean fact"}]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_get_unknown_arbitration_404s(client):
    resp = client.get("/v1/arbitrations/does-not-exist")
    assert resp.status_code == 404


def test_count_and_list_routes_not_shadowed_by_dynamic_route(client):
    client.post("/v1/arbitrate", json={"output": "some output"})
    count_resp = client.get("/v1/arbitrations/count")
    assert count_resp.status_code == 200
    assert count_resp.json()["count"] == 1

    list_resp = client.get("/v1/arbitrations")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1
