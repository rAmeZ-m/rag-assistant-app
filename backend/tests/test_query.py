import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # with = يشغّل الـ lifespan ويحمّل الـ vector store
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query_happy_path(client):
    r = client.post("/query", json={"question": "What is meta-reinforcement learning?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert len(body["sources"]) >= 1


def test_query_out_of_scope_is_refused(client):
    r = client.post("/query", json={"question": "What is the capital of France?"})
    assert r.status_code == 200
    assert r.json()["sources"] == []


def test_query_invalid_input(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={}).status_code == 422