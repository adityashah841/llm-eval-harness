import pytest
from fastapi.testclient import TestClient

import api.db as db
from api.main import app
from api.store import run_store
from llm_eval.adapters.base import ModelResponse
from llm_eval.adapters.ollama_adapter import OllamaAdapter


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point every test at a throwaway SQLite file instead of data/metrics.db."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_metrics.db")


@pytest.fixture(autouse=True)
def stub_ollama(monkeypatch):
    """Replace real Ollama HTTP calls with a fast, deterministic stub.

    Keeps these tests independent of whether `ollama serve` happens to be
    running, without touching llm_eval/adapters itself.
    """

    async def fake_ask(self, prompt, system_prompt=""):
        return ModelResponse(
            model_name=self.model_name,
            response_text="This is a stubbed response for testing.",
            latency_ms=1.0,
            input_tokens=10,
            output_tokens=5,
        )

    monkeypatch.setattr(OllamaAdapter, "ask", fake_ask)


@pytest.fixture
def client():
    run_store._runs.clear()
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    names = {m["name"] for m in data}
    assert "gemma3:4b" in names
    assert all("available" in m for m in data)


def test_create_run_unknown_dataset_404(client):
    resp = client.post("/runs", json={"dataset": "does_not_exist", "models": ["gemma3:4b"]})
    assert resp.status_code == 404


def test_create_run_requires_at_least_one_model(client):
    resp = client.post("/runs", json={"dataset": "legal_qa", "models": []})
    assert resp.status_code == 422


def test_create_and_fetch_run(client):
    resp = client.post(
        "/runs",
        json={
            "dataset": "legal_qa",
            "models": ["gemma3:4b"],
            "smoke_test": True,
            "use_mlflow": False,
        },
    )
    assert resp.status_code == 202
    body = resp.json()
    run_id = body["id"]
    assert body["status"] in ("pending", "running", "completed")

    status_resp = client.get(f"/runs/{run_id}")
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body["status"] == "completed"
    assert status_body["sample_count"] == 2
    assert status_body["error"] is None

    results_resp = client.get(f"/runs/{run_id}/results")
    assert results_resp.status_code == 200
    results = results_resp.json()
    assert len(results) == 2
    assert results[0]["model_name"] == "gemma3:4b"
    assert results[0]["response_text"] == "This is a stubbed response for testing."


def test_get_unknown_run_404(client):
    resp = client.get("/runs/does-not-exist")
    assert resp.status_code == 404


def test_get_results_unknown_run_404(client):
    resp = client.get("/runs/does-not-exist/results")
    assert resp.status_code == 404


def test_timeseries_returns_list(client):
    resp = client.get("/metrics/timeseries")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_completed_run_is_persisted_and_appears_in_timeseries(client):
    resp = client.post(
        "/runs",
        json={
            "dataset": "legal_qa",
            "models": ["gemma3:4b"],
            "smoke_test": True,
            "use_mlflow": False,
        },
    )
    run_id = resp.json()["id"]

    timeseries = client.get("/metrics/timeseries").json()
    point = next((p for p in timeseries if p["run_id"] == run_id), None)
    assert point is not None
    assert point["model_name"] == "gemma3:4b"
    assert point["dataset"] == "datasets\\legal_qa" or point["dataset"] == "datasets/legal_qa"
    assert point["rougeL_mean"] is not None
    assert point["halluc_rate"] in (0.0, 0.5, 1.0)

    filtered = client.get("/metrics/timeseries", params={"model": "gemma3:4b"}).json()
    assert any(p["run_id"] == run_id for p in filtered)

    filtered_out = client.get("/metrics/timeseries", params={"model": "phi4"}).json()
    assert all(p["run_id"] != run_id for p in filtered_out)
