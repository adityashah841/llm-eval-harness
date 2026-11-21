from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import api.db as db
from api.config import cors_origins
from api.main import app
from api.store import RunRecord, RunStatus, run_store
from llm_eval.runner.prompt_runner import EVAL_VERSION

from .conftest import STUB_TEXT

RUN = {"dataset": "legal_qa", "models": ["gemma3:4b"], "smoke_test": True, "use_mlflow": False}


@pytest.fixture
def client():
    run_store._runs.clear()
    with TestClient(app) as c:
        yield c


def _start(client, **overrides):
    resp = client.post("/runs", json={**RUN, **overrides})
    assert resp.status_code == 202, resp.text
    return resp.json()["id"]


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "eval_version": EVAL_VERSION}


def test_list_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    names = {m["name"] for m in resp.json()}
    assert "gemma3:4b" in names
    assert all("available" in m for m in resp.json())


def test_list_datasets(client):
    resp = client.get("/datasets")
    assert resp.status_code == 200
    by_name = {d["name"]: d for d in resp.json()}
    assert set(by_name) >= {"legal_qa", "summarization", "code_gen"}
    assert by_name["code_gen"]["scoring"] == "unit_tests"
    assert by_name["legal_qa"]["scoring"] == "rouge_nli"
    assert by_name["legal_qa"]["sample_count"] == 5
    assert by_name["legal_qa"]["path"] == "datasets/legal_qa"


@pytest.mark.parametrize(
    "dataset", ["does_not_exist", "/etc", "../api", "datasets/../api", "api", "legal_qa/legal_qa.yaml", "C:\\Windows"]
)
def test_create_run_rejects_unknown_or_outside_datasets(client, dataset):
    resp = client.post("/runs", json={**RUN, "dataset": dataset})
    assert resp.status_code == 404


def test_create_run_requires_at_least_one_model(client):
    resp = client.post("/runs", json={**RUN, "models": []})
    assert resp.status_code == 422


def test_create_and_fetch_run(client):
    run_id = _start(client)

    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "completed"
    assert body["sample_count"] == 2
    assert body["error"] is None
    assert body["eval_version"] == EVAL_VERSION
    assert body["model_digests"] == {"gemma3:4b": "sha256:gemma3:4b"}

    results = client.get(f"/runs/{run_id}/results").json()
    assert len(results) == 2
    first = results[0]
    assert first["model_name"] == "gemma3:4b"
    assert first["response_text"] == STUB_TEXT
    assert first["rougeL"] is not None
    assert first["hallucination_flag"] in (True, False)
    assert first["tests_passed"] is None  # text sample: no unit tests
    assert first["generation_ms"] == 0.5


def test_dataset_path_form_is_accepted_and_normalized(client):
    run_id = _start(client, dataset="datasets/legal_qa")
    assert client.get(f"/runs/{run_id}").json()["dataset"] == "datasets/legal_qa"


def test_code_dataset_is_scored_by_execution(client):
    run_id = _start(client, dataset="code_gen")
    results = client.get(f"/runs/{run_id}/results").json()
    assert len(results) == 2
    for r in results:
        # The stub reply is prose, not code, so the tests must fail.
        assert r["tests_passed"] is False
        assert r["exec_error"]
        assert r["rougeL"] is None and r["hallucination_flag"] is None


def test_partial_run_records_failures(client):
    run_id = _start(client, models=["gemma3:4b", "broken:model"])
    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "partial"
    assert body["sample_count"] == 2
    assert len(body["failures"]) == 2
    assert body["failures"][0]["model_name"] == "broken:model"
    assert "model not found" in body["error"]


def test_run_where_every_call_fails_is_failed_not_completed(client):
    run_id = _start(client, models=["broken:model"])
    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "failed"
    assert body["sample_count"] == 0
    assert "2 of 2 model calls failed" in body["error"]


def test_parallel_mode_gives_same_results(client):
    run_id = _start(client, models=["gemma3:4b", "phi4"], parallel=True)
    results = client.get(f"/runs/{run_id}/results").json()
    assert sorted(r["model_name"] for r in results) == ["gemma3:4b", "gemma3:4b", "phi4", "phi4"]


def test_run_is_still_reachable_after_in_memory_registry_forgets_it(client):
    run_id = _start(client)
    run_store._runs.clear()  # simulate an API restart

    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "completed"
    assert body["dataset"] == "datasets/legal_qa"
    assert body["models"] == ["gemma3:4b"]
    assert body["sample_count"] == 2

    results = client.get(f"/runs/{run_id}/results").json()
    assert len(results) == 2
    assert results[0]["response_text"] == STUB_TEXT


def test_list_runs_newest_first(client):
    first = _start(client)
    second = _start(client, dataset="summarization")
    ids = [r["id"] for r in client.get("/runs").json()]
    assert ids[:2] == [second, first]


def test_results_for_unfinished_persisted_run_is_409(client):
    record = RunRecord(id="pending-run", dataset="datasets/legal_qa", models=["gemma3:4b"])
    record.status = RunStatus.RUNNING
    db.save_run(record)
    resp = client.get("/runs/pending-run/results")
    assert resp.status_code == 409


def test_restart_marks_interrupted_runs_failed():
    record = RunRecord(id="stuck", dataset="datasets/legal_qa", models=["gemma3:4b"])
    record.status = RunStatus.RUNNING
    record.started_at = datetime.now(timezone.utc)
    db.save_run(record)
    run_store._runs.clear()
    with TestClient(app) as c:  # lifespan startup runs here
        body = c.get("/runs/stuck").json()
    assert body["status"] == "failed"
    assert "interrupted" in body["error"]


def test_get_unknown_run_404(client):
    assert client.get("/runs/does-not-exist").status_code == 404
    assert client.get("/runs/does-not-exist/results").status_code == 404


def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("LLM_EVAL_API_KEY", "s3cret")
    assert client.post("/runs", json=RUN).status_code == 401
    assert client.post("/runs", json=RUN, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/runs", json=RUN, headers={"X-API-Key": "s3cret"}).status_code == 202
    # Reads stay open.
    assert client.get("/runs").status_code == 200


def test_cors_origins_from_env(monkeypatch):
    monkeypatch.setenv("LLM_EVAL_CORS_ORIGINS", "https://app.example.com/, https://other.example.com")
    assert cors_origins() == ["https://app.example.com", "https://other.example.com"]
    monkeypatch.delenv("LLM_EVAL_CORS_ORIGINS")
    assert "http://localhost:5173" in cors_origins()


def test_cors_preflight_allows_local_frontend(client):
    resp = client.options(
        "/runs",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_timeseries(client):
    assert client.get("/metrics/timeseries").json() == []

    run_id = _start(client)
    _start(client, models=["broken:model"])  # failed run: excluded

    points = client.get("/metrics/timeseries").json()
    assert [p["run_id"] for p in points] == [run_id]
    point = points[0]
    assert point["model_name"] == "gemma3:4b"
    assert point["dataset"] == "datasets/legal_qa"
    assert point["eval_version"] == EVAL_VERSION
    assert point["model_digest"] == "sha256:gemma3:4b"
    assert point["num_samples"] == 2
    assert point["rougeL_mean"] is not None
    assert point["pass_rate"] is None
    assert point["latency_p50"] == 1.0

    assert client.get("/metrics/timeseries", params={"model": "phi4"}).json() == []
    assert len(client.get("/metrics/timeseries", params={"eval_version": EVAL_VERSION}).json()) == 1
    assert client.get("/metrics/timeseries", params={"eval_version": 1}).json() == []
    assert client.get("/metrics/timeseries", params={"limit": 0}).status_code == 422
