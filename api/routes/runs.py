import asyncio
import hmac
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query

from llm_eval.adapters import OllamaAdapter
from llm_eval.runner.dataset_loader import load_dataset
from llm_eval.runner.prompt_runner import PromptRunner

from ..config import api_key, max_concurrent_runs
from ..db import fetch_run, fetch_run_results, list_runs, save_run
from ..schemas import RunCreateRequest, RunSummary, SampleResult
from ..store import RunRecord, RunStatus, run_store
from .datasets import resolve_dataset

router = APIRouter(prefix="/runs", tags=["runs"])

# One semaphore per event loop (test clients may each run their own loop).
_semaphores: Dict[int, asyncio.Semaphore] = {}


def _run_slots() -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    if loop_id not in _semaphores:
        _semaphores[loop_id] = asyncio.Semaphore(max_concurrent_runs())
    return _semaphores[loop_id]


def _require_api_key(x_api_key: Optional[str]) -> None:
    expected = api_key()
    if expected is None:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


def _to_summary(record: RunRecord) -> RunSummary:
    return RunSummary(
        id=record.id,
        dataset=record.dataset,
        models=record.models,
        status=record.status.value,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        sample_count=len(record.results),
        error=record.error,
        eval_version=record.eval_version,
        model_digests=record.model_digests,
        failures=[
            {"sample_id": f.sample_id, "model_name": f.model_name, "error": f.error}
            for f in record.failures
        ],
    )


async def _execute_run(
    run_id: str, dataset_dir: str, models: List[str], smoke_test: bool, use_mlflow: bool, parallel: bool
) -> None:
    record = run_store.get(run_id)
    if record is None:
        return

    async with _run_slots():
        record.status = RunStatus.RUNNING
        record.started_at = datetime.now(timezone.utc)
        save_run(record)
        try:
            samples = load_dataset(dataset_dir)
            if smoke_test:
                samples = samples[:2]

            adapters = [OllamaAdapter(m) for m in models]
            runner = PromptRunner(adapters, use_mlflow=use_mlflow, parallel=parallel)
            outcome = await runner.run(samples, run_name=run_id)
            record.results = outcome.results
            record.failures = outcome.failures
            record.model_digests = {r.model_name: r.model_digest for r in outcome.results}
            record.status = RunStatus(outcome.status)
            record.error = outcome.failure_summary()
        except Exception as exc:
            record.status = RunStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
        finally:
            record.completed_at = datetime.now(timezone.utc)
            save_run(record)


@router.post("", response_model=RunSummary, status_code=202)
async def create_run(
    payload: RunCreateRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(default=None),
):
    _require_api_key(x_api_key)
    dataset_key, dataset_dir = resolve_dataset(payload.dataset)

    record = run_store.create(dataset=dataset_key, models=list(dict.fromkeys(payload.models)))
    save_run(record)
    background_tasks.add_task(
        _execute_run,
        record.id,
        str(dataset_dir),
        record.models,
        payload.smoke_test,
        payload.use_mlflow,
        payload.parallel,
    )
    return _to_summary(record)


@router.get("", response_model=List[RunSummary])
async def get_runs(limit: int = Query(50, ge=1, le=500)):
    """Most recent runs, newest first (from the persistent store, with live
    status for runs this process is executing)."""
    summaries = []
    for row in list_runs(limit=limit):
        live = run_store.get(row["id"])
        summaries.append(_to_summary(live) if live else RunSummary(**row))
    return summaries


@router.get("/{run_id}", response_model=RunSummary)
async def get_run(run_id: str):
    record = run_store.get(run_id)
    if record is not None:
        return _to_summary(record)

    # Falls back to the persistent store once the in-memory run registry has
    # forgotten this run (e.g. after an API process restart).
    persisted = fetch_run(run_id)
    if persisted is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunSummary(**persisted)


@router.get("/{run_id}/results", response_model=List[SampleResult])
async def get_run_results(run_id: str):
    record = run_store.get(run_id)
    if record is not None:
        if record.status in (RunStatus.PENDING, RunStatus.RUNNING):
            raise HTTPException(
                status_code=409, detail=f"Run is {record.status.value}, results not ready yet"
            )
        return [SampleResult(**r.__dict__) for r in record.results]

    persisted = fetch_run(run_id)
    if persisted is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if persisted["status"] in ("pending", "running"):
        raise HTTPException(status_code=409, detail=f"Run is {persisted['status']}, results not ready yet")
    return [SampleResult(**row) for row in fetch_run_results(run_id)]
