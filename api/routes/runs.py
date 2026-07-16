from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException

from llm_eval.adapters import OllamaAdapter
from llm_eval.runner.dataset_loader import load_dataset
from llm_eval.runner.prompt_runner import PromptRunner

from ..db import save_run
from ..schemas import RunCreateRequest, RunSummary, SampleResult
from ..store import RunRecord, RunStatus, run_store

router = APIRouter(prefix="/runs", tags=["runs"])


def _resolve_dataset(dataset: str) -> Path:
    p = Path(dataset)
    if p.exists():
        return p
    candidate = Path("datasets") / dataset
    if candidate.exists():
        return candidate
    raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")


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
    )


async def _execute_run(
    run_id: str, dataset_path: str, models: List[str], smoke_test: bool, use_mlflow: bool
) -> None:
    record = run_store.get(run_id)
    if record is None:
        return

    record.status = RunStatus.RUNNING
    record.started_at = datetime.now(timezone.utc)
    try:
        samples = load_dataset(dataset_path)
        if smoke_test:
            samples = samples[:2]

        adapters = [OllamaAdapter(m) for m in models]
        runner = PromptRunner(adapters, use_mlflow=use_mlflow)
        record.results = await runner.run(samples, run_name=run_id)
        record.status = RunStatus.COMPLETED
    except Exception as exc:
        record.status = RunStatus.FAILED
        record.error = str(exc)
    finally:
        record.completed_at = datetime.now(timezone.utc)
        save_run(record)


@router.post("", response_model=RunSummary, status_code=202)
async def create_run(payload: RunCreateRequest, background_tasks: BackgroundTasks):
    dataset_path = str(_resolve_dataset(payload.dataset))

    record = run_store.create(dataset=dataset_path, models=list(payload.models))
    background_tasks.add_task(
        _execute_run, record.id, dataset_path, list(payload.models), payload.smoke_test, payload.use_mlflow
    )
    return _to_summary(record)


@router.get("/{run_id}", response_model=RunSummary)
async def get_run(run_id: str):
    record = run_store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _to_summary(record)


@router.get("/{run_id}/results", response_model=List[SampleResult])
async def get_run_results(run_id: str):
    record = run_store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if record.status in (RunStatus.PENDING, RunStatus.RUNNING):
        raise HTTPException(
            status_code=409, detail=f"Run is {record.status.value}, results not ready yet"
        )

    return [
        SampleResult(
            sample_id=r.sample_id,
            domain=r.domain,
            prompt=r.prompt,
            expected=r.expected,
            model_name=r.model_name,
            response_text=r.response_text,
            latency_ms=r.latency_ms,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            rouge1=r.rouge1,
            rouge2=r.rouge2,
            rougeL=r.rougeL,
            nli_label=r.nli_label,
            nli_confidence=r.nli_confidence,
            hallucination_flag=r.hallucination_flag,
        )
        for r in record.results
    ]
