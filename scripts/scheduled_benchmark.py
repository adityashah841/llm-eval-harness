#!/usr/bin/env python3
"""
Re-runs a fixed benchmark and persists the results to the metrics DB.

Invoked daily by .github/workflows/scheduled-benchmark.yml so
GET /metrics/timeseries has a real time series to serve. Every dataset is run
in full with one fixed model and greedy decoding, so day-to-day differences in
scores point at a change in the model, Ollama or the harness rather than
sampling noise. (Latency on shared CI runners varies with the machine, so
treat latency trends from this job as indicative only.)

Exits non-zero if any run failed or was partial, so the workflow shows red.
"""
import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.db import save_run  # noqa: E402
from api.store import RunRecord, RunStatus  # noqa: E402
from llm_eval.adapters import OllamaAdapter  # noqa: E402
from llm_eval.runner.dataset_loader import load_dataset  # noqa: E402
from llm_eval.runner.prompt_runner import PromptRunner  # noqa: E402

DATASETS = ["datasets/legal_qa", "datasets/summarization", "datasets/code_gen"]
MODELS = ["gemma3:4b"]


async def run_one(dataset: str) -> RunStatus:
    record = RunRecord(id=str(uuid.uuid4()), dataset=dataset, models=MODELS)
    record.status = RunStatus.RUNNING
    record.started_at = datetime.now(timezone.utc)

    try:
        samples = load_dataset(dataset)
        runner = PromptRunner([OllamaAdapter(m) for m in MODELS], use_mlflow=False)
        outcome = await runner.run(samples, run_name="scheduled_benchmark")
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
        print(
            f"{dataset}: run {record.id} {record.status.value}, "
            f"{len(record.results)} result(s), {len(record.failures)} failure(s)"
            + (f" — {record.error}" if record.error else "")
        )
    return record.status


async def main() -> int:
    statuses = [await run_one(d) for d in DATASETS]
    return 0 if all(s == RunStatus.COMPLETED for s in statuses) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
