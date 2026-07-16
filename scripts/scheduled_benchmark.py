#!/usr/bin/env python3
"""
Re-runs a fixed, small benchmark and persists the results to the metrics DB.

Invoked on a cron schedule by .github/workflows/scheduled-benchmark.yml so
GET /metrics/timeseries has a real time series to serve, built from repeated
runs over time, rather than only from runs triggered ad hoc through the API.
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

# Kept small and fixed so the scheduled job stays fast and comparable across runs.
DATASET = "datasets/legal_qa"
MODELS = ["gemma3:4b"]
SAMPLE_LIMIT = 2


async def main() -> None:
    record = RunRecord(id=str(uuid.uuid4()), dataset=DATASET, models=MODELS)
    record.status = RunStatus.RUNNING
    record.started_at = datetime.now(timezone.utc)

    try:
        samples = load_dataset(DATASET)[:SAMPLE_LIMIT]
        adapters = [OllamaAdapter(m) for m in MODELS]
        runner = PromptRunner(adapters, use_mlflow=False)
        record.results = await runner.run(samples, run_name="scheduled_benchmark")
        record.status = RunStatus.COMPLETED
    except Exception as exc:
        record.status = RunStatus.FAILED
        record.error = str(exc)
        raise
    finally:
        record.completed_at = datetime.now(timezone.utc)
        save_run(record)
        print(f"Saved scheduled benchmark run {record.id}: "
              f"{record.status.value}, {len(record.results)} result(s).")


if __name__ == "__main__":
    asyncio.run(main())
