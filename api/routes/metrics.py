from datetime import datetime, timezone
from typing import List, Optional

import mlflow
from fastapi import APIRouter

from ..schemas import TimeseriesPoint

router = APIRouter(prefix="/metrics", tags=["metrics"])

METRIC_SUFFIXES = ("_rouge1_mean", "_rougeL_mean", "_halluc_rate", "_latency_p50", "_latency_p95")


def _sanitize(model_name: str) -> str:
    return model_name.replace(":", "_").replace("/", "_")


@router.get("/timeseries", response_model=List[TimeseriesPoint])
async def get_timeseries(model: Optional[str] = None, limit: int = 100):
    """Aggregated per-model metrics across historical runs, sourced from MLflow."""
    client = mlflow.tracking.MlflowClient()
    points: List[TimeseriesPoint] = []

    try:
        experiments = client.search_experiments()
    except Exception:
        return points

    model_filter = _sanitize(model) if model else None

    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit,
        )
        for run in runs:
            metrics = run.data.metrics
            model_prefixes = {
                key[: -len(suffix)]
                for key in metrics
                for suffix in METRIC_SUFFIXES
                if key.endswith(suffix)
            }

            for prefix in model_prefixes:
                if model_filter and prefix != model_filter:
                    continue
                start_time = (
                    datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc)
                    if run.info.start_time
                    else None
                )
                points.append(
                    TimeseriesPoint(
                        run_id=run.info.run_id,
                        run_name=run.data.tags.get("mlflow.runName"),
                        model_name=prefix,
                        start_time=start_time,
                        rouge1_mean=metrics.get(f"{prefix}_rouge1_mean"),
                        rougeL_mean=metrics.get(f"{prefix}_rougeL_mean"),
                        halluc_rate=metrics.get(f"{prefix}_halluc_rate"),
                        latency_p50=metrics.get(f"{prefix}_latency_p50"),
                        latency_p95=metrics.get(f"{prefix}_latency_p95"),
                    )
                )

    points.sort(key=lambda p: p.start_time or datetime.min.replace(tzinfo=timezone.utc))
    return points
