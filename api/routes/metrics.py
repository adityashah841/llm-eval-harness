from typing import List, Optional

from fastapi import APIRouter

from ..db import fetch_timeseries
from ..schemas import TimeseriesPoint

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/timeseries", response_model=List[TimeseriesPoint])
async def get_timeseries(model: Optional[str] = None, dataset: Optional[str] = None, limit: int = 100):
    """Aggregated per-model metrics across historical runs, sourced from the
    persistent SQLite store (populated by every API-triggered run and by the
    scheduled benchmark job)."""
    points = fetch_timeseries(model=model, dataset=dataset, limit=limit)
    return [TimeseriesPoint(**p) for p in points]
