from typing import List, Optional

from fastapi import APIRouter, Query

from ..db import fetch_timeseries
from ..schemas import TimeseriesPoint

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/timeseries", response_model=List[TimeseriesPoint])
async def get_timeseries(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    eval_version: Optional[int] = Query(
        None, description="Only runs scored with this methodology version (see EVAL_VERSION)"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Most recent runs to include"),
):
    """Per-(run, model) aggregates across historical runs, oldest first,
    sourced from the persistent SQLite store. Failed runs are excluded."""
    points = fetch_timeseries(model=model, dataset=dataset, limit=limit, eval_version=eval_version)
    return [TimeseriesPoint(**p) for p in points]
