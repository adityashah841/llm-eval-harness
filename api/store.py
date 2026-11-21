import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from llm_eval.runner.prompt_runner import EVAL_VERSION, EvalFailure, EvalResult


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"  # finished, but some model calls failed
    FAILED = "failed"


@dataclass
class RunRecord:
    id: str
    dataset: str
    models: List[str]
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    eval_version: int = EVAL_VERSION
    model_digests: Dict[str, Optional[str]] = field(default_factory=dict)
    results: List[EvalResult] = field(default_factory=list)
    failures: List[EvalFailure] = field(default_factory=list)


class RunStore:
    """In-memory registry of eval runs, keyed by run id.

    Scoped to a single API process. Every run is also written to the SQLite
    store (api/db.py) when it starts and when it finishes, which is what
    GET /runs falls back to after a restart.
    """

    def __init__(self):
        self._runs: Dict[str, RunRecord] = {}

    def create(self, dataset: str, models: List[str]) -> RunRecord:
        run_id = str(uuid.uuid4())
        record = RunRecord(id=run_id, dataset=dataset, models=models)
        self._runs[run_id] = record
        return record

    def get(self, run_id: str) -> Optional[RunRecord]:
        return self._runs.get(run_id)

    def all(self) -> List[RunRecord]:
        return list(self._runs.values())


run_store = RunStore()
