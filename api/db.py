import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "metrics.db"
DB_PATH = Path(os.environ.get("LLM_EVAL_DB_PATH", DEFAULT_DB_PATH))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    models TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS sample_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    sample_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    prompt TEXT NOT NULL,
    expected TEXT NOT NULL,
    model_name TEXT NOT NULL,
    response_text TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    rouge1 REAL NOT NULL,
    rouge2 REAL NOT NULL,
    rougeL REAL NOT NULL,
    nli_label TEXT NOT NULL,
    nli_confidence REAL NOT NULL,
    hallucination_flag INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sample_results_run_model
    ON sample_results(run_id, model_name);
"""


def init_db(db_path: Optional[Path] = None) -> None:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_connection(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = db_path or DB_PATH
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def save_run(record, db_path: Optional[Path] = None) -> None:
    """Persist a run (api.store.RunRecord) and its sample results."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO runs (id, dataset, models, status, created_at, started_at, completed_at, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   status=excluded.status,
                   started_at=excluded.started_at,
                   completed_at=excluded.completed_at,
                   error=excluded.error""",
            (
                record.id,
                record.dataset,
                json.dumps(record.models),
                record.status.value,
                record.created_at.isoformat(),
                record.started_at.isoformat() if record.started_at else None,
                record.completed_at.isoformat() if record.completed_at else None,
                record.error,
            ),
        )
        conn.execute("DELETE FROM sample_results WHERE run_id = ?", (record.id,))
        conn.executemany(
            """INSERT INTO sample_results
               (run_id, sample_id, domain, prompt, expected, model_name, response_text,
                latency_ms, input_tokens, output_tokens, rouge1, rouge2, rougeL,
                nli_label, nli_confidence, hallucination_flag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    record.id,
                    r.sample_id,
                    r.domain,
                    r.prompt,
                    r.expected,
                    r.model_name,
                    r.response_text,
                    r.latency_ms,
                    r.input_tokens,
                    r.output_tokens,
                    r.rouge1,
                    r.rouge2,
                    r.rougeL,
                    r.nli_label,
                    r.nli_confidence,
                    int(r.hallucination_flag),
                )
                for r in record.results
            ],
        )
        conn.commit()


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def fetch_timeseries(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """Per (run, model) aggregates, most recent first — the source for GET /metrics/timeseries."""
    clauses = []
    params: List[str] = []
    if model:
        clauses.append("sr.model_name = ?")
        params.append(model)
    if dataset:
        clauses.append("r.dataset = ?")
        params.append(dataset)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT r.id AS run_id, r.dataset AS dataset, r.completed_at AS completed_at,
                   sr.model_name AS model_name, sr.rouge1 AS rouge1, sr.rougeL AS rougeL,
                   sr.hallucination_flag AS hallucination_flag, sr.latency_ms AS latency_ms
            FROM sample_results sr
            JOIN runs r ON r.id = sr.run_id
            {where}
            ORDER BY r.completed_at DESC
            """,
            params,
        ).fetchall()

    grouped: dict = {}
    for row in rows:
        key = (row["run_id"], row["model_name"])
        grouped.setdefault(key, []).append(row)

    points = []
    for (run_id, model_name), group_rows in grouped.items():
        latencies = sorted(r["latency_ms"] for r in group_rows)
        points.append(
            {
                "run_id": run_id,
                "run_name": None,
                "dataset": group_rows[0]["dataset"],
                "model_name": model_name,
                "start_time": group_rows[0]["completed_at"],
                "rouge1_mean": round(mean(r["rouge1"] for r in group_rows), 4),
                "rougeL_mean": round(mean(r["rougeL"] for r in group_rows), 4),
                "halluc_rate": round(mean(r["hallucination_flag"] for r in group_rows), 4),
                "latency_p50": round(_percentile(latencies, 50), 2),
                "latency_p95": round(_percentile(latencies, 95), 2),
            }
        )

    points.sort(key=lambda p: p["start_time"] or "")
    return points[-limit:] if limit else points
