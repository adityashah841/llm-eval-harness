"""SQLite persistence for runs and per-sample results.

Schema versions (tracked with PRAGMA user_version):
  0/1 — original layout (every metric NOT NULL, no eval_version).
  2   — nullable metrics (code samples have no ROUGE/NLI, text samples have no
        test result), eval_version + model digests + failures on runs, and
        server-side timing columns. Existing rows are migrated in place and
        tagged eval_version=1 so they never mix with new-methodology runs.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

from llm_eval.runner.summary import aggregate_rows

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "metrics.db"
DB_PATH = Path(os.environ.get("LLM_EVAL_DB_PATH", DEFAULT_DB_PATH))

SCHEMA_VERSION = 2

RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    models TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error TEXT,
    eval_version INTEGER NOT NULL DEFAULT 1,
    model_digests TEXT,
    failures TEXT
)
"""

SAMPLES_DDL = """
CREATE TABLE IF NOT EXISTS sample_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    sample_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    prompt TEXT NOT NULL,
    expected TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_digest TEXT,
    response_text TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    load_ms REAL,
    generation_ms REAL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    rouge1 REAL,
    rouge2 REAL,
    rougeL REAL,
    nli_label TEXT,
    nli_confidence REAL,
    entailment_rate REAL,
    contradiction_rate REAL,
    hallucination_flag INTEGER,
    tests_passed INTEGER,
    exec_error TEXT
)
"""

SAMPLES_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_sample_results_run_model
    ON sample_results(run_id, model_name)
"""

# Columns shared by the v1 and v2 sample_results layouts.
_V1_SAMPLE_COLUMNS = (
    "id, run_id, sample_id, domain, prompt, expected, model_name, response_text, "
    "latency_ms, input_tokens, output_tokens, rouge1, rouge2, rougeL, nli_label, "
    "nli_confidence, hallucination_flag"
)

SAMPLE_FIELDS = [
    "sample_id", "domain", "prompt", "expected", "model_name", "model_digest",
    "response_text", "latency_ms", "load_ms", "generation_ms", "input_tokens",
    "output_tokens", "rouge1", "rouge2", "rougeL", "nli_label", "nli_confidence",
    "entailment_rate", "contradiction_rate", "hallucination_flag", "tests_passed",
    "exec_error",
]
_BOOL_FIELDS = {"hallucination_flag", "tests_passed"}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= SCHEMA_VERSION:
        return

    legacy = _table_exists(conn, "runs") and version < 2
    conn.execute("BEGIN")
    try:
        if legacy:
            existing = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
            for col, ddl in (
                ("eval_version", "INTEGER NOT NULL DEFAULT 1"),
                ("model_digests", "TEXT"),
                ("failures", "TEXT"),
            ):
                if col not in existing:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {ddl}")
            if _table_exists(conn, "sample_results"):
                conn.execute("DROP INDEX IF EXISTS idx_sample_results_run_model")
                conn.execute("ALTER TABLE sample_results RENAME TO sample_results_v1")
                conn.execute(SAMPLES_DDL)
                conn.execute(
                    f"INSERT INTO sample_results ({_V1_SAMPLE_COLUMNS}) "
                    f"SELECT {_V1_SAMPLE_COLUMNS} FROM sample_results_v1"
                )
                conn.execute("DROP TABLE sample_results_v1")
                # Before failure tracking, a run where every model call failed
                # was still saved as 'completed' with zero results.
                conn.execute(
                    "UPDATE runs SET status='failed', error=COALESCE(error, "
                    "'no model call succeeded (recorded before failure tracking existed)') "
                    "WHERE status='completed' AND NOT EXISTS "
                    "(SELECT 1 FROM sample_results s WHERE s.run_id = runs.id)"
                )
        for stmt in (RUNS_DDL, SAMPLES_DDL, SAMPLES_INDEX_DDL):
            conn.execute(stmt)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def init_db(db_path: Optional[Path] = None) -> None:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        _migrate(conn)
    finally:
        conn.close()


@contextmanager
def get_connection(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or DB_PATH)
    init_db(path)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


def save_run(record, db_path: Optional[Path] = None) -> None:
    """Persist a run (api.store.RunRecord) and its sample results."""
    failures = [
        {"sample_id": f.sample_id, "model_name": f.model_name, "error": f.error}
        for f in record.failures
    ]
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO runs (id, dataset, models, status, created_at, started_at,
                                 completed_at, error, eval_version, model_digests, failures)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   status=excluded.status,
                   started_at=excluded.started_at,
                   completed_at=excluded.completed_at,
                   error=excluded.error,
                   eval_version=excluded.eval_version,
                   model_digests=excluded.model_digests,
                   failures=excluded.failures""",
            (
                record.id,
                record.dataset,
                json.dumps(record.models),
                record.status.value,
                _iso(record.created_at),
                _iso(record.started_at),
                _iso(record.completed_at),
                record.error,
                record.eval_version,
                json.dumps(record.model_digests),
                json.dumps(failures),
            ),
        )
        conn.execute("DELETE FROM sample_results WHERE run_id = ?", (record.id,))
        placeholders = ", ".join("?" for _ in range(len(SAMPLE_FIELDS) + 1))
        conn.executemany(
            f"INSERT INTO sample_results (run_id, {', '.join(SAMPLE_FIELDS)}) VALUES ({placeholders})",
            [
                (record.id, *[
                    (None if getattr(r, f) is None else int(getattr(r, f)))
                    if f in _BOOL_FIELDS else getattr(r, f)
                    for f in SAMPLE_FIELDS
                ])
                for r in record.results
            ],
        )
        conn.commit()


def mark_interrupted_runs(db_path: Optional[Path] = None) -> int:
    """Runs left pending/running by a previous process can never finish;
    mark them failed so they don't show as in-progress forever."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "UPDATE runs SET status='failed', error=COALESCE(error, 'interrupted: API process restarted') "
            "WHERE status IN ('pending', 'running')"
        )
        conn.commit()
        return cur.rowcount


def _run_row_to_dict(row: sqlite3.Row, sample_count: int) -> dict:
    return {
        "id": row["id"],
        "dataset": row["dataset"],
        "models": json.loads(row["models"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "sample_count": sample_count,
        "error": row["error"],
        "eval_version": row["eval_version"],
        "model_digests": json.loads(row["model_digests"]) if row["model_digests"] else {},
        "failures": json.loads(row["failures"]) if row["failures"] else [],
    }


def fetch_run(run_id: str, db_path: Optional[Path] = None) -> Optional[dict]:
    """A run's summary from the persistent store — the fallback for
    GET /runs/{id} once the in-memory run registry has forgotten it (e.g.
    after an API process restart)."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        count = conn.execute(
            "SELECT COUNT(*) FROM sample_results WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    return _run_row_to_dict(row, count)


def list_runs(limit: int = 50, db_path: Optional[Path] = None) -> List[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT r.*, (SELECT COUNT(*) FROM sample_results s WHERE s.run_id = r.id) AS n
               FROM runs r ORDER BY r.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [_run_row_to_dict(r, r["n"]) for r in rows]


def _sample_row_to_dict(r: sqlite3.Row) -> dict:
    d = {f: r[f] for f in SAMPLE_FIELDS}
    for f in _BOOL_FIELDS:
        if d[f] is not None:
            d[f] = bool(d[f])
    return d


def fetch_run_results(run_id: str, db_path: Optional[Path] = None) -> List[dict]:
    """Per-sample results for a run from the persistent store — the
    fallback for GET /runs/{id}/results."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT {', '.join(SAMPLE_FIELDS)} FROM sample_results WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
    return [_sample_row_to_dict(r) for r in rows]


def fetch_timeseries(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    limit: int = 100,
    eval_version: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """Per (run, model) aggregates for the most recent `limit` runs that
    produced results, oldest first — the source for GET /metrics/timeseries."""
    clauses = ["r.status IN ('completed', 'partial')"]
    params: list = []
    if model:
        clauses.append("EXISTS (SELECT 1 FROM sample_results x WHERE x.run_id = r.id AND x.model_name = ?)")
        params.append(model)
    if dataset:
        clauses.append("r.dataset = ?")
        params.append(dataset)
    if eval_version is not None:
        clauses.append("r.eval_version = ?")
        params.append(eval_version)
    where = " AND ".join(clauses)

    with get_connection(db_path) as conn:
        runs = conn.execute(
            f"""SELECT r.id, r.dataset, r.completed_at, r.eval_version FROM runs r
                WHERE {where} ORDER BY r.completed_at DESC LIMIT ?""",
            [*params, limit],
        ).fetchall()
        if not runs:
            return []
        run_info = {r["id"]: r for r in runs}
        marks = ", ".join("?" for _ in run_info)
        model_clause = "AND model_name = ?" if model else ""
        rows = conn.execute(
            f"""SELECT run_id, {', '.join(SAMPLE_FIELDS)} FROM sample_results
                WHERE run_id IN ({marks}) {model_clause} ORDER BY id""",
            [*run_info.keys(), *([model] if model else [])],
        ).fetchall()

    grouped: dict = {}
    for row in rows:
        grouped.setdefault((row["run_id"], row["model_name"]), []).append(_sample_row_to_dict(row))

    points = []
    for (run_id, model_name), group in grouped.items():
        info = run_info[run_id]
        digests = {g["model_digest"] for g in group if g["model_digest"]}
        points.append(
            {
                "run_id": run_id,
                "run_name": None,
                "dataset": info["dataset"],
                "model_name": model_name,
                "model_digest": next(iter(digests)) if len(digests) == 1 else None,
                "eval_version": info["eval_version"],
                "start_time": info["completed_at"],
                **aggregate_rows(group),
            }
        )
    points.sort(key=lambda p: (p["start_time"] or "", p["model_name"]))
    return points
