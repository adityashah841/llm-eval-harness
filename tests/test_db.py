import json
import sqlite3

import api.db as db

V1_SCHEMA = """
CREATE TABLE runs (
    id TEXT PRIMARY KEY, dataset TEXT NOT NULL, models TEXT NOT NULL, status TEXT NOT NULL,
    created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT, error TEXT
);
CREATE TABLE sample_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id),
    sample_id TEXT NOT NULL, domain TEXT NOT NULL, prompt TEXT NOT NULL, expected TEXT NOT NULL,
    model_name TEXT NOT NULL, response_text TEXT NOT NULL, latency_ms REAL NOT NULL,
    input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, rouge1 REAL NOT NULL,
    rouge2 REAL NOT NULL, rougeL REAL NOT NULL, nli_label TEXT NOT NULL,
    nli_confidence REAL NOT NULL, hallucination_flag INTEGER NOT NULL
);
CREATE INDEX idx_sample_results_run_model ON sample_results(run_id, model_name);
"""


def _make_v1(path):
    conn = sqlite3.connect(path)
    conn.executescript(V1_SCHEMA)
    for run_id, has_samples in (("r1", True), ("r_empty", False)):
        conn.execute(
            "INSERT INTO runs VALUES (?, 'datasets/legal_qa', ?, 'completed', '2026-07-20T00:00:00+00:00', "
            "'2026-07-20T00:00:00+00:00', '2026-07-20T00:01:00+00:00', NULL)",
            (run_id, json.dumps(["gemma3:4b"])),
        )
        if has_samples:
            conn.execute(
                "INSERT INTO sample_results (run_id, sample_id, domain, prompt, expected, model_name, "
                "response_text, latency_ms, input_tokens, output_tokens, rouge1, rouge2, rougeL, nli_label, "
                "nli_confidence, hallucination_flag) VALUES ('r1', 'lq_001', 'legal', 'p', 'e', 'gemma3:4b', "
                "'resp', 1000.0, 10, 20, 0.1, 0.05, 0.08, 'neutral', 0.9, 0)"
            )
    conn.commit()
    conn.close()


def test_migrates_v1_database_in_place(tmp_path):
    path = tmp_path / "old.db"
    _make_v1(path)

    db.init_db(path)
    db.init_db(path)  # idempotent

    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()

    run = db.fetch_run("r1", db_path=path)
    assert run["eval_version"] == 1 and run["status"] == "completed" and run["sample_count"] == 1
    rows = db.fetch_run_results("r1", db_path=path)
    assert rows[0]["rougeL"] == 0.08 and rows[0]["hallucination_flag"] is False
    assert rows[0]["tests_passed"] is None and rows[0]["generation_ms"] is None

    # A legacy "completed" run with no results was really a total failure.
    empty = db.fetch_run("r_empty", db_path=path)
    assert empty["status"] == "failed" and "before failure tracking" in empty["error"]

    points = db.fetch_timeseries(db_path=path)
    assert len(points) == 1 and points[0]["eval_version"] == 1
    assert db.fetch_timeseries(eval_version=2, db_path=path) == []


def test_fresh_database(tmp_path):
    path = tmp_path / "new.db"
    db.init_db(path)
    assert db.list_runs(db_path=path) == []
    assert db.fetch_timeseries(db_path=path) == []
