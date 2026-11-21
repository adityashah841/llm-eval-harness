"""Aggregation shared by the CLI, Streamlit dashboard, MLflow logging and API.

Keeping one implementation means every surface reports the same number for
the same data (e.g. p50 is always the linearly interpolated median).
"""
from typing import Dict, Iterable, List, Optional, Sequence


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile (same as numpy's default)."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def _mean(values: Iterable) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _r(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(value, digits)


def aggregate_rows(rows: List[dict]) -> dict:
    """Aggregate per-sample result dicts (all for one model) into summary stats.

    Metrics that do not apply to a sample (ROUGE/NLI for code, pass/fail for
    text) are stored as None and excluded, so a code dataset reports no
    hallucination rate rather than a misleading 0%.
    """
    latencies = [r.get("latency_ms") for r in rows]
    generation = [r.get("generation_ms") for r in rows]
    return {
        "num_samples": len(rows),
        "rouge1_mean": _r(_mean(r.get("rouge1") for r in rows), 4),
        "rougeL_mean": _r(_mean(r.get("rougeL") for r in rows), 4),
        "halluc_rate": _r(_mean(r.get("hallucination_flag") for r in rows), 4),
        "entailment_rate_mean": _r(_mean(r.get("entailment_rate") for r in rows), 4),
        "pass_rate": _r(_mean(r.get("tests_passed") for r in rows), 4),
        "latency_p50": _r(percentile(latencies, 50), 2),
        "latency_p95": _r(percentile(latencies, 95), 2),
        "generation_p50": _r(percentile(generation, 50), 2),
    }


def summarize_by_model(rows: List[dict]) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}
    for r in rows:
        grouped.setdefault(r["model_name"], []).append(r)
    return {model: aggregate_rows(group) for model, group in grouped.items()}
