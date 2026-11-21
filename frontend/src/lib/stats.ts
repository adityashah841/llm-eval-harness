// Mirrors llm_eval/runner/summary.py so the UI and the API report the same
// numbers for the same data.
import type { SampleResult } from '../api/types';

export function percentile(values: number[], pct: number): number | null {
  const vals = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (vals.length === 0) return null;
  const k = (vals.length - 1) * (pct / 100);
  const f = Math.floor(k);
  const c = Math.min(f + 1, vals.length - 1);
  return vals[f] + (vals[c] - vals[f]) * (k - f);
}

export function meanOf<T>(rows: T[], pick: (r: T) => number | boolean | null | undefined): number | null {
  const vals = rows
    .map(pick)
    .filter((v): v is number | boolean => v !== null && v !== undefined)
    .map((v) => (typeof v === 'boolean' ? (v ? 1 : 0) : v));
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
}

export interface ModelAggregate {
  model: string;
  count: number;
  rouge1: number | null;
  rougeL: number | null;
  entailmentRate: number | null;
  hallucRate: number | null;
  passRate: number | null;
  latencyP50: number | null;
  latencyP95: number | null;
}

export function aggregateByModel(results: SampleResult[]): ModelAggregate[] {
  const byModel = new Map<string, SampleResult[]>();
  for (const r of results) {
    byModel.set(r.model_name, [...(byModel.get(r.model_name) ?? []), r]);
  }
  return Array.from(byModel.entries()).map(([model, rows]) => {
    const latencies = rows.map((r) => r.latency_ms);
    return {
      model,
      count: rows.length,
      rouge1: meanOf(rows, (r) => r.rouge1),
      rougeL: meanOf(rows, (r) => r.rougeL),
      entailmentRate: meanOf(rows, (r) => r.entailment_rate),
      hallucRate: meanOf(rows, (r) => r.hallucination_flag),
      passRate: meanOf(rows, (r) => r.tests_passed),
      latencyP50: percentile(latencies, 50),
      latencyP95: percentile(latencies, 95),
    };
  });
}

export const fmtScore = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(3));
export const fmtPct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`);
export const fmtMs = (v: number | null | undefined) =>
  v == null ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${v.toFixed(0)}ms`;
