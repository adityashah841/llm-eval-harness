import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getRun, getRunResults } from '../api/client';
import type { RunSummary, SampleResult } from '../api/types';
import { useRuns } from '../context/RunsContext';

const SERIES_COLORS = [
  'var(--series-1)',
  'var(--series-2)',
  'var(--series-3)',
  'var(--series-4)',
  'var(--series-5)',
  'var(--series-6)',
  'var(--series-7)',
  'var(--series-8)',
];

interface ModelAggregate {
  model: string;
  rouge1: number;
  rougeL: number;
  hallucRate: number;
  latencyP50: number;
  count: number;
}

function aggregate(results: SampleResult[]): ModelAggregate[] {
  const byModel = new Map<string, SampleResult[]>();
  for (const r of results) {
    const list = byModel.get(r.model_name) ?? [];
    list.push(r);
    byModel.set(r.model_name, list);
  }

  return Array.from(byModel.entries()).map(([model, rows]) => {
    const mean = (fn: (r: SampleResult) => number) => rows.reduce((sum, r) => sum + fn(r), 0) / rows.length;
    const latencies = rows.map((r) => r.latency_ms).sort((a, b) => a - b);
    const p50 = latencies[Math.floor(latencies.length / 2)] ?? 0;
    return {
      model,
      rouge1: mean((r) => r.rouge1),
      rougeL: mean((r) => r.rougeL),
      hallucRate: mean((r) => (r.hallucination_flag ? 1 : 0)),
      latencyP50: p50,
      count: rows.length,
    };
  });
}

export default function ComparisonPage() {
  const { recentRuns } = useRuns();
  const [searchParams, setSearchParams] = useSearchParams();
  const runId = searchParams.get('run') ?? '';

  const [run, setRun] = useState<RunSummary | null>(null);
  const [results, setResults] = useState<SampleResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) {
      setRun(null);
      setResults([]);
      return;
    }
    setError(null);
    Promise.all([getRun(runId), getRunResults(runId)])
      .then(([runData, resultsData]) => {
        setRun(runData);
        setResults(resultsData);
      })
      .catch((err: Error) => setError(err.message));
  }, [runId]);

  const rows = useMemo(() => aggregate(results), [results]);
  const maxRougeL = Math.max(0.0001, ...rows.map((r) => r.rougeL));

  return (
    <div>
      <h1>Model comparison</h1>
      <p>Side-by-side scores for every model evaluated in the same run.</p>

      <div className="card">
        <div className="field-row">
          <div className="field">
            <label htmlFor="compare-run-select">Run</label>
            <select
              id="compare-run-select"
              value={runId}
              onChange={(e) => setSearchParams(e.target.value ? { run: e.target.value } : {})}
            >
              <option value="">Select a run…</option>
              {recentRuns
                .filter((r) => r.models.length > 1)
                .map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.dataset} · {r.models.join('+')} · {new Date(r.created_at).toLocaleString()}
                  </option>
                ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="compare-run-id-input">or paste a run id</label>
            <input
              id="compare-run-id-input"
              value={runId}
              placeholder="run id"
              onChange={(e) => setSearchParams(e.target.value ? { run: e.target.value } : {})}
            />
          </div>
        </div>
        {run ? (
          <p style={{ fontSize: 13 }}>
            Dataset: {run.dataset} · Status: {run.status}
          </p>
        ) : (
          <p style={{ fontSize: 13 }}>
            Only runs with 2+ models are useful here — trigger one from{' '}
            <a href="/">Run configuration</a> if the list above is empty.
          </p>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {rows.length > 0 && (
        <>
          <div className="card">
            <h3>ROUGE-L mean</h3>
            {rows.map((row, i) => (
              <div className="bar-row" key={row.model}>
                <span>{row.model}</span>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${(row.rougeL / maxRougeL) * 100}%`,
                      background: SERIES_COLORS[i % SERIES_COLORS.length],
                    }}
                  />
                </div>
                <span className="bar-value">{row.rougeL.toFixed(3)}</span>
              </div>
            ))}
          </div>

          <div className="card">
            <h3>All metrics</h3>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Samples</th>
                  <th>ROUGE-1</th>
                  <th>ROUGE-L</th>
                  <th>Hallucination rate</th>
                  <th>Latency p50 (ms)</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.model}>
                    <td>{row.model}</td>
                    <td>{row.count}</td>
                    <td>{row.rouge1.toFixed(3)}</td>
                    <td>{row.rougeL.toFixed(3)}</td>
                    <td>{(row.hallucRate * 100).toFixed(0)}%</td>
                    <td>{row.latencyP50.toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!runId && <div className="empty-state">Pick a run above to compare its models.</div>}
      {runId && rows.length === 0 && !error && <p>Loading…</p>}
    </div>
  );
}
