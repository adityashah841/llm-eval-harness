import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { getRun, getRunResults } from '../api/client';
import type { RunSummary, SampleResult } from '../api/types';
import { useRuns } from '../context/RunsContext';

type FailureFilter = 'all' | 'hallucinated' | 'clean';

export default function ResultsPage() {
  const { recentRuns } = useRuns();
  const [searchParams, setSearchParams] = useSearchParams();
  const runId = searchParams.get('run') ?? '';

  const [run, setRun] = useState<RunSummary | null>(null);
  const [results, setResults] = useState<SampleResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [modelFilter, setModelFilter] = useState<string>('all');
  const [failureFilter, setFailureFilter] = useState<FailureFilter>('all');

  useEffect(() => {
    if (!runId) {
      setRun(null);
      setResults([]);
      return;
    }
    setLoading(true);
    setError(null);
    Promise.all([getRun(runId), getRunResults(runId)])
      .then(([runData, resultsData]) => {
        setRun(runData);
        setResults(resultsData);
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [runId]);

  const modelOptions = useMemo(
    () => Array.from(new Set(results.map((r) => r.model_name))),
    [results]
  );

  const filtered = results.filter((r) => {
    if (modelFilter !== 'all' && r.model_name !== modelFilter) return false;
    if (failureFilter === 'hallucinated' && !r.hallucination_flag) return false;
    if (failureFilter === 'clean' && r.hallucination_flag) return false;
    return true;
  });

  return (
    <div>
      <h1>Results</h1>
      <p>Per-sample scores for a completed evaluation run.</p>

      <div className="card">
        <div className="field-row">
          <div className="field">
            <label htmlFor="run-select">Run</label>
            <select
              id="run-select"
              value={runId}
              onChange={(e) => setSearchParams(e.target.value ? { run: e.target.value } : {})}
            >
              <option value="">Select a run…</option>
              {recentRuns.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.dataset} · {r.models.join('+')} · {new Date(r.created_at).toLocaleString()}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="run-id-input">or paste a run id</label>
            <input
              id="run-id-input"
              value={runId}
              placeholder="run id"
              onChange={(e) => setSearchParams(e.target.value ? { run: e.target.value } : {})}
            />
          </div>
        </div>

        {run && (
          <p>
            Status: {run.status} · {run.sample_count} sample(s) · {run.models.join(', ')}
          </p>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && <p>Loading…</p>}

      {results.length > 0 && (
        <div className="card">
          <div className="filter-row">
            <div className="field">
              <label htmlFor="model-filter">Model</label>
              <select id="model-filter" value={modelFilter} onChange={(e) => setModelFilter(e.target.value)}>
                <option value="all">All models</option>
                {modelOptions.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="failure-filter">Failure type</label>
              <select
                id="failure-filter"
                value={failureFilter}
                onChange={(e) => setFailureFilter(e.target.value as FailureFilter)}
              >
                <option value="all">All results</option>
                <option value="hallucinated">Hallucinations only</option>
                <option value="clean">No hallucination</option>
              </select>
            </div>
          </div>

          <table className="data-table">
            <thead>
              <tr>
                <th>Sample</th>
                <th>Model</th>
                <th>ROUGE-L</th>
                <th>Hallucination</th>
                <th>Latency (ms)</th>
                <th>Response</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => (
                <tr key={`${r.sample_id}-${r.model_name}-${i}`}>
                  <td>{r.sample_id}</td>
                  <td>{r.model_name}</td>
                  <td>{r.rougeL.toFixed(3)}</td>
                  <td>
                    <span className={`pill ${r.hallucination_flag ? 'pill-critical' : 'pill-good'}`}>
                      {r.hallucination_flag ? 'flagged' : 'clean'}
                    </span>
                  </td>
                  <td>{r.latency_ms.toFixed(0)}</td>
                  <td style={{ maxWidth: 360 }}>{r.response_text}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && <p className="empty-state">No results match this filter.</p>}
        </div>
      )}

      {!runId && !loading && <div className="empty-state">Pick a run above to see its results.</div>}
    </div>
  );
}
