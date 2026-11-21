import { useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import RunFailures from '../components/RunFailures';
import RunPicker from '../components/RunPicker';
import StatusPill from '../components/StatusPill';
import { useRunData } from '../hooks/useRunData';
import { aggregateByModel, fmtMs, fmtPct, fmtScore } from '../lib/stats';

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

export default function ComparisonPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const runId = searchParams.get('run') ?? '';
  const { run, results, loading, error } = useRunData(runId);

  const rows = useMemo(() => aggregateByModel(results), [results]);
  const isCode = rows.length > 0 && rows.every((r) => r.passRate !== null);
  const headline = (r: (typeof rows)[number]) => (isCode ? r.passRate : r.rougeL) ?? 0;
  const maxHeadline = Math.max(0.0001, ...rows.map(headline));

  return (
    <div>
      <h1>Model comparison</h1>
      <p>Side-by-side scores for every model evaluated in the same run.</p>

      <div className="card">
        <RunPicker
          idPrefix="compare-run"
          runId={runId}
          onChange={(id) => setSearchParams(id ? { run: id } : {})}
          filter={(r) => r.models.length > 1}
        />
        {run ? (
          <p style={{ fontSize: 13 }}>
            Dataset: {run.dataset} · Status: <StatusPill status={run.status} /> · methodology v{run.eval_version}
          </p>
        ) : (
          <p style={{ fontSize: 13 }}>
            The list shows runs with 2+ models — trigger one from <Link to="/">Run configuration</Link> if it is empty.
          </p>
        )}
      </div>

      {run && <RunFailures run={run} />}
      {error && <div className="error-banner">{error}</div>}
      {loading && <p>Loading…</p>}

      {rows.length > 0 && (
        <>
          <div className="card">
            <h3>{isCode ? 'Unit-test pass rate' : 'ROUGE-L mean'}</h3>
            {rows.map((row, i) => (
              <div className="bar-row" key={row.model}>
                <span>{row.model}</span>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${(headline(row) / maxHeadline) * 100}%`,
                      background: SERIES_COLORS[i % SERIES_COLORS.length],
                    }}
                  />
                </div>
                <span className="bar-value">{isCode ? fmtPct(row.passRate) : fmtScore(row.rougeL)}</span>
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
                  {isCode ? (
                    <th>Pass rate</th>
                  ) : (
                    <>
                      <th>ROUGE-1</th>
                      <th>ROUGE-L</th>
                      <th>Supported sentences</th>
                      <th>Responses with a contradiction</th>
                    </>
                  )}
                  <th>Latency p50</th>
                  <th>Latency p95</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.model}>
                    <td>{row.model}</td>
                    <td>{row.count}</td>
                    {isCode ? (
                      <td>{fmtPct(row.passRate)}</td>
                    ) : (
                      <>
                        <td>{fmtScore(row.rouge1)}</td>
                        <td>{fmtScore(row.rougeL)}</td>
                        <td>{fmtPct(row.entailmentRate)}</td>
                        <td>{fmtPct(row.hallucRate)}</td>
                      </>
                    )}
                    <td>{fmtMs(row.latencyP50)}</td>
                    <td>{fmtMs(row.latencyP95)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!runId && <div className="empty-state">Pick a run above to compare its models.</div>}
      {run && !loading && rows.length === 0 && !['pending', 'running'].includes(run.status) && (
        <div className="empty-state">This run has no scored results.</div>
      )}
    </div>
  );
}
