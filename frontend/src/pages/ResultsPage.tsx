import { useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import type { SampleResult } from '../api/types';
import RunFailures from '../components/RunFailures';
import RunPicker from '../components/RunPicker';
import StatusPill from '../components/StatusPill';
import { useRunData } from '../hooks/useRunData';
import { fmtMs, fmtPct, fmtScore } from '../lib/stats';

type OutcomeFilter = 'all' | 'problems' | 'clean';

/** A sample "has a problem" if a sentence was contradicted or its code failed. */
function hasProblem(r: SampleResult) {
  return r.hallucination_flag === true || r.tests_passed === false;
}

export default function ResultsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const runId = searchParams.get('run') ?? '';
  const { run, results, loading, error } = useRunData(runId);

  const [modelFilter, setModelFilter] = useState<string>('all');
  const [outcomeFilter, setOutcomeFilter] = useState<OutcomeFilter>('all');

  const modelOptions = useMemo(() => Array.from(new Set(results.map((r) => r.model_name))), [results]);
  const isCode = results.length > 0 && results.every((r) => r.tests_passed !== null || r.exec_error);

  const filtered = results.filter((r) => {
    if (modelFilter !== 'all' && r.model_name !== modelFilter) return false;
    if (outcomeFilter === 'problems' && !hasProblem(r)) return false;
    if (outcomeFilter === 'clean' && hasProblem(r)) return false;
    return true;
  });

  return (
    <div>
      <h1>Results</h1>
      <p>Per-sample scores for an evaluation run.</p>

      <div className="card">
        <RunPicker
          idPrefix="results-run"
          runId={runId}
          onChange={(id) => setSearchParams(id ? { run: id } : {})}
        />
        {run && (
          <p>
            Status: <StatusPill status={run.status} /> · {run.sample_count} result(s) · {run.models.join(', ')} ·
            methodology v{run.eval_version}
          </p>
        )}
        {run?.eval_version === 1 && (
          <p style={{ fontSize: 13 }}>
            This run used the original scoring (free-length answers, whole-response NLI). Its ROUGE and
            hallucination numbers are not comparable with newer runs.
          </p>
        )}
      </div>

      {run && <RunFailures run={run} />}
      {error && <div className="error-banner">{error}</div>}
      {loading && <p>Loading…</p>}
      {run && ['pending', 'running'].includes(run.status) && <p>This run is still {run.status}. Results appear when it finishes.</p>}

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
              <label htmlFor="outcome-filter">Outcome</label>
              <select
                id="outcome-filter"
                value={outcomeFilter}
                onChange={(e) => setOutcomeFilter(e.target.value as OutcomeFilter)}
              >
                <option value="all">All results</option>
                <option value="problems">{isCode ? 'Failed tests only' : 'Contradictions only'}</option>
                <option value="clean">{isCode ? 'Passed tests only' : 'No contradiction'}</option>
              </select>
            </div>
          </div>

          <table className="data-table">
            <thead>
              <tr>
                <th>Sample</th>
                <th>Model</th>
                {isCode ? (
                  <th>Unit tests</th>
                ) : (
                  <>
                    <th>ROUGE-L</th>
                    <th title="Share of response sentences the NLI model judged supported by the premise">Supported</th>
                    <th title="Whether any response sentence was contradicted by the premise">Contradiction</th>
                  </>
                )}
                <th>Latency</th>
                <th>Response</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r, i) => (
                <tr key={`${r.sample_id}-${r.model_name}-${i}`}>
                  <td>{r.sample_id}</td>
                  <td>{r.model_name}</td>
                  {isCode ? (
                    <td>
                      {r.tests_passed === null ? (
                        <span className="pill pill-neutral">not run</span>
                      ) : (
                        <span className={`pill ${r.tests_passed ? 'pill-good' : 'pill-critical'}`} title={r.exec_error ?? ''}>
                          {r.tests_passed ? 'passed' : 'failed'}
                        </span>
                      )}
                    </td>
                  ) : (
                    <>
                      <td>{fmtScore(r.rougeL)}</td>
                      <td>{fmtPct(r.entailment_rate)}</td>
                      <td>
                        {r.hallucination_flag === null ? (
                          <span className="pill pill-neutral">n/a</span>
                        ) : (
                          <span className={`pill ${r.hallucination_flag ? 'pill-critical' : 'pill-good'}`}>
                            {r.hallucination_flag ? 'flagged' : 'none'}
                          </span>
                        )}
                      </td>
                    </>
                  )}
                  <td>{fmtMs(r.latency_ms)}</td>
                  <td style={{ maxWidth: 420, whiteSpace: 'pre-wrap' }}>
                    {r.response_text}
                    {r.exec_error && <div style={{ fontSize: 12, marginTop: 4 }}>⚠ {r.exec_error}</div>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {filtered.length === 0 && <p className="empty-state">No results match this filter.</p>}
        </div>
      )}

      {!runId && <div className="empty-state">Pick a run above to see its results.</div>}
    </div>
  );
}
