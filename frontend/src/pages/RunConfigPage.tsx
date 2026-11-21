import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { createRun, getRun, listDatasets, listModels } from '../api/client';
import type { DatasetInfo, ModelInfo, RunSummary } from '../api/types';
import RunFailures from '../components/RunFailures';
import StatusPill from '../components/StatusPill';

const POLL_INTERVAL_MS = 2000;
const MAX_CONSECUTIVE_POLL_ERRORS = 5;

export default function RunConfigPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [dataset, setDataset] = useState('');
  const [smokeTest, setSmokeTest] = useState(false);
  const [useMlflow, setUseMlflow] = useState(true);
  const [parallel, setParallel] = useState(false);

  const [run, setRun] = useState<RunSummary | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [pollWarning, setPollWarning] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    Promise.all([listModels(), listDatasets()])
      .then(([modelData, datasetData]) => {
        setModels(modelData);
        setDatasets(datasetData);
        const firstAvailable = modelData.find((m) => m.available);
        if (firstAvailable) setSelectedModels([firstAvailable.name]);
        if (datasetData.length > 0) setDataset(datasetData[0].name);
      })
      .catch((err: Error) => setLoadError(err.message));
  }, []);

  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  function toggleModel(name: string) {
    setSelectedModels((prev) => (prev.includes(name) ? prev.filter((m) => m !== name) : [...prev, name]));
  }

  function stopPolling() {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = null;
  }

  async function handleSubmit() {
    setSubmitError(null);
    setPollWarning(null);
    setSubmitting(true);
    try {
      const created = await createRun({
        dataset,
        models: selectedModels,
        smoke_test: smokeTest,
        use_mlflow: useMlflow,
        parallel,
      });
      setRun(created);

      stopPolling();
      let consecutiveErrors = 0;
      pollRef.current = setInterval(async () => {
        try {
          const updated = await getRun(created.id);
          consecutiveErrors = 0;
          setPollWarning(null);
          setRun(updated);
          if (!['pending', 'running'].includes(updated.status)) stopPolling();
        } catch (err) {
          // A single dropped request shouldn't end polling; give up only if
          // the API stays unreachable.
          consecutiveErrors += 1;
          setPollWarning(`Lost contact with the API (${(err as Error).message}); retrying…`);
          if (consecutiveErrors >= MAX_CONSECUTIVE_POLL_ERRORS) {
            stopPolling();
            setPollWarning('Stopped polling: the API has not answered for 10 seconds. Reload to check the run.');
          }
        }
      }, POLL_INTERVAL_MS);
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  const canSubmit = selectedModels.length > 0 && !!dataset && !submitting;
  const selectedDataset = datasets.find((d) => d.name === dataset);
  const finished = run && !['pending', 'running'].includes(run.status);

  return (
    <div>
      <h1>Run configuration</h1>
      <p>Select a dataset and one or more models, then trigger an evaluation run against the API.</p>

      <div className="card">
        {loadError && <div className="error-banner">Could not reach the API: {loadError}</div>}

        <div className="field-row">
          <div className="field">
            <label htmlFor="dataset-select">Dataset</label>
            <select id="dataset-select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {datasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({d.sample_count} samples)
                </option>
              ))}
            </select>
            {selectedDataset && (
              <p style={{ fontSize: 13, marginTop: 6 }}>
                Scored by{' '}
                {selectedDataset.scoring === 'unit_tests'
                  ? 'running each answer against unit tests (pass rate).'
                  : 'ROUGE against a reference answer and sentence-level NLI grounding.'}
              </p>
            )}
          </div>
        </div>

        <div className="field">
          <label>Models</label>
          <div>
            {models.map((m) => (
              <label key={m.name} className="checkbox-row" style={{ marginBottom: 6 }}>
                <input type="checkbox" checked={selectedModels.includes(m.name)} onChange={() => toggleModel(m.name)} />
                {m.name} {!m.available && <span className="pill pill-neutral">not pulled</span>}
              </label>
            ))}
            {models.length === 0 && !loadError && <p>Loading models…</p>}
          </div>
        </div>

        <label className="checkbox-row" style={{ marginBottom: 8 }}>
          <input type="checkbox" checked={smokeTest} onChange={(e) => setSmokeTest(e.target.checked)} />
          Smoke test (first 2 samples only)
        </label>

        <label className="checkbox-row" style={{ marginBottom: 8 }}>
          <input type="checkbox" checked={parallel} onChange={(e) => setParallel(e.target.checked)} />
          Query models in parallel (faster; latency then includes time spent queued on the Ollama server)
        </label>

        <label className="checkbox-row" style={{ marginBottom: 16 }}>
          <input type="checkbox" checked={useMlflow} onChange={(e) => setUseMlflow(e.target.checked)} />
          Log to MLflow
        </label>

        <button className="btn-primary" onClick={handleSubmit} disabled={!canSubmit}>
          {submitting ? 'Starting…' : 'Run evaluation'}
        </button>

        {submitError && <div className="error-banner" style={{ marginTop: 16 }}>{submitError}</div>}
      </div>

      {run && (
        <div className="card">
          <h3>Run {run.id}</h3>
          <p>
            Status: <StatusPill status={run.status} /> · Dataset: {run.dataset} · Models: {run.models.join(', ')}
          </p>
          {pollWarning && <p style={{ fontSize: 13 }}>{pollWarning}</p>}
          {finished && run.sample_count > 0 && (
            <p>
              {run.sample_count} result(s) ready. <Link to={`/results?run=${run.id}`}>View results →</Link>
              {run.models.length > 1 && (
                <>
                  {' '}
                  · <Link to={`/compare?run=${run.id}`}>Compare models →</Link>
                </>
              )}
            </p>
          )}
          <RunFailures run={run} />
        </div>
      )}
    </div>
  );
}
