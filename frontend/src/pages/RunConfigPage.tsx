import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { createRun, getRun, listModels } from '../api/client';
import { DATASETS } from '../api/constants';
import type { ModelInfo, RunSummary } from '../api/types';
import { useRuns } from '../context/RunsContext';

const POLL_INTERVAL_MS = 2000;

export default function RunConfigPage() {
  const { addRun } = useRuns();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [dataset, setDataset] = useState(DATASETS[0].path);
  const [smokeTest, setSmokeTest] = useState(false);
  const [useMlflow, setUseMlflow] = useState(true);

  const [run, setRun] = useState<RunSummary | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    listModels()
      .then((data) => {
        setModels(data);
        const firstAvailable = data.find((m) => m.available);
        if (firstAvailable) setSelectedModels([firstAvailable.name]);
      })
      .catch((err: Error) => setModelsError(err.message));
  }, []);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  function toggleModel(name: string) {
    setSelectedModels((prev) =>
      prev.includes(name) ? prev.filter((m) => m !== name) : [...prev, name]
    );
  }

  async function handleSubmit() {
    setSubmitError(null);
    setSubmitting(true);
    try {
      const created = await createRun({
        dataset,
        models: selectedModels,
        smoke_test: smokeTest,
        use_mlflow: useMlflow,
      });
      setRun(created);
      addRun({
        id: created.id,
        dataset: created.dataset,
        models: created.models,
        created_at: created.created_at,
      });

      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const updated = await getRun(created.id);
          setRun(updated);
          if (updated.status === 'completed' || updated.status === 'failed') {
            if (pollRef.current) clearInterval(pollRef.current);
          }
        } catch {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      }, POLL_INTERVAL_MS);
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  const canSubmit = selectedModels.length > 0 && dataset && !submitting;

  return (
    <div>
      <h1>Run configuration</h1>
      <p>Select a dataset and one or more models, then trigger an evaluation run against the API.</p>

      <div className="card">
        {modelsError && (
          <div className="error-banner">Could not load models: {modelsError}</div>
        )}

        <div className="field-row">
          <div className="field">
            <label htmlFor="dataset-select">Dataset</label>
            <select id="dataset-select" value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {DATASETS.map((d) => (
                <option key={d.path} value={d.path}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="field">
          <label>Models</label>
          <div>
            {models.map((m) => (
              <label key={m.name} className="checkbox-row" style={{ marginBottom: 6 }}>
                <input
                  type="checkbox"
                  checked={selectedModels.includes(m.name)}
                  onChange={() => toggleModel(m.name)}
                />
                {m.name} {!m.available && <span className="pill pill-neutral">not pulled</span>}
              </label>
            ))}
            {models.length === 0 && !modelsError && <p>Loading models…</p>}
          </div>
        </div>

        <label className="checkbox-row" style={{ marginBottom: 8 }}>
          <input type="checkbox" checked={smokeTest} onChange={(e) => setSmokeTest(e.target.checked)} />
          Smoke test (first 2 samples only)
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
          {run.status === 'completed' && (
            <p>
              {run.sample_count} result(s) ready. <Link to={`/results?run=${run.id}`}>View results →</Link>
            </p>
          )}
          {run.status === 'failed' && <div className="error-banner">{run.error}</div>}
        </div>
      )}
    </div>
  );
}

function StatusPill({ status }: { status: RunSummary['status'] }) {
  const cls = status === 'completed' ? 'pill-good' : status === 'failed' ? 'pill-critical' : 'pill-neutral';
  return <span className={`pill ${cls}`}>{status}</span>;
}
