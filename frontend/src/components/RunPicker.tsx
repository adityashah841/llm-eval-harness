import { useEffect, useState } from 'react';
import { listRuns } from '../api/client';
import type { RunSummary } from '../api/types';

interface RunPickerProps {
  runId: string;
  onChange: (runId: string) => void;
  idPrefix: string;
  filter?: (run: RunSummary) => boolean;
}

/** Choose a run from the API's run history, or paste an id. */
export default function RunPicker({ runId, onChange, idPrefix, filter }: RunPickerProps) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRuns(100)
      .then(setRuns)
      .catch((err: Error) => setError(err.message));
  }, []);

  const options = runs.filter((r) => r.status !== 'pending' && r.status !== 'running').filter(filter ?? (() => true));

  return (
    <>
      <div className="field-row">
        <div className="field">
          <label htmlFor={`${idPrefix}-select`}>Run</label>
          <select id={`${idPrefix}-select`} value={options.some((r) => r.id === runId) ? runId : ''} onChange={(e) => onChange(e.target.value)}>
            <option value="">Select a run…</option>
            {options.map((r) => (
              <option key={r.id} value={r.id}>
                {r.dataset.replace(/^datasets\//, '')} · {r.models.join('+')} · {r.status} ·{' '}
                {new Date(r.created_at).toLocaleString()}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor={`${idPrefix}-input`}>or paste a run id</label>
          <input
            id={`${idPrefix}-input`}
            value={runId}
            placeholder="run id"
            onChange={(e) => onChange(e.target.value.trim())}
          />
        </div>
      </div>
      {error && <p style={{ fontSize: 13 }}>Could not load run history: {error}</p>}
    </>
  );
}
