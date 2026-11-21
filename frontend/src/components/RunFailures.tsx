import type { RunSummary } from '../api/types';

/** Lists model calls that failed in a run (timeouts, missing models, ...). */
export default function RunFailures({ run }: { run: RunSummary }) {
  if (run.failures.length === 0) {
    return run.status === 'failed' && run.error ? <div className="error-banner">{run.error}</div> : null;
  }
  return (
    <div className="error-banner">
      <strong>
        {run.failures.length} model call{run.failures.length === 1 ? '' : 's'} failed
      </strong>
      <ul style={{ margin: '6px 0 0', paddingLeft: 18 }}>
        {run.failures.slice(0, 10).map((f, i) => (
          <li key={i}>
            {f.model_name} / {f.sample_id}: {f.error}
          </li>
        ))}
      </ul>
    </div>
  );
}
