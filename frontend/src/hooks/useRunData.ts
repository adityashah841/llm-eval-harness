import { useEffect, useState } from 'react';
import { ApiError, getRun, getRunResults } from '../api/client';
import type { RunSummary, SampleResult } from '../api/types';

const DEBOUNCE_MS = 300;

interface RunData {
  run: RunSummary | null;
  results: SampleResult[];
  loading: boolean;
  error: string | null;
}

/** Loads a run and its results. Debounced (the id may be typed by hand) and
 * abortable, so a stale response can never overwrite a newer one. A run that
 * is still in progress returns its status with no results rather than an error. */
export function useRunData(runId: string): RunData {
  const [state, setState] = useState<RunData>({ run: null, results: [], loading: false, error: null });

  useEffect(() => {
    if (!runId) {
      setState({ run: null, results: [], loading: false, error: null });
      return;
    }
    const controller = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));

    const timer = setTimeout(async () => {
      try {
        const run = await getRun(runId, controller.signal);
        let results: SampleResult[] = [];
        if (run.status !== 'pending' && run.status !== 'running') {
          results = await getRunResults(runId, controller.signal);
        }
        setState({ run, results, loading: false, error: null });
      } catch (err) {
        if (controller.signal.aborted) return;
        const message = err instanceof ApiError && err.status === 404 ? `No run with id "${runId}"` : (err as Error).message;
        setState({ run: null, results: [], loading: false, error: message });
      }
    }, DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [runId]);

  return state;
}
