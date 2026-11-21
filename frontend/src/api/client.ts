import type {
  DatasetInfo,
  ModelInfo,
  RunCreateRequest,
  RunSummary,
  SampleResult,
  TimeseriesPoint,
} from './types';

const API_BASE_URL = (
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'
).replace(/\/$/, '');

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === 'string') detail = body.detail;
    } catch {
      // response had no JSON body
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

export function listModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>('/models');
}

export function listDatasets(): Promise<DatasetInfo[]> {
  return request<DatasetInfo[]>('/datasets');
}

export function listRuns(limit = 50): Promise<RunSummary[]> {
  return request<RunSummary[]>(`/runs?limit=${limit}`);
}

export function createRun(payload: RunCreateRequest): Promise<RunSummary> {
  return request<RunSummary>('/runs', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getRun(runId: string, signal?: AbortSignal): Promise<RunSummary> {
  return request<RunSummary>(`/runs/${encodeURIComponent(runId)}`, { signal });
}

export function getRunResults(runId: string, signal?: AbortSignal): Promise<SampleResult[]> {
  return request<SampleResult[]>(`/runs/${encodeURIComponent(runId)}/results`, { signal });
}

export function getTimeseries(params?: {
  model?: string;
  dataset?: string;
  eval_version?: number;
  limit?: number;
}): Promise<TimeseriesPoint[]> {
  const qs = new URLSearchParams();
  if (params?.model) qs.set('model', params.model);
  if (params?.dataset) qs.set('dataset', params.dataset);
  if (params?.eval_version != null) qs.set('eval_version', String(params.eval_version));
  if (params?.limit) qs.set('limit', String(params.limit));
  const suffix = qs.toString() ? `?${qs.toString()}` : '';
  return request<TimeseriesPoint[]>(`/metrics/timeseries${suffix}`);
}

export { API_BASE_URL };
