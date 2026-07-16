import type {
  ModelInfo,
  RunCreateRequest,
  RunSummary,
  SampleResult,
  TimeseriesPoint,
} from './types';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // response had no JSON body
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export function listModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>('/models');
}

export function createRun(payload: RunCreateRequest): Promise<RunSummary> {
  return request<RunSummary>('/runs', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getRun(runId: string): Promise<RunSummary> {
  return request<RunSummary>(`/runs/${runId}`);
}

export function getRunResults(runId: string): Promise<SampleResult[]> {
  return request<SampleResult[]>(`/runs/${runId}/results`);
}

export function getTimeseries(params?: { model?: string; limit?: number }): Promise<TimeseriesPoint[]> {
  const qs = new URLSearchParams();
  if (params?.model) qs.set('model', params.model);
  if (params?.limit) qs.set('limit', String(params.limit));
  const suffix = qs.toString() ? `?${qs.toString()}` : '';
  return request<TimeseriesPoint[]>(`/metrics/timeseries${suffix}`);
}

export { API_BASE_URL };
