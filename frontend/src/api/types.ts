export type RunStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface RunSummary {
  id: string;
  dataset: string;
  models: string[];
  status: RunStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  sample_count: number;
  error: string | null;
}

export interface SampleResult {
  sample_id: string;
  domain: string;
  prompt: string;
  expected: string;
  model_name: string;
  response_text: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  rouge1: number;
  rouge2: number;
  rougeL: number;
  nli_label: string;
  nli_confidence: number;
  hallucination_flag: boolean;
}

export interface ModelInfo {
  name: string;
  available: boolean;
}

export interface TimeseriesPoint {
  run_id: string;
  run_name: string | null;
  dataset: string | null;
  model_name: string;
  start_time: string | null;
  rouge1_mean: number | null;
  rougeL_mean: number | null;
  halluc_rate: number | null;
  latency_p50: number | null;
  latency_p95: number | null;
}

export interface RunCreateRequest {
  dataset: string;
  models: string[];
  smoke_test?: boolean;
  use_mlflow?: boolean;
}
