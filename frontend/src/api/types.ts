export type RunStatus = 'pending' | 'running' | 'completed' | 'partial' | 'failed';

export interface RunFailure {
  sample_id: string;
  model_name: string;
  error: string;
}

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
  eval_version: number;
  model_digests: Record<string, string | null>;
  failures: RunFailure[];
}

export interface SampleResult {
  sample_id: string;
  domain: string;
  prompt: string;
  expected: string;
  model_name: string;
  model_digest: string | null;
  response_text: string;
  latency_ms: number;
  load_ms: number | null;
  generation_ms: number | null;
  input_tokens: number;
  output_tokens: number;
  // Text samples (QA, summarization); null for code.
  rouge1: number | null;
  rouge2: number | null;
  rougeL: number | null;
  nli_label: string | null;
  nli_confidence: number | null;
  entailment_rate: number | null;
  contradiction_rate: number | null;
  hallucination_flag: boolean | null;
  // Code samples; null for text.
  tests_passed: boolean | null;
  exec_error: string | null;
}

export interface ModelInfo {
  name: string;
  available: boolean;
}

export interface DatasetInfo {
  name: string;
  path: string;
  sample_count: number;
  scoring: 'unit_tests' | 'rouge_nli';
}

export interface TimeseriesPoint {
  run_id: string;
  run_name: string | null;
  dataset: string | null;
  model_name: string;
  model_digest: string | null;
  eval_version: number;
  start_time: string | null;
  num_samples: number;
  rouge1_mean: number | null;
  rougeL_mean: number | null;
  halluc_rate: number | null;
  entailment_rate_mean: number | null;
  pass_rate: number | null;
  latency_p50: number | null;
  latency_p95: number | null;
  generation_p50: number | null;
}

export interface RunCreateRequest {
  dataset: string;
  models: string[];
  smoke_test?: boolean;
  use_mlflow?: boolean;
  parallel?: boolean;
}
