from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    dataset: str = Field(
        ..., description="Dataset name, e.g. 'legal_qa' (or 'datasets/legal_qa'); see GET /datasets"
    )
    models: List[str] = Field(
        ..., min_length=1, max_length=8, description="Ollama model names to evaluate, e.g. ['gemma3:4b']"
    )
    smoke_test: bool = Field(False, description="Only run the first 2 samples")
    use_mlflow: bool = Field(True, description="Log this run to MLflow")
    parallel: bool = Field(
        False,
        description="Send every model's request for a sample at once. Faster, but wall-clock "
        "latency then includes server-side queueing.",
    )


class RunFailure(BaseModel):
    sample_id: str
    model_name: str
    error: str


class RunSummary(BaseModel):
    id: str
    dataset: str
    models: List[str]
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    sample_count: int = 0
    error: Optional[str] = None
    eval_version: int = 1
    model_digests: Dict[str, Optional[str]] = Field(default_factory=dict)
    failures: List[RunFailure] = Field(default_factory=list)


class SampleResult(BaseModel):
    sample_id: str
    domain: str
    prompt: str
    expected: str
    model_name: str
    model_digest: Optional[str] = None
    response_text: str
    latency_ms: float
    load_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    input_tokens: int
    output_tokens: int
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    nli_label: Optional[str] = None
    nli_confidence: Optional[float] = None
    entailment_rate: Optional[float] = None
    contradiction_rate: Optional[float] = None
    hallucination_flag: Optional[bool] = None
    tests_passed: Optional[bool] = None
    exec_error: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    available: bool


class DatasetInfo(BaseModel):
    name: str
    path: str
    sample_count: int
    scoring: str = Field(description="'unit_tests' for code datasets, otherwise 'rouge_nli'")


class TimeseriesPoint(BaseModel):
    run_id: str
    run_name: Optional[str] = None
    dataset: Optional[str] = None
    model_name: str
    model_digest: Optional[str] = None
    eval_version: int = 1
    start_time: Optional[datetime] = None
    num_samples: int = 0
    rouge1_mean: Optional[float] = None
    rougeL_mean: Optional[float] = None
    halluc_rate: Optional[float] = None
    entailment_rate_mean: Optional[float] = None
    pass_rate: Optional[float] = None
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    generation_p50: Optional[float] = None
