from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    dataset: str = Field(
        ..., description="Dataset name or path, e.g. 'legal_qa' or 'datasets/legal_qa'"
    )
    models: List[str] = Field(
        ..., min_length=1, description="Ollama model names to evaluate, e.g. ['gemma3:4b']"
    )
    smoke_test: bool = Field(False, description="Only run the first 2 samples")
    use_mlflow: bool = Field(True, description="Log this run to MLflow")


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


class SampleResult(BaseModel):
    sample_id: str
    domain: str
    prompt: str
    expected: str
    model_name: str
    response_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    rouge1: float
    rouge2: float
    rougeL: float
    nli_label: str
    nli_confidence: float
    hallucination_flag: bool


class ModelInfo(BaseModel):
    name: str
    available: bool


class TimeseriesPoint(BaseModel):
    run_id: str
    run_name: Optional[str] = None
    dataset: Optional[str] = None
    model_name: str
    start_time: Optional[datetime] = None
    rouge1_mean: Optional[float] = None
    rougeL_mean: Optional[float] = None
    halluc_rate: Optional[float] = None
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
