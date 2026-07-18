import asyncio
import mlflow
from typing import List, Optional
from dataclasses import dataclass, asdict

from rich.progress import Progress, SpinnerColumn, TextColumn
import pandas as pd

from ..adapters.base import BaseAdapter, ModelResponse
from ..evaluators.rouge_evaluator import RougeEvaluator
from ..evaluators.hallucination_evaluator import HallucinationEvaluator
from .dataset_loader import EvalSample


@dataclass
class EvalResult:
    sample_id: str
    domain: str
    prompt: str
    expected: str
    model_name: str
    response_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    rouge1: float
    rouge2: float
    rougeL: float
    nli_label: str
    nli_confidence: float
    hallucination_flag: bool


class PromptRunner:
    def __init__(self, adapters: List[BaseAdapter], use_mlflow: bool = True):
        self.adapters = adapters
        self.rouge = RougeEvaluator()
        self.hallucination = HallucinationEvaluator()
        self.use_mlflow = use_mlflow

    async def _run_single(
        self, adapter: BaseAdapter, sample: EvalSample
    ) -> EvalResult:
        response: ModelResponse = await adapter.ask(sample.prompt)
        rouge_scores = self.rouge.score(response.response_text, sample.expected)
        halluc_scores = self.hallucination.score(
            response.response_text, sample.expected
        )
        return EvalResult(
            sample_id=sample.id,
            domain=sample.domain,
            prompt=sample.prompt,
            expected=sample.expected,
            model_name=response.model_name,
            response_text=response.response_text,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens or 0,
            output_tokens=response.output_tokens or 0,
            estimated_cost_usd=0.0,
            rouge1=rouge_scores["rouge1"],
            rouge2=rouge_scores["rouge2"],
            rougeL=rouge_scores["rougeL"],
            nli_label=halluc_scores["nli_label"],
            nli_confidence=halluc_scores["nli_confidence"],
            hallucination_flag=halluc_scores["hallucination_flag"],
        )

    async def run(
        self, samples: List[EvalSample], run_name: str = "eval_run"
    ) -> List[EvalResult]:
        all_results = []

        # Uses MlflowClient with an explicit run_id rather than the fluent
        # mlflow.start_run() API: the fluent API tracks a single global
        # "active run" per process, so two runs triggered close together
        # (e.g. two API-triggered evals) crash with "Run ... is already
        # active" instead of both logging independently.
        client: Optional[mlflow.tracking.MlflowClient] = None
        run_id: Optional[str] = None
        if self.use_mlflow:
            client = mlflow.tracking.MlflowClient()
            run_id = client.create_run(experiment_id="0", run_name=run_name).info.run_id
            client.log_param(run_id, "num_samples", len(samples))
            client.log_param(run_id, "models", [a.model_name for a in self.adapters])

        try:
            with Progress(
                SpinnerColumn(), TextColumn("{task.description}")
            ) as progress:
                task = progress.add_task(
                    f"Evaluating {len(samples)} samples × {len(self.adapters)} models...",
                    total=len(samples),
                )

                for sample in samples:
                    responses = await asyncio.gather(
                        *[self._run_single(a, sample) for a in self.adapters],
                        return_exceptions=True,
                    )
                    for result in responses:
                        if isinstance(result, Exception):
                            print(f"[error] {result}")
                            continue
                        all_results.append(result)
                    progress.advance(task)

            if client and all_results:
                df = pd.DataFrame([asdict(r) for r in all_results])
                for model in df["model_name"].unique():
                    mdf = df[df["model_name"] == model]
                    prefix = model.replace(":", "_").replace("/", "_")
                    client.log_metric(run_id, f"{prefix}_rouge1_mean",
                                      round(float(mdf["rouge1"].mean()), 4))
                    client.log_metric(run_id, f"{prefix}_rougeL_mean",
                                      round(float(mdf["rougeL"].mean()), 4))
                    client.log_metric(run_id, f"{prefix}_halluc_rate",
                                      round(float(mdf["hallucination_flag"].mean()), 4))
                    client.log_metric(run_id, f"{prefix}_latency_p50",
                                      round(float(mdf["latency_ms"].median()), 2))
                    client.log_metric(run_id, f"{prefix}_latency_p95",
                                      round(float(mdf["latency_ms"].quantile(0.95)), 2))
        except Exception:
            if client and run_id:
                client.set_terminated(run_id, status="FAILED")
            raise
        else:
            if client and run_id:
                client.set_terminated(run_id, status="FINISHED")

        return all_results
