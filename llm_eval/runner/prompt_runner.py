import asyncio
import sys
import threading
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

from ..adapters.base import BaseAdapter, ModelResponse
from ..evaluators.code_execution_evaluator import CodeExecutionEvaluator
from ..evaluators.rouge_evaluator import RougeEvaluator
from .dataset_loader import EvalSample
from .summary import summarize_by_model

# Bumped whenever scoring changes in a way that makes new scores incomparable
# with old ones. Stored with every persisted run so trend charts and drift
# checks never mix methodologies.
#   1 — original: free-length answers, whole-response NLI vs reference
#       (first 500 chars), ROUGE on code.
#   2 — concise-answer prompts, greedy decoding (temperature 0, seed 42),
#       sentence-level NLI vs source/reference, unit-test execution for code.
EVAL_VERSION = 2

# rich's spinner animation writes Unicode Braille frames (e.g. U+280B); on
# Windows, a background process's stdout defaults to the system codepage
# (cp1252) rather than UTF-8, so the very first frame raises
# UnicodeEncodeError and aborts the run before any sample executes. This
# never changes what's printed on a real terminal (already UTF-8-capable
# there) — it only stops a cosmetic progress spinner from crashing the run
# when stdout can't represent it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass


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
    estimated_cost_usd: float = 0.0
    model_digest: Optional[str] = None
    load_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    # Text samples (QA, summarization): ROUGE + NLI grounding. None for code.
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    nli_label: Optional[str] = None
    nli_confidence: Optional[float] = None
    entailment_rate: Optional[float] = None
    contradiction_rate: Optional[float] = None
    hallucination_flag: Optional[bool] = None
    # Code samples: unit-test execution. None for text.
    tests_passed: Optional[bool] = None
    exec_error: Optional[str] = None


@dataclass
class EvalFailure:
    sample_id: str
    model_name: str
    error: str


@dataclass
class RunOutcome:
    results: List[EvalResult] = field(default_factory=list)
    failures: List[EvalFailure] = field(default_factory=list)

    @property
    def status(self) -> str:
        """completed: every call scored; partial: some calls failed;
        failed: calls were attempted and none succeeded."""
        if self.failures and not self.results:
            return "failed"
        if self.failures:
            return "partial"
        return "completed"

    def failure_summary(self, limit: int = 5) -> Optional[str]:
        if not self.failures:
            return None
        total = len(self.failures) + len(self.results)
        head = "; ".join(f"{f.model_name}/{f.sample_id}: {f.error}" for f in self.failures[:limit])
        more = f" (+{len(self.failures) - limit} more)" if len(self.failures) > limit else ""
        return f"{len(self.failures)} of {total} model calls failed — {head}{more}"


# Evaluators are expensive to build (the NLI model is ~0.1B parameters), so
# they are created once per process and shared by every run.
_evaluator_lock = threading.Lock()
_evaluators: dict = {}


def get_evaluator(name: str):
    with _evaluator_lock:
        if name not in _evaluators:
            if name == "rouge":
                _evaluators[name] = RougeEvaluator()
            elif name == "nli":
                from ..evaluators.hallucination_evaluator import HallucinationEvaluator

                _evaluators[name] = HallucinationEvaluator()
            elif name == "code":
                _evaluators[name] = CodeExecutionEvaluator()
            else:
                raise KeyError(name)
        return _evaluators[name]


def _error_message(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class PromptRunner:
    def __init__(
        self,
        adapters: List[BaseAdapter],
        use_mlflow: bool = True,
        parallel: bool = False,
        evaluators: Optional[dict] = None,
    ):
        """
        parallel=False (default) sends one request at a time. A single Ollama
        server queues concurrent requests and may swap models in and out of
        memory, so wall-clock latency measured under parallel load includes
        waiting time and is not a fair per-model comparison. parallel=True
        sends every model's request for a sample at once — faster overall when
        the server has room to keep all models loaded, but treat latency_ms as
        indicative only (generation_ms, reported by the server, still excludes
        queueing and loading).
        """
        self.adapters = adapters
        self.use_mlflow = use_mlflow
        self.parallel = parallel
        self._injected = evaluators or {}

    def _evaluator(self, name: str):
        return self._injected.get(name) or get_evaluator(name)

    def _score(self, sample: EvalSample, response: ModelResponse, digest: Optional[str]) -> EvalResult:
        result = EvalResult(
            sample_id=sample.id,
            domain=sample.domain,
            prompt=sample.prompt,
            expected=sample.expected,
            model_name=response.model_name,
            model_digest=digest,
            response_text=response.response_text,
            latency_ms=response.latency_ms,
            load_ms=response.load_ms,
            generation_ms=response.generation_ms,
            input_tokens=response.input_tokens or 0,
            output_tokens=response.output_tokens or 0,
        )
        if sample.tests:
            code = self._evaluator("code").score(response.response_text, sample.tests)
            result.tests_passed = code["tests_passed"]
            result.exec_error = code["exec_error"]
            return result

        rouge = self._evaluator("rouge").score(response.response_text, sample.expected)
        result.rouge1, result.rouge2, result.rougeL = rouge["rouge1"], rouge["rouge2"], rouge["rougeL"]
        premise = sample.context or sample.expected
        nli = self._evaluator("nli").score(response.response_text, premise)
        result.nli_label = nli["nli_label"]
        result.nli_confidence = nli["nli_confidence"]
        result.entailment_rate = nli["entailment_rate"]
        result.contradiction_rate = nli["contradiction_rate"]
        result.hallucination_flag = nli["hallucination_flag"]
        return result

    async def _run_single(
        self, adapter: BaseAdapter, sample: EvalSample, digest: Optional[str]
    ) -> EvalResult:
        response = await adapter.ask(sample.prompt, system_prompt=sample.system_prompt)
        # Scoring is CPU-bound (NLI model, subprocesses); run it off the event
        # loop so an API server stays responsive while a run is in progress.
        return await asyncio.to_thread(self._score, sample, response, digest)

    async def _attempt(self, adapter, sample, digest, outcome: RunOutcome) -> None:
        try:
            outcome.results.append(await self._run_single(adapter, sample, digest))
        except Exception as exc:  # recorded, not swallowed
            outcome.failures.append(EvalFailure(sample.id, adapter.model_name, _error_message(exc)))
            print(f"[error] {adapter.model_name}/{sample.id}: {_error_message(exc)}", file=sys.stderr)

    async def run(self, samples: List[EvalSample], run_name: str = "eval_run") -> RunOutcome:
        outcome = RunOutcome()
        digests = await asyncio.gather(*[a.model_digest() for a in self.adapters])
        digest_by_model = {a.model_name: d for a, d in zip(self.adapters, digests)}

        # Uses MlflowClient with an explicit run_id rather than the fluent
        # mlflow.start_run() API: the fluent API tracks a single global
        # "active run" per process, so two runs triggered close together
        # (e.g. two API-triggered evals) crash with "Run ... is already
        # active" instead of both logging independently.
        client = None
        run_id: Optional[str] = None
        if self.use_mlflow:
            import mlflow

            client = mlflow.tracking.MlflowClient()
            run_id = client.create_run(experiment_id="0", run_name=run_name).info.run_id
            client.log_param(run_id, "num_samples", len(samples))
            client.log_param(run_id, "models", ",".join(a.model_name for a in self.adapters))
            client.log_param(run_id, "eval_version", EVAL_VERSION)
            client.log_param(run_id, "parallel", self.parallel)

        try:
            with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
                task = progress.add_task(
                    f"Evaluating {len(samples)} samples × {len(self.adapters)} models...",
                    total=len(samples),
                )
                for sample in samples:
                    if self.parallel:
                        await asyncio.gather(
                            *[self._attempt(a, sample, digest_by_model[a.model_name], outcome) for a in self.adapters]
                        )
                    else:
                        for a in self.adapters:
                            await self._attempt(a, sample, digest_by_model[a.model_name], outcome)
                    progress.advance(task)

            if client and outcome.results:
                summary = summarize_by_model([asdict(r) for r in outcome.results])
                for model, stats in summary.items():
                    prefix = model.replace(":", "_").replace("/", "_")
                    for key, value in stats.items():
                        if value is not None and key != "num_samples":
                            client.log_metric(run_id, f"{prefix}_{key}", value)
                client.log_metric(run_id, "failed_calls", len(outcome.failures))
        except Exception:
            if client and run_id:
                client.set_terminated(run_id, status="FAILED")
            raise
        else:
            if client and run_id:
                client.set_terminated(run_id, status="FINISHED" if outcome.status != "failed" else "FAILED")

        return outcome
