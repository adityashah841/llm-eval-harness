import asyncio
import json
import textwrap

import httpx
import pytest

from llm_eval.adapters import ollama_adapter
from llm_eval.adapters.base import BaseAdapter, ModelResponse
from llm_eval.adapters.ollama_adapter import OllamaAdapter, strip_thinking
from llm_eval.evaluators import CodeExecutionEvaluator, HallucinationEvaluator, RougeEvaluator, extract_code, split_claims
from llm_eval.runner.dataset_loader import DatasetError, EvalSample, load_dataset
from llm_eval.runner.prompt_runner import PromptRunner
from llm_eval.runner.summary import aggregate_rows, percentile

from .conftest import REAL_ASK, REAL_MODEL_DIGEST, fake_nli


# ---------------------------------------------------------------- adapter


def test_strip_thinking():
    assert strip_thinking("<think>\nlet me reason\n</think>\n\nParis.") == "Paris."
    assert strip_thinking("<think>never closed") == ""
    assert strip_thinking("No thinking here.") == "No thinking here."


def _patch_transport(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(ollama_adapter.httpx, "AsyncClient", factory)


def test_ollama_adapter_request_and_response_parsing(monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "<think>hmm</think>Answer."},
            "prompt_eval_count": 12,
            "eval_count": 3,
            "load_duration": 2_000_000_000,
            "prompt_eval_duration": 100_000_000,
            "eval_duration": 400_000_000,
        })

    monkeypatch.setattr(OllamaAdapter, "ask", REAL_ASK)  # undo the autouse stub
    _patch_transport(monkeypatch, handler)

    resp = asyncio.run(OllamaAdapter("gemma3:4b").ask("Q?", system_prompt="Be brief."))
    assert resp.response_text == "Answer."
    assert resp.load_ms == 2000.0
    assert resp.generation_ms == 500.0
    assert (resp.input_tokens, resp.output_tokens) == (12, 3)
    body = seen["body"]
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["seed"] == 42


def test_ollama_error_message_is_surfaced(monkeypatch):
    monkeypatch.setattr(OllamaAdapter, "ask", REAL_ASK)
    _patch_transport(monkeypatch, lambda r: httpx.Response(404, json={"error": "model 'x' not found"}))
    with pytest.raises(ollama_adapter.OllamaError, match="HTTP 404: model 'x' not found"):
        asyncio.run(OllamaAdapter("x").ask("hi"))


def test_ollama_model_digest_lookup(monkeypatch):
    monkeypatch.setattr(OllamaAdapter, "model_digest", REAL_MODEL_DIGEST)
    _patch_transport(monkeypatch, lambda r: httpx.Response(200, json={"models": [
        {"name": "phi4:latest", "model": "phi4:latest", "digest": "abc"},
        {"name": "gemma3:4b", "model": "gemma3:4b", "digest": "def"},
    ]}))
    assert asyncio.run(OllamaAdapter("phi4").model_digest()) == "abc"
    assert asyncio.run(OllamaAdapter("gemma3:4b").model_digest()) == "def"
    assert asyncio.run(OllamaAdapter("missing").model_digest()) is None


# ------------------------------------------------------------- evaluators


def test_rouge_identical_and_disjoint():
    r = RougeEvaluator()
    assert r.score("the cat sat", "the cat sat")["rougeL"] == 1.0
    assert r.score("alpha beta", "gamma delta")["rouge1"] == 0.0


def test_rouge_penalizes_verbosity():
    """Why datasets ask for short answers: the same correct sentence buried
    in filler scores far lower."""
    r = RougeEvaluator()
    ref = "Consideration is something of value exchanged between parties to a contract."
    concise = r.score(ref, ref)["rougeL"]
    padded = r.score(ref + " " + "Here is some additional background detail. " * 20, ref)["rougeL"]
    assert concise == 1.0 and padded < 0.3


def test_split_claims_drops_markdown_code_and_fragments():
    text = textwrap.dedent("""\
        ## Answer
        **Force majeure** is a clause. It excuses performance.
        - Natural disasters like floods count.
        ```python
        x = 1
        ```
        OK.
        """)
    assert split_claims(text) == [
        "Force majeure is a clause.",
        "It excuses performance.",
        "Natural disasters like floods count.",
    ]


def test_split_claims_caps_count():
    assert len(split_claims("This is a sentence. " * 50)) == 20


def test_nli_aggregation():
    ev = HallucinationEvaluator(classifier=fake_nli)
    premise = "Gradient descent updates parameters opposite to the loss gradient."

    supported = ev.score("Gradient descent updates parameters opposite to the gradient.", premise)
    assert supported["nli_label"] == "entailment"
    assert supported["entailment_rate"] == 1.0
    assert supported["hallucination_flag"] is False

    mixed = ev.score(
        "Gradient descent updates parameters opposite to the gradient. It was invented on Mars by robots.",
        premise,
    )
    assert mixed["nli_label"] == "neutral"
    assert mixed["entailment_rate"] == 0.5
    assert mixed["hallucination_flag"] is False

    contradicted = ev.score("Gradient descent does not update parameters at all.", premise)
    assert contradicted["nli_label"] == "contradiction"
    assert contradicted["hallucination_flag"] is True
    assert contradicted["nli_confidence"] == 0.9


def test_nli_empty_response_is_not_scored():
    ev = HallucinationEvaluator(classifier=fake_nli)
    out = ev.score("", "premise text here")
    assert out["hallucination_flag"] is None and out["num_claims"] == 0


def test_extract_code():
    assert extract_code("Sure:\n```python\ndef f():\n    return 1\n```\nDone") == "def f():\n    return 1"
    assert extract_code("```\nx = 2\n```") == "x = 2"
    assert extract_code("x = 3") == "x = 3"
    assert extract_code("```text\nnote\n```\n```py\ny = 4\n```") == "y = 4"


def test_code_execution_pass_fail_timeout(monkeypatch):
    ev = CodeExecutionEvaluator(timeout_s=3)
    assert ev.score("```python\ndef add(a, b):\n    return a + b\n```", "assert add(2, 3) == 5")["tests_passed"] is True
    failed = ev.score("def add(a, b):\n    return a - b", "assert add(2, 3) == 5")
    assert failed["tests_passed"] is False and "AssertionError" in failed["exec_error"]
    timed_out = ev.score("while True:\n    pass", "pass")
    assert timed_out["tests_passed"] is False and "timed out" in timed_out["exec_error"]

    monkeypatch.setenv("LLM_EVAL_DISABLE_CODE_EXEC", "1")
    assert ev.score("x = 1", "pass")["tests_passed"] is None


def test_code_execution_cannot_see_parent_environment(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "leak")
    ev = CodeExecutionEvaluator()
    out = ev.score("import os", "assert 'SUPER_SECRET_TOKEN' not in os.environ")
    assert out["tests_passed"] is True


# --------------------------------------------------------- loader/summary


def test_loader_validates(tmp_path):
    (tmp_path / "a.yaml").write_text("- id: x1\n  prompt: p\n")
    with pytest.raises(DatasetError, match="missing expected"):
        load_dataset(str(tmp_path))

    (tmp_path / "a.yaml").write_text("- id: x1\n  prompt: p\n  expected: e\n")
    (tmp_path / "b.yaml").write_text("- id: x1\n  prompt: p2\n  expected: e2\n")
    with pytest.raises(DatasetError, match="Duplicate"):
        load_dataset(str(tmp_path))

    with pytest.raises(DatasetError, match="not found"):
        load_dataset(str(tmp_path / "nope"))


def test_loader_system_prompt_inheritance(tmp_path):
    (tmp_path / "main.yaml").write_text(
        "system_prompt: Be brief.\nsamples:\n  - {id: a, prompt: p, expected: e}\n"
    )
    (tmp_path / "main_generated.yaml").write_text("- {id: b, prompt: p, expected: e}\n")
    samples = {s.id: s for s in load_dataset(str(tmp_path))}
    assert samples["a"].system_prompt == "Be brief."
    assert samples["b"].system_prompt == "Be brief."


def test_percentile_matches_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10, 20], 95) == pytest.approx(19.5)


def test_aggregate_skips_inapplicable_metrics():
    rows = [
        {"latency_ms": 10, "tests_passed": True, "rougeL": None, "hallucination_flag": None},
        {"latency_ms": 30, "tests_passed": False, "rougeL": None, "hallucination_flag": None},
    ]
    agg = aggregate_rows(rows)
    assert agg["pass_rate"] == 0.5
    assert agg["rougeL_mean"] is None
    assert agg["halluc_rate"] is None  # not 0% — the metric does not apply
    assert agg["latency_p50"] == 20


# ----------------------------------------------------------------- runner


class RecordingAdapter(BaseAdapter):
    active = 0
    max_active = 0

    async def ask(self, prompt, system_prompt=""):
        RecordingAdapter.active += 1
        RecordingAdapter.max_active = max(RecordingAdapter.max_active, RecordingAdapter.active)
        await asyncio.sleep(0.01)
        RecordingAdapter.active -= 1
        if self.model_name == "bad":
            raise TimeoutError("took too long")
        return ModelResponse(self.model_name, f"[{system_prompt}] reply", 1.0, 1, 1)


SAMPLES = [
    EvalSample(id="s1", prompt="p1", expected="reply", domain="t", system_prompt="Be brief."),
    EvalSample(id="s2", prompt="p2", expected="reply", domain="t"),
]


@pytest.mark.parametrize("parallel, expected_max", [(False, 1), (True, 2)])
def test_runner_concurrency_mode(parallel, expected_max):
    RecordingAdapter.max_active = 0
    runner = PromptRunner([RecordingAdapter("a"), RecordingAdapter("b")], use_mlflow=False, parallel=parallel)
    outcome = asyncio.run(runner.run(SAMPLES))
    assert RecordingAdapter.max_active == expected_max
    assert len(outcome.results) == 4 and outcome.status == "completed"


def test_runner_passes_system_prompt_and_records_failures():
    runner = PromptRunner([RecordingAdapter("good"), RecordingAdapter("bad")], use_mlflow=False)
    outcome = asyncio.run(runner.run(SAMPLES))
    assert outcome.status == "partial"
    assert [r.response_text for r in outcome.results] == ["[Be brief.] reply", "[] reply"]
    assert [(f.sample_id, f.model_name) for f in outcome.failures] == [("s1", "bad"), ("s2", "bad")]
    assert "TimeoutError: took too long" in outcome.failure_summary()


def test_runner_all_failed():
    outcome = asyncio.run(PromptRunner([RecordingAdapter("bad")], use_mlflow=False).run(SAMPLES))
    assert outcome.status == "failed" and not outcome.results


def test_runner_logs_to_mlflow(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        outcome = asyncio.run(PromptRunner([RecordingAdapter("good")], use_mlflow=True).run(SAMPLES, run_name="t"))
        runs = mlflow.search_runs(experiment_ids=["0"])
    finally:
        mlflow.set_tracking_uri(None)
    assert outcome.status == "completed"
    assert len(runs) == 1
    assert runs.iloc[0]["params.eval_version"] == "2"
    assert "metrics.good_rougeL_mean" in runs.columns
