"""Shared fixtures: keep the default test run offline and fast.

* The NLI model is replaced with a tiny rule-based classifier (the real model
  is exercised by tests marked `requires_models`, which CI runs separately).
* Ollama HTTP calls are replaced with a deterministic stub.
* Every test gets its own throwaway SQLite file.
"""
import re

import pytest

import api.db as db
from llm_eval.adapters.base import ModelResponse
from llm_eval.adapters.ollama_adapter import OllamaAdapter
from llm_eval.evaluators.hallucination_evaluator import HallucinationEvaluator
from llm_eval.runner import prompt_runner

_WORD_RE = re.compile(r"[a-z0-9]+")


def fake_nli(pairs):
    """Contradiction if the claim says 'not'/'never', entailment if most of
    its words appear in the premise, otherwise neutral."""
    out = []
    for premise, claim in pairs:
        p = set(_WORD_RE.findall(premise.lower()))
        c = _WORD_RE.findall(claim.lower())
        if {"not", "never"} & set(c):
            out.append({"contradiction": 0.9, "entailment": 0.05, "neutral": 0.05})
        elif c and sum(w in p for w in c) / len(c) >= 0.6:
            out.append({"contradiction": 0.05, "entailment": 0.9, "neutral": 0.05})
        else:
            out.append({"contradiction": 0.05, "entailment": 0.1, "neutral": 0.85})
    return out


STUB_TEXT = "This is a stubbed response for testing."

# Captured before the autouse stub replaces them, for tests of the real adapter.
REAL_ASK = OllamaAdapter.ask
REAL_MODEL_DIGEST = OllamaAdapter.model_digest


@pytest.fixture(autouse=True)
def fake_evaluators(monkeypatch):
    monkeypatch.setitem(prompt_runner._evaluators, "nli", HallucinationEvaluator(classifier=fake_nli))


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_metrics.db")


@pytest.fixture(autouse=True)
def stub_ollama(monkeypatch):
    async def fake_ask(self, prompt, system_prompt=""):
        if self.model_name.startswith("broken"):
            raise RuntimeError("model not found")
        return ModelResponse(
            model_name=self.model_name,
            response_text=STUB_TEXT,
            latency_ms=1.0,
            input_tokens=10,
            output_tokens=5,
            load_ms=0.0,
            generation_ms=0.5,
        )

    async def fake_digest(self):
        return f"sha256:{self.model_name}"

    monkeypatch.setattr(OllamaAdapter, "ask", fake_ask)
    monkeypatch.setattr(OllamaAdapter, "model_digest", fake_digest)
