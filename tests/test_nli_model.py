"""Sanity checks against the real NLI model (downloads ~0.1B params from
Hugging Face). Skipped by default; run with `uv run pytest -m requires_models`.
CI runs these in the model job.

These pin the behaviour the README describes: a faithful sentence is
entailed, a direct contradiction is flagged, and an unrelated claim is not
counted as supported.
"""
import pytest

from llm_eval.evaluators.hallucination_evaluator import HallucinationEvaluator

pytestmark = pytest.mark.requires_models


@pytest.fixture(scope="module")
def nli():
    return HallucinationEvaluator()


PREMISE = (
    "Gradient descent is an iterative optimization algorithm that adjusts model parameters "
    "in the direction opposite to the gradient of the loss function, gradually reducing "
    "prediction error over many training steps."
)


def test_faithful_summary_is_entailed(nli):
    out = nli.score(
        "Gradient descent iteratively adjusts model parameters opposite to the loss gradient to reduce error.",
        PREMISE,
    )
    assert out["nli_label"] == "entailment"
    assert out["hallucination_flag"] is False


def test_contradiction_is_flagged(nli):
    out = nli.score(
        "Gradient descent adjusts model parameters in the same direction as the gradient of the loss function.",
        PREMISE,
    )
    assert out["hallucination_flag"] is True
    assert out["nli_label"] == "contradiction"


def test_unsupported_claim_is_not_entailed(nli):
    out = nli.score(
        "Gradient descent iteratively adjusts parameters to reduce error. "
        "It was invented by Isaac Newton in 1687 to study planetary orbits.",
        PREMISE,
    )
    assert out["num_claims"] == 2
    assert out["entailment_rate"] < 1.0


def test_qa_reference_premise(nli):
    reference = (
        "Promissory estoppel is a legal doctrine that enforces a promise even without formal "
        "consideration, when one party reasonably relied on the promise to their detriment."
    )
    good = nli.score(
        "Promissory estoppel enforces a promise without consideration when someone reasonably relied on it to their detriment.",
        reference,
    )
    bad = nli.score("Promissory estoppel enforces a promise only when formal consideration was exchanged.", reference)
    assert good["hallucination_flag"] is False
    assert bad["hallucination_flag"] is True
