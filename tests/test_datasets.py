"""Checks on the shipped datasets themselves."""
from pathlib import Path

import pytest

from llm_eval.evaluators import CodeExecutionEvaluator
from llm_eval.runner.dataset_loader import load_dataset

ROOT = Path(__file__).resolve().parent.parent / "datasets"
DATASETS = sorted(p for p in ROOT.iterdir() if p.is_dir())
CODE_SAMPLES = [s for d in DATASETS for s in load_dataset(str(d)) if s.tests]


@pytest.mark.parametrize("folder", DATASETS, ids=lambda p: p.name)
def test_dataset_loads_with_system_prompt(folder):
    samples = load_dataset(str(folder))
    assert samples
    assert all(s.system_prompt for s in samples)


def test_summarization_samples_carry_their_source():
    for s in load_dataset(str(ROOT / "summarization")):
        assert s.context and s.context in s.prompt


@pytest.mark.parametrize("sample", CODE_SAMPLES, ids=lambda s: s.id)
def test_reference_solutions_pass_their_tests(sample):
    """If a reference solution fails, the tests (not the models) are wrong."""
    out = CodeExecutionEvaluator().score(f"```python\n{sample.expected}\n```", sample.tests)
    assert out["tests_passed"] is True, out["exec_error"]
