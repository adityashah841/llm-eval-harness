#!/usr/bin/env python3
"""
CLI entry point for running evaluations from the terminal.

Examples:
    # Two models on legal QA
    uv run python scripts/run_eval.py -d datasets/legal_qa -m gemma3:4b -m phi4

    # Smoke test: first 2 samples, no MLflow (what CI runs)
    uv run python scripts/run_eval.py -d datasets/legal_qa -m gemma3:4b --smoke-test --no-mlflow

Exit code: 0 if every model call succeeded, 1 if any failed (details are
printed and the successful results are still written to the CSV).
"""
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_eval.adapters import OllamaAdapter  # noqa: E402
from llm_eval.runner.dataset_loader import load_dataset  # noqa: E402
from llm_eval.runner.prompt_runner import EVAL_VERSION, PromptRunner  # noqa: E402
from llm_eval.runner.summary import summarize_by_model  # noqa: E402


@click.command()
@click.option("--dataset", "-d", default="datasets/legal_qa",
              help="Path to dataset directory or YAML file")
@click.option("--models", "-m", multiple=True, default=["gemma3:4b"],
              help="Ollama model name — repeat for multiple, e.g. -m gemma3:4b -m phi4")
@click.option("--smoke-test", is_flag=True,
              help="Run only the first 2 samples (fast CI check)")
@click.option("--parallel", is_flag=True,
              help="Query all models for a sample at once (faster; latency then includes queueing)")
@click.option("--output", "-o", default="outputs/results.csv",
              help="Path to write per-sample CSV results")
@click.option("--summary-json", default=None,
              help="Optional path to also write per-model aggregate scores as JSON")
@click.option("--no-mlflow", is_flag=True,
              help="Disable MLflow logging for this run")
def main(dataset, models, smoke_test, parallel, output, summary_json, no_mlflow):
    samples = load_dataset(dataset)
    if smoke_test:
        samples = samples[:2]
        click.echo("[smoke-test] Running on first 2 samples only.")

    click.echo(f"Dataset : {dataset} ({len(samples)} samples)")
    click.echo(f"Models  : {list(models)}")
    click.echo(f"Mode    : {'parallel' if parallel else 'sequential'}, eval_version {EVAL_VERSION}")
    click.echo(f"MLflow  : {'disabled' if no_mlflow else 'enabled'}")

    adapters = [OllamaAdapter(m) for m in models]
    runner = PromptRunner(adapters, use_mlflow=not no_mlflow, parallel=parallel)
    outcome = asyncio.run(runner.run(samples))

    rows = [asdict(r) for r in outcome.results]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    click.echo(f"\nResults saved to {output}")

    summary = summarize_by_model(rows)
    if summary:
        click.echo("\nAggregate scores (None = metric does not apply to this dataset):")
        click.echo(pd.DataFrame(summary).T.to_string())

    failed_tests = [r for r in outcome.results if r.tests_passed is False]
    if failed_tests:
        click.echo("\nFailed unit tests:")
        for r in failed_tests:
            click.echo(f"  {r.model_name} / {r.sample_id}: {r.exec_error}")
    contradicted = [r for r in outcome.results if r.hallucination_flag]
    if contradicted:
        click.echo("\nResponses with a sentence contradicted by the premise:")
        for r in contradicted:
            click.echo(f"  {r.model_name} / {r.sample_id}")

    if summary_json:
        Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_json).write_text(json.dumps({
            "dataset": dataset,
            "eval_version": EVAL_VERSION,
            "status": outcome.status,
            "failures": [asdict(f) for f in outcome.failures],
            "models": summary,
        }, indent=2))

    if outcome.failures:
        click.echo(f"\n{outcome.failure_summary()}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
