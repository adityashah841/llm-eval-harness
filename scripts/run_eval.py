#!/usr/bin/env python3
"""
CLI entry point for running evaluations from the terminal.

Examples:
    # Run full benchmark on legal QA with two models
    python scripts/run_eval.py --dataset datasets/legal_qa --models gemma3:4b phi4

    # Smoke test with a single sample (useful for CI)
    python scripts/run_eval.py --dataset datasets/legal_qa --models gemma3:4b --smoke-test

    # All four models on summarization
    python scripts/run_eval.py --dataset datasets/summarization \
        --models gemma3:4b phi4 deepseek-r1:14b llama3.1:70b
"""
import asyncio
import click
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_eval.adapters import OllamaAdapter
from llm_eval.runner.prompt_runner import PromptRunner
from llm_eval.runner.dataset_loader import load_dataset
from dataclasses import asdict


@click.command()
@click.option("--dataset", "-d", default="datasets/legal_qa",
              help="Path to dataset directory or YAML file")
@click.option("--models", "-m", multiple=True, default=["gemma3:4b"],
              help="Ollama model names — repeat for multiple, e.g. -m gemma3:4b -m phi4")
@click.option("--smoke-test", is_flag=True,
              help="Run only the first 2 samples (fast CI check)")
@click.option("--output", "-o", default="outputs/results.csv",
              help="Path to write CSV results")
@click.option("--no-mlflow", is_flag=True,
              help="Disable MLflow logging for this run")
def main(dataset, models, smoke_test, output, no_mlflow):
    samples = load_dataset(dataset)
    if smoke_test:
        samples = samples[:2]
        click.echo("[smoke-test] Running on first 2 samples only.")

    click.echo(f"Dataset : {dataset} ({len(samples)} samples)")
    click.echo(f"Models  : {list(models)}")
    click.echo(f"MLflow  : {'disabled' if no_mlflow else 'enabled'}")

    adapters = [OllamaAdapter(m) for m in models]
    runner = PromptRunner(adapters, use_mlflow=not no_mlflow)
    results = asyncio.run(runner.run(samples))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output, index=False)

    click.echo(f"\nResults saved to {output}")
    click.echo("\nAggregate scores:")
    summary = (
        df.groupby("model_name")[["rouge1", "rougeL", "hallucination_flag", "latency_ms"]]
        .mean()
        .round(4)
    )
    click.echo(summary.to_string())


if __name__ == "__main__":
    main()
