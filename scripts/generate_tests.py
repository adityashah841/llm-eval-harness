#!/usr/bin/env python3
"""
Draft extra test cases with a local Ollama model (no API key needed).

Usage:
    uv run python scripts/generate_tests.py --domain legal_qa --count 10
    uv run python scripts/generate_tests.py --domain legal_qa --count 5 --model gemma3:4b

Writes to datasets/<domain>/<domain>_generated.yaml, appending to anything
already there. Generated reference answers come from a model and can be
wrong: review every case before relying on scores computed against it.
Malformed or duplicate-id cases are dropped and reported.

For code_gen, generated cases have no `tests`, so they are scored with
ROUGE/NLI like text rather than executed — add asserts by hand to score them
by execution.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

import click
import httpx
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_eval.runner.dataset_loader import DatasetError, load_dataset  # noqa: E402

SYSTEM_PROMPT = """You are an expert LLM evaluation engineer. Generate challenging
test cases that expose weaknesses in language models. Each test case must:
1. Be genuinely difficult — avoid simple factual recall
2. Have a clear, verifiable expected answer of one or two sentences
3. Test for hallucination, reasoning errors, or overconfidence

Output ONLY valid YAML. No preamble, no explanation, no code fences.
Use exactly this schema:
- id: <domain>_auto_<three_digit_number>
  prompt: "<the question>"
  expected: "<the reference answer>"
  domain: <domain>
  tags: [<tag1>, <tag2>]
  difficulty: <easy|medium|hard>"""

REQUIRED = ("id", "prompt", "expected")


async def generate(domain: str, count: int, start_index: int, model: str, base_url: str, timeout_s: float) -> list:
    prompt = (
        f"Generate {count} test cases for the domain: {domain}.\n"
        f"Start IDs at {domain}_auto_{start_index:03d}."
    )
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.7},
            },
        )
        response.raise_for_status()
        raw_text = response.json()["message"]["content"]

    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    raw_text = re.sub(r"```(?:yaml)?", "", raw_text).strip()
    data = yaml.safe_load(raw_text)
    return data if isinstance(data, list) else []


@click.command()
@click.option("--domain", "-d", required=True, help="Dataset folder name, e.g. legal_qa")
@click.option("--count", "-n", default=10, help="Number of test cases to request")
@click.option("--model", default="llama3.1:70b", show_default=True, help="Ollama model that writes the cases")
@click.option("--timeout", "timeout_s", default=1800.0, show_default=True,
              help="Seconds to wait for the model (70B on CPU can take well over 10 minutes)")
@click.option("--output-dir", default="datasets/", help="Base datasets directory")
def main(domain, count, model, timeout_s, output_dir):
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    click.echo(f"Generating {count} test cases for '{domain}' with {model} at {base_url} ...")

    folder = Path(output_dir) / domain
    out_file = folder / f"{domain}_generated.yaml"
    folder.mkdir(parents=True, exist_ok=True)

    existing = []
    if out_file.exists():
        existing = yaml.safe_load(out_file.read_text(encoding="utf-8")) or []
    try:
        taken_ids = {s.id for s in load_dataset(str(folder))}
    except DatasetError:
        taken_ids = {str(c.get("id")) for c in existing if isinstance(c, dict)}

    candidates = asyncio.run(generate(domain, count, len(existing) + 1, model, base_url, timeout_s))

    accepted, rejected = [], 0
    for case in candidates:
        if not isinstance(case, dict) or any(not case.get(k) for k in REQUIRED):
            rejected += 1
            continue
        case["id"] = str(case["id"])
        if case["id"] in taken_ids:
            rejected += 1
            continue
        case.setdefault("domain", domain)
        taken_ids.add(case["id"])
        accepted.append(case)

    combined = existing + accepted
    out_file.write_text(yaml.dump(combined, allow_unicode=True, default_flow_style=False, sort_keys=False),
                        encoding="utf-8")
    click.echo(f"Added {len(accepted)} case(s), dropped {rejected} malformed/duplicate → {out_file}")
    click.echo(f"Total in file: {len(combined)}. Review the reference answers before using them.")


if __name__ == "__main__":
    main()
