#!/usr/bin/env python3
"""
Generate adversarial test cases using a local Ollama model.
No API key required — uses llama3.1:70b running locally.

Usage:
    python scripts/generate_tests.py --domain legal_qa --count 10
    python scripts/generate_tests.py --domain code_gen --count 5
"""
import asyncio
import click
import httpx
import yaml
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

SYSTEM_PROMPT = """You are an expert LLM evaluation engineer. Generate adversarial,
challenging test cases that expose weaknesses in language models. Each test case must:
1. Be genuinely difficult — avoid simple factual recall
2. Have a clear, verifiable expected answer
3. Test for hallucination, reasoning errors, or overconfidence

Output ONLY valid YAML. No preamble, no explanation, no code fences.
Use exactly this schema:
- id: <domain>_auto_<three_digit_number>
  prompt: "<the question>"
  expected: "<the reference answer>"
  domain: <domain>
  tags: [<tag1>, <tag2>]
  difficulty: <easy|medium|hard>"""


async def generate(domain: str, count: int, start_index: int = 1) -> list:
    prompt = (
        f"Generate {count} adversarial test cases for the domain: {domain}.\n"
        f"Use your knowledge of the domain.\n"
        f"Start IDs at {domain}_auto_{start_index:03d}."
    )

    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.1:70b",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        raw_text = response.json()["message"]["content"]

    # Strip any accidental code fences
    raw_text = re.sub(r"```(?:yaml)?", "", raw_text).strip()
    return yaml.safe_load(raw_text) or []


@click.command()
@click.option("--domain", "-d", required=True,
              help="Dataset domain (e.g. legal_qa, code_gen, summarization)")
@click.option("--count", "-n", default=10, help="Number of test cases to generate")
@click.option("--output-dir", default="datasets/", help="Base datasets directory")
def main(domain, count, output_dir):
    click.echo(f"Generating {count} test cases for domain '{domain}' using llama3.1:70b...")
    click.echo("This uses your local Ollama model — no API key needed.")

    out_file = Path(output_dir) / domain / f"{domain}_generated.yaml"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if out_file.exists():
        with open(out_file) as f:
            existing = yaml.safe_load(f) or []

    start_index = len(existing) + 1
    new_samples = asyncio.run(generate(domain, count, start_index))

    combined = existing + new_samples
    with open(out_file, "w") as f:
        yaml.dump(combined, f, allow_unicode=True, default_flow_style=False)

    click.echo(f"Generated {len(new_samples)} test cases → {out_file}")
    click.echo(f"Total in file: {len(combined)}")


if __name__ == "__main__":
    main()
