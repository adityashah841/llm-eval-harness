# CLAUDE.md — LLM Evaluation Harness

## Project Overview

This is a fully local LLM evaluation framework. It compares open-source models served
by Ollama using ROUGE scores, NLI-based hallucination detection, and latency metrics.
No API keys, no paid services — everything runs on the developer's machine.

## Repository

GitHub: https://github.com/adityashah841/llm-eval-harness
Owner: adityashah841

## Tech Stack

- **Python 3.11** (managed by uv)
- **uv** for dependency management (`uv sync`, `uv run`)
- **Ollama** for local model serving (must be running at http://localhost:11434)
- **httpx** for async HTTP calls to Ollama API
- **HuggingFace evaluate + rouge-score** for ROUGE metrics
- **transformers** (cross-encoder/nli-deberta-v3-small) for hallucination detection
- **MLflow** for experiment tracking
- **Streamlit** for the dashboard UI
- **Click** for CLI entry points

## Models Targeted

- `llama3.1:70b` — ~20 GB, best quality
- `deepseek-r1:14b` — ~9 GB, reasoning
- `phi4` — ~9 GB, strong reasoning
- `gemma3:4b` — ~3 GB, fast baseline (start here for testing)

## Project Structure

```
llm_eval/
├── adapters/
│   ├── base.py              # BaseAdapter ABC + ModelResponse dataclass
│   ├── ollama_adapter.py    # OllamaAdapter (async httpx calls)
│   └── __init__.py
├── evaluators/
│   ├── rouge_evaluator.py   # ROUGE-1/2/L via HuggingFace evaluate
│   ├── hallucination_evaluator.py  # NLI via DeBERTa-v3-small
│   └── __init__.py
├── runner/
│   ├── dataset_loader.py    # Loads YAML datasets into EvalSample dataclasses
│   ├── prompt_runner.py     # Async runner + EvalResult + MLflow logging
│   └── __init__.py
└── dashboard/
    ├── app.py               # Streamlit dashboard
    └── __init__.py
datasets/
├── legal_qa/legal_qa.yaml   # 5 legal QA pairs
├── code_gen/code_gen.yaml   # 4 code generation tasks
└── summarization/summarization.yaml  # 4 summarization tasks
scripts/
├── run_eval.py              # Click CLI entry point
└── generate_tests.py        # Generates adversarial tests via local llama3.1:70b
.github/workflows/eval.yml   # GitHub Actions smoke test (gemma3:4b, 2 samples)
Dockerfile                   # Containerized dashboard
docker-compose.yml           # Dashboard + MLflow services
```

## Key Commands

```bash
# Install dependencies
uv sync

# Launch Streamlit dashboard
uv run streamlit run llm_eval/dashboard/app.py

# CLI benchmark
uv run python scripts/run_eval.py --dataset datasets/legal_qa --models gemma3:4b phi4

# Smoke test (CI-style, 2 samples, no MLflow)
uv run python scripts/run_eval.py --dataset datasets/legal_qa --models gemma3:4b --smoke-test --no-mlflow --output outputs/smoke_test.csv

# View MLflow UI
uv run mlflow ui

# Generate more test cases with local model
python scripts/generate_tests.py --domain legal_qa --count 20
```

## Ollama Setup (required before running)

1. Install from OllamaSetup.exe (already downloaded to Desktop/claude/)
2. Ollama auto-starts on Windows after install
3. Pull models (start with smallest):
   ```
   ollama pull gemma3:4b
   ollama pull phi4
   ollama pull deepseek-r1:14b
   ollama pull llama3.1:70b
   ```
4. Verify: `ollama list`

## Setup Progress

- [x] Step 1 — Environment checks (git 2.47, Python 3.11 via uv, uv 0.11.6, gh 2.65.0)
- [ ] Step 2 — Pull Ollama models (BLOCKED: Ollama not yet installed)
- [x] Step 3 — GitHub repo created and cloned (adityashah841/llm-eval-harness)
- [x] Step 4 — uv init, python pin 3.11, all dependencies installed
- [x] Step 5 — Directory structure created
- [x] Step 6 — Model adapters written (base.py, ollama_adapter.py)
- [x] Step 7 — Evaluators written (rouge, hallucination/NLI)
- [x] Step 8 — Dataset loader written
- [x] Step 9 — Prompt runner written
- [x] Step 10 — Test case generator script written
- [x] Step 11 — Streamlit dashboard written
- [x] Step 12 — CLI entry point written
- [x] Step 13 — Sample datasets written (legal_qa, code_gen, summarization)
- [x] Step 14 — Docker files written
- [x] Step 15 — GitHub Actions CI written
- [x] Step 16 — README written
- [ ] Step 17 — Commit and push to GitHub (pending)
- [ ] Step 18 — Final smoke test (requires Ollama)

## Important Notes

- The `.venv/` is managed by uv — activate with `.venv\Scripts\Activate.ps1` (PowerShell)
  or `source .venv/Scripts/activate` (bash)
- `outputs/` and `*.csv` are gitignored — results stay local
- `mlruns/` is gitignored — MLflow data stays local
- The NLI hallucination model (DeBERTa-v3-small, ~85 MB) downloads automatically on
  first run from HuggingFace
- Docker's `extra_hosts: host.docker.internal:host-gateway` allows the container to
  reach Ollama running on the host
