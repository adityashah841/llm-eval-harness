# CLAUDE.md — LLM Evaluation Harness

## Project Overview

This is a fully local LLM evaluation framework. It compares open-source models served
by Ollama using ROUGE scores, NLI-based hallucination detection, and latency metrics.
No API keys, no paid services — everything runs on the developer's machine.

## Repository

GitHub: https://github.com/adityashah841/llm-eval-harness
Owner: adityashah841
Do not give claude the co-ownership. Let it be completely owned by adityashah841 and the repository must not show any commits by claude.

## Tech Stack

- **Python 3.11** (managed by uv)
- **uv** for dependency management (`uv sync`, `uv run`)
- **Ollama** for local model serving (must be running at http://localhost:11434)
- **httpx** for async HTTP calls to Ollama API (timeout: 600s)
- **HuggingFace evaluate + rouge-score** for ROUGE metrics
- **transformers** (cross-encoder/nli-deberta-v3-small) for hallucination detection
- **MLflow** for experiment tracking
- **Streamlit** for the dashboard UI
- **Click** for CLI entry points

## Models Available (all pulled and ready)

- `llama3.1:70b` — ~42 GB on disk, best quality, very slow on CPU (~20+ min/sample)
- `deepseek-r1:14b` — ~9 GB, reasoning with chain-of-thought
- `phi4` — ~9 GB, strong reasoning, ~6-8 min/sample on CPU
- `gemma3:4b` — ~3.3 GB, fast baseline, ~2.5 min/sample on CPU ← start here

## Project Structure

```
llm_eval/
├── adapters/
│   ├── base.py              # BaseAdapter ABC + ModelResponse dataclass
│   ├── ollama_adapter.py    # OllamaAdapter (async httpx, 600s timeout)
│   └── __init__.py
├── evaluators/
│   ├── rouge_evaluator.py   # ROUGE-1/2/L via HuggingFace evaluate
│   ├── hallucination_evaluator.py  # NLI via DeBERTa-v3-small (CPU)
│   └── __init__.py
├── runner/
│   ├── dataset_loader.py    # Loads YAML datasets into EvalSample dataclasses
│   ├── prompt_runner.py     # Async runner + EvalResult + MLflow logging
│   └── __init__.py
└── dashboard/
    ├── app.py               # Streamlit dashboard (port 8501)
    └── __init__.py
datasets/
├── legal_qa/legal_qa.yaml         # 5 legal QA pairs
├── code_gen/code_gen.yaml         # 4 coding tasks
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
# Start Ollama (if not already running)
"C:\Users\adity\AppData\Local\Programs\Ollama\ollama.exe" serve &

# Install dependencies
uv sync

# Launch Streamlit dashboard
uv run streamlit run llm_eval/dashboard/app.py
# → opens at http://localhost:8501

# CLI benchmark
uv run python scripts/run_eval.py --dataset datasets/code_gen --models gemma3:4b

# Smoke test (2 samples, no MLflow)
uv run python scripts/run_eval.py --dataset datasets/legal_qa --models gemma3:4b --smoke-test --no-mlflow --output outputs/smoke_test.csv

# View MLflow experiment history
uv run mlflow ui
# → opens at http://localhost:5000

# Generate more test cases with local model
python scripts/generate_tests.py --domain legal_qa --count 20
```

## Ollama Setup

- Installed at: `C:\Users\adity\AppData\Local\Programs\Ollama\ollama.exe`
- Does NOT auto-start — must run `ollama serve` manually each session
- All 4 models are already pulled and ready

## Current Status (fully working)

- [x] All code written and pushed to GitHub
- [x] All 4 Ollama models pulled (gemma3:4b, phi4, deepseek-r1:14b, llama3.1:70b)
- [x] Smoke test passed (gemma3:4b, 2 legal QA samples, ROUGE + hallucination scores)
- [x] Dashboard confirmed working (gemma3:4b on code_gen dataset)
- [x] All fixes pushed to GitHub (2 commits on main)

## Known Issues & Fixes Applied

### Timeout on large models (phi4, deepseek-r1, llama3.1)
- **Problem:** httpx default 180s timeout was too short for CPU inference
- **Fix:** Raised to 600s in `llm_eval/adapters/ollama_adapter.py`
- **Note:** phi4 takes ~6-8 min/sample on CPU; llama3.1:70b takes 20+ min/sample

### KeyError: 'model_name' in dashboard
- **Problem:** If all model calls time out, results list is empty → DataFrame has no columns → groupby crashes
- **Fix:** Added empty DataFrame guard in `llm_eval/dashboard/app.py` before the groupby

## Performance Expectations (CPU, no GPU)

| Model | Time/sample | 4-sample dataset |
|---|---|---|
| gemma3:4b | ~2.5 min | ~10 min |
| phi4 | ~6-8 min | ~25-32 min |
| deepseek-r1:14b | ~6-8 min | ~25-32 min |
| llama3.1:70b | ~20+ min | ~80+ min |

Running 2 models in parallel: time = slowest model × num_samples (not sum of both).

## Important Notes

- The `.venv/` is managed by uv — activate with `.venv\Scripts\Activate.ps1` (PowerShell)
- `outputs/` and `*.csv` are gitignored — results stay local
- `mlruns/` is gitignored — MLflow data stays local
- The NLI model (DeBERTa-v3-small, ~85 MB) is cached after first download from HuggingFace
- The symlink warning on Windows and DeBERTa `UNEXPECTED key` warning are both harmless
- Docker's `extra_hosts: host.docker.internal:host-gateway` lets the container reach host Ollama
