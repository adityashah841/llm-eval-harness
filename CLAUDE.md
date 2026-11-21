# CLAUDE.md — LLM Evaluation Harness

## Project Overview

A fully local LLM evaluation framework. It compares open-source models served by
Ollama on accuracy (ROUGE for text, unit tests for code), grounding (sentence-level
NLI) and latency. No API keys, no paid services. README.md is the user-facing
source of truth for how scoring works.

## Repository

GitHub: https://github.com/adityashah841/llm-eval-harness
Owner: adityashah841
Do not give claude the co-ownership. Let it be completely owned by adityashah841 and the repository must not show any commits by claude.

## Tech Stack

- **Python 3.11** (managed by uv: `uv sync`, `uv run`)
- **Ollama** for model serving (default http://localhost:11434, `OLLAMA_BASE_URL` to change)
- **httpx** async calls to Ollama (`OLLAMA_TIMEOUT_S`, default 600 s)
- **rouge-score** for ROUGE-1/2/L (no runtime download)
- **transformers** + `cross-encoder/nli-deberta-v3-small` for sentence-level NLI (CPU)
- **MLflow** experiment tracking (default store: `mlflow.db` in the working directory)
- **FastAPI** API, **SQLite** run history (`data/metrics.db`, not committed)
- **Streamlit** dashboard, **React + Vite + TypeScript** frontend
- **Click** CLI, **pytest** tests

## Evaluation methodology (EVAL_VERSION = 2)

Defined in `llm_eval/runner/prompt_runner.py`. Bump `EVAL_VERSION` (and document it in
the README's "Methodology changes") whenever a scoring change makes new scores
incomparable with old ones; trend charts and drift checks only compare runs with the
same version.

- Dataset YAML may set `system_prompt` (file level), and per sample `context` (NLI
  premise, e.g. the passage being summarized) and `tests` (asserts → scored by
  execution, not ROUGE/NLI).
- Decoding is greedy: `temperature 0`, `seed 42`, `num_predict 1024`. Scores still vary
  slightly across machines (CPU floating point), so don't describe runs as fully deterministic.
- `<think>…</think>` is stripped from responses before scoring.
- Models are queried sequentially by default; `--parallel` / `"parallel": true` overlaps them.
- Failed model calls are recorded in `RunOutcome.failures`; run status is
  completed / partial / failed.

## Project Structure

```
llm_eval/
├── adapters/      base.py (BaseAdapter, ModelResponse), ollama_adapter.py
├── evaluators/    rouge_evaluator.py, hallucination_evaluator.py (sentence NLI),
│                  code_execution_evaluator.py (unit tests in a subprocess)
├── runner/        dataset_loader.py, prompt_runner.py (EvalResult, RunOutcome, EVAL_VERSION),
│                  summary.py (shared aggregation/percentiles)
└── dashboard/app.py   Streamlit (port 8501)
api/               FastAPI: main.py, config.py (env vars), routes/, schemas.py, store.py,
                   db.py (SQLite + schema migration via PRAGMA user_version)
frontend/          React app (pages/, components/, hooks/, lib/stats.ts mirrors summary.py)
datasets/          legal_qa (5), summarization (4, with context), code_gen (4, with tests)
scripts/           run_eval.py (CLI), generate_tests.py, scheduled_benchmark.py
tests/             offline suite (stubs Ollama + NLI) and requires_models tests
.github/workflows/ eval.yml (CI), scheduled-benchmark.yml (daily → benchmark-data branch)
```

## Key Commands

```bash
uv sync                                   # install
ollama serve                              # if Ollama isn't already running
uv run pytest                             # offline tests
uv run pytest -m requires_models          # real NLI model tests (needs Hugging Face access)
uv run python scripts/run_eval.py -d datasets/legal_qa -m gemma3:4b --smoke-test --no-mlflow
uv run streamlit run llm_eval/dashboard/app.py
uv run uvicorn api.main:app --reload --port 8000
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
cd frontend && npm install && npm run dev  # also: npm run lint, npm run build
```

On Windows, Ollama is installed at `C:\Users\adity\AppData\Local\Programs\Ollama\ollama.exe`
and does not auto-start. The uv venv is activated with `.venv\Scripts\Activate.ps1`.

## Data and history

- `data/`, `outputs/`, `mlruns/`, `mlflow.db` and `*.csv` are gitignored.
- The daily benchmark's history is on the `benchmark-data` branch (`metrics.db`). Load it with
  `git fetch origin benchmark-data && git show origin/benchmark-data:metrics.db > data/metrics.db`.
- `api/db.py` migrates older databases in place; pre-v2 runs are tagged `eval_version` 1.

## Known caveats

- ROUGE is lexical. It's only meaningful because the datasets ask for short answers.
- NLI flags contradictions with the premise. It can't catch false claims the premise
  doesn't mention.
- Generated code is executed with resource limits, but that is not a security sandbox
  (`LLM_EVAL_DISABLE_CODE_EXEC=1` turns it off).
- Latency on GitHub-hosted runners varies by machine; treat CI latency as indicative.
- The DeBERTa `UNEXPECTED key` load warning and the Windows symlink warning from
  huggingface_hub are harmless.
