# LLM Evaluation Harness

A modular, open-source framework for evaluating and comparing local open-source LLMs
across accuracy, hallucination rate, and latency. Started as a Streamlit + MLflow tool
for fast local experimentation, and is growing into a full-stack platform (REST API,
React frontend, persistent metrics, observability) on top of the same evaluation
engine.

**Zero API keys. Zero cost. Runs entirely on your machine via Ollama.**

![CI](https://github.com/adityashah841/llm-eval-harness/actions/workflows/eval.yml/badge.svg)

## Models compared

| Model | Size | RAM needed | Strength |
|---|---|---|---|
| Llama 3.1 70B | 70B | ~20 GB | Best overall quality |
| DeepSeek-R1 14B | 14B | ~9 GB | Reasoning, shows chain-of-thought |
| Phi-4 14B | 14B | ~9 GB | Strong reasoning, synthetic-data trained |
| Gemma 3 4B | 4B | ~3 GB | Fast, efficient baseline |

## What it measures

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — token overlap with the reference answer
- **Hallucination flag** — NLI-based grounding check using DeBERTa-v3-small (runs locally)
- **Latency** — wall-clock response time, p50 and p95 across the dataset
- **MLflow** — every run logged for experiment comparison

## Quick start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com)
- [uv](https://github.com/astral-sh/uv): `pip install uv`

### Install

```bash
git clone https://github.com/adityashah841/llm-eval-harness
cd llm-eval-harness
uv sync
```

### Pull models

```bash
ollama pull gemma3:4b        # 3 GB  — start here
ollama pull phi4             # 9 GB
ollama pull deepseek-r1:14b  # 9 GB
ollama pull llama3.1:70b     # 20 GB
```

### Launch the dashboard

```bash
uv run streamlit run llm_eval/dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Run a benchmark from the CLI

```bash
# Two models, legal QA dataset
uv run python scripts/run_eval.py \
  --dataset datasets/legal_qa \
  --models gemma3:4b phi4

# All four models
uv run python scripts/run_eval.py \
  --dataset datasets/summarization \
  --models gemma3:4b phi4 deepseek-r1:14b llama3.1:70b
```

### View experiment history

```bash
uv run mlflow ui   # opens at http://localhost:5000
```

### Generate more test cases (uses local llama3.1:70b — no API key)

```bash
python scripts/generate_tests.py --domain legal_qa --count 20
python scripts/generate_tests.py --domain code_gen --count 10
```

### Run with Docker

```bash
# Ollama must be running on your host machine first
docker-compose up
```

## API (productionized layer)

A FastAPI service exposes the same evaluation engine used by the CLI and the
Streamlit dashboard over REST, so other clients (a web frontend, CI jobs,
scripts) can trigger and read evaluation runs without importing Python.
The Streamlit dashboard remains the fastest way to run an ad-hoc, interactive
comparison; the API is for everything that needs to be automated or driven
from outside a Python REPL.

### Launch the API

```bash
uv run uvicorn api.main:app --reload --port 8000
```

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/runs` | Start an eval run (dataset + model list) in the background |
| `GET` | `/runs/{id}` | Run status (`pending` / `running` / `completed` / `failed`) |
| `GET` | `/runs/{id}/results` | Per-sample scores (ROUGE, hallucination flag, latency) once complete |
| `GET` | `/models` | Models available for evaluation, live from Ollama when reachable |
| `GET` | `/metrics/timeseries` | Aggregated per-model metrics across historical MLflow runs |

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"dataset": "legal_qa", "models": ["gemma3:4b"], "smoke_test": true}'
```

### Run the API tests

```bash
uv run pytest tests/test_api.py
```

The `api/` layer only imports from `llm_eval/` — the adapters, evaluators, and
runner package are untouched and still work standalone via the CLI and
Streamlit dashboard.

## Project structure

```
llm_eval/
├── adapters/        # OllamaAdapter + BaseAdapter interface
├── evaluators/      # ROUGE scorer, NLI hallucination detector
├── runner/          # Async parallel runner, dataset loader, EvalResult
└── dashboard/       # Streamlit app (rapid local prototyping)
api/
├── main.py          # FastAPI app
├── routes/          # runs, models, metrics endpoints
├── schemas.py        # Pydantic request/response models
└── store.py          # In-memory run registry
datasets/
├── legal_qa/        # 5 hand-written QA pairs (+ generated ones)
├── code_gen/        # 4 coding tasks
└── summarization/   # 4 summarization tasks
scripts/
├── run_eval.py      # CLI entry point
└── generate_tests.py  # Local test case generator (uses llama3.1:70b)
tests/
└── test_api.py      # FastAPI endpoint tests
```

## Architecture

```
Dataset YAML
    ↓
PromptRunner  ──── asyncio.gather() ────►  OllamaAdapter (llama3.1:70b)
                                       ├►  OllamaAdapter (deepseek-r1:14b)
                                       ├►  OllamaAdapter (phi4)
                                       └►  OllamaAdapter (gemma3:4b)
    ↓
RougeEvaluator + HallucinationEvaluator (DeBERTa-v3 NLI, local)
    ↓
MLflow  +  Streamlit dashboard  +  CSV export
```

All model calls are fired in parallel per sample — total eval time equals the
slowest model, not the sum of all four.

## Key design decisions

- **Adapter pattern** — adding a new model is one new file; the pipeline never changes
- **Async-first** — `asyncio.gather()` parallelises all model calls per sample
- **Fully local** — Ollama serves every model; DeBERTa runs on CPU; no internet needed
  after the initial model downloads
- **Reproducible** — MLflow versions every run with params, metrics, and model list

## Built by

Aditya Shah — [LinkedIn](https://linkedin.com/in/aditya-r-shah26)
