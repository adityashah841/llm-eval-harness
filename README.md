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

This starts four services: the Streamlit `dashboard` (8501), `mlflow` ui
(5000), the FastAPI `api` (8000), and the `frontend` (5173, served via
nginx). `api` and `dashboard` both reach the host's Ollama through
`host.docker.internal`.

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
| `GET` | `/metrics/timeseries` | Aggregated per-`(run, model)` metrics from the persistent SQLite store |

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

## Frontend (Vite + React + TypeScript)

A `frontend/` app talks to the API above and is the productionized UI for
triggering and reviewing runs — the Streamlit dashboard is still there for
quick one-off local experiments, but this is where the multi-page workflow
(configure → results → compare → observe) lives.

```bash
cd frontend
npm install
cp .env.example .env   # point VITE_API_BASE_URL at your API, if not localhost:8000
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The API must be running
(`uv run uvicorn api.main:app --port 8000`) for the pages to load data.

### Pages

- **Run configuration** — pick a dataset and one or more models (live from
  `GET /models`), trigger a run, and watch its status poll to completion.
- **Results** — per-sample table for a run, filterable by model and by
  hallucination flag.
- **Model comparison** — side-by-side aggregate scores (ROUGE, hallucination
  rate, latency) for every model in a multi-model run.
- **System health** — latency percentile trends, hallucination-rate trend,
  and a rolling accuracy drift flag, sourced from `GET /metrics/timeseries`.
  See [Observability dashboard](#observability-dashboard) below.

### Build & lint

```bash
npm run build   # tsc -b && vite build
npm run lint     # oxlint
```

## Persistent metrics storage

MLflow's file store is great for one-off experiment comparison, but it isn't
something `GET /metrics/timeseries` can query efficiently across many runs
over time. `api/db.py` adds a small SQLite store (`data/metrics.db`, checked
into the repo) that every API-triggered run is persisted to on completion —
the run record plus its per-sample scores. `/metrics/timeseries` reads from
this store, aggregating ROUGE/hallucination/latency per `(run, model)`.

### Scheduled benchmark

`.github/workflows/scheduled-benchmark.yml` runs daily (and on-demand via
`workflow_dispatch`), reusing `eval.yml`'s setup steps (install Ollama, pull
`gemma3:4b`), then runs `scripts/scheduled_benchmark.py` — a fixed 2-sample
legal-QA benchmark — and commits the updated `data/metrics.db` back to `main`
as `github-actions[bot]`. This is what turns `/metrics/timeseries` into an
actual time series instead of only reflecting runs someone happened to
trigger through the API.

```bash
# Run the same benchmark locally
uv run python scripts/scheduled_benchmark.py
```

## Observability dashboard

The frontend's **System health** page (`/health`) turns the persistent metrics
store into a monitoring view, scoped by a model + dataset filter:

- **Latency percentiles over time** — p50 and p95 as a two-series trend line.
- **Hallucination-rate trend** — flagged-response rate per run.
- **Accuracy trend & drift flag** — ROUGE-L per run against a rolling
  threshold: the mean of the previous 5 runs minus a 15% tolerance. If the
  latest run's ROUGE-L falls below that line, the page surfaces a
  `drift detected` banner instead of silently logging a bad run.
- A raw data table beneath the charts keeps every value reachable without
  hovering, for accessibility and for copy-pasting into other tools.

This is read-only against the same `/metrics/timeseries` endpoint the CLI and
scheduled benchmark job feed — no separate observability backend.

## Project structure

```
llm_eval/
├── adapters/        # OllamaAdapter + BaseAdapter interface
├── evaluators/      # ROUGE scorer, NLI hallucination detector
├── runner/          # Async parallel runner, dataset loader, EvalResult
└── dashboard/       # Streamlit app (rapid local prototyping)
api/
├── Dockerfile         # api container image (built by docker-compose and CI deploy)
├── main.py            # FastAPI app
├── routes/            # runs, models, metrics endpoints
├── schemas.py         # Pydantic request/response models
├── store.py           # In-memory run registry (per-process run status)
└── db.py              # SQLite persistence — the source for /metrics/timeseries
data/
└── metrics.db       # Persisted run + sample results (checked into git)
frontend/
├── Dockerfile       # multi-stage build → nginx-served static bundle
├── nginx.conf       # SPA fallback (try_files → index.html)
└── src/
    ├── api/         # typed fetch client for the FastAPI backend
    ├── pages/       # Run configuration, Results, Comparison, System health
    ├── components/  # Nav, TrendChart (shared SVG line chart)
    └── context/     # RunsContext — tracks runs triggered this session
datasets/
├── legal_qa/        # 5 hand-written QA pairs (+ generated ones)
├── code_gen/        # 4 coding tasks
└── summarization/   # 4 summarization tasks
scripts/
├── run_eval.py             # CLI entry point
├── generate_tests.py       # Local test case generator (uses llama3.1:70b)
└── scheduled_benchmark.py  # Fixed benchmark run by the cron workflow, persists to data/metrics.db
tests/
└── test_api.py      # FastAPI endpoint tests
.github/workflows/
├── eval.yml                 # CI: Ollama smoke test, backend tests, frontend
│                             # build/lint, AWS deploy (no-ops until configured)
└── scheduled-benchmark.yml  # Daily cron: re-runs a fixed benchmark, commits data/metrics.db
Dockerfile            # Streamlit dashboard image
docker-compose.yml    # dashboard + mlflow + api + frontend, wired to host Ollama
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

## Deployment (AWS)

CI (`.github/workflows/eval.yml`) runs on every push/PR to `main`:
`eval-smoke-test` (the existing Ollama smoke test), `backend-tests`
(`pytest tests/test_api.py`, no Ollama needed — HTTP calls are stubbed), and
`frontend-checks` (`npm run lint && npm run build`). A `deploy` job runs
after all three succeed, but only on a push to `main`.

`deploy` builds `api/Dockerfile` and `frontend/Dockerfile`, pushes both to
ECR, and redeploys two AWS App Runner services from the new images. It
authenticates via GitHub's OIDC provider (no long-lived AWS keys in
secrets) and is a no-op — it logs a message and exits cleanly — until the
one-time AWS setup below is done, so it never blocks CI for anyone who
hasn't configured deployment.

**One-time setup, out of band (not something CI can do for you):**

1. Create two ECR repositories: `llm-eval-harness/api`, `llm-eval-harness/frontend`.
2. Create two App Runner services (source: the ECR repos above, port 8000
   for the api service and port 80 for the frontend service). Give the api
   service's App Runner instance role network access to wherever Ollama is
   reachable from (App Runner has no built-in host-network escape hatch —
   plan on a small always-on Ollama host, e.g. an EC2 box on the same VPC,
   rather than `host.docker.internal`, which only works in local Docker).
3. Add a GitHub OIDC identity provider to the AWS account (one-time, if not
   already present) and an IAM role that trusts
   `repo:adityashah841/llm-eval-harness:ref:refs/heads/main`, scoped to
   `ecr:*` on the two repos above and `apprunner:StartDeployment` on the two
   services.
4. Add repo secrets: `AWS_DEPLOY_ROLE_ARN` (the role from step 3),
   `AWS_APPRUNNER_API_SERVICE_ARN`, `AWS_APPRUNNER_FRONTEND_SERVICE_ARN`,
   and `API_PUBLIC_URL` (the api service's App Runner URL, baked into the
   frontend build as `VITE_API_BASE_URL`).

## Key design decisions

- **Adapter pattern** — adding a new model is one new file; the pipeline never changes
- **Async-first** — `asyncio.gather()` parallelises all model calls per sample
- **Fully local** — Ollama serves every model; DeBERTa runs on CPU; no internet needed
  after the initial model downloads
- **Reproducible** — MLflow versions every run with params, metrics, and model list

## Built by

Aditya Shah — [LinkedIn](https://linkedin.com/in/aditya-r-shah26)
