# LLM Evaluation Harness

A modular, open-source framework for evaluating and comparing local open-source LLMs
on answer accuracy, grounding (hallucination checks) and latency. It started as a
Streamlit + MLflow tool for local experimentation and now also has a REST API, a
React frontend, a persistent metrics store and a daily benchmark, all built on the
same evaluation engine.

**No API keys and no paid services.** Models are served locally by Ollama; the NLI
model is downloaded once from Hugging Face and runs on CPU.

![CI](https://github.com/adityashah841/llm-eval-harness/actions/workflows/eval.yml/badge.svg)

## Models

Any model Ollama can serve works. The dashboard and API list these four by default:

| Model | Ollama tag | Download size | Notes |
|---|---|---|---|
| Gemma 3 4B | `gemma3:4b` | 3.4 GB | Fast baseline; used by CI and the daily benchmark |
| Phi-4 14B | `phi4` | 9.1 GB | |
| DeepSeek-R1 14B | `deepseek-r1:14b` | 9.0 GB | Reasoning model; its `<think>` text is stripped before scoring |
| Llama 3.1 70B | `llama3.1:70b` | 43 GB | Needs a machine with well over 43 GB of free RAM/VRAM |

Download sizes are from the [Ollama library](https://ollama.com/library). As a rule of
thumb a model needs at least its download size in free memory, plus room for context.

## How answers are scored

Each dataset is scored in the way that fits its task. Scores from a run are stored with
the methodology version (`EVAL_VERSION`, currently **2**) so runs scored differently are
never mixed in trends.

| Dataset | What the model is asked | Accuracy metric | Grounding check (NLI premise) |
|---|---|---|---|
| `legal_qa` (5 questions) | Answer in one or two sentences | ROUGE-1/2/L vs. a reference answer | The reference answer |
| `summarization` (4 passages) | One-sentence summary | ROUGE-1/2/L vs. a reference summary | The **source passage** |
| `code_gen` (4 tasks) | Return only the function | **Unit tests**: the code is run against asserts; pass rate | — (not applicable) |

**Why short answers?** ROUGE measures word overlap with the reference. A correct but
long answer scores low simply because it is long: in the original daily benchmark,
gemma3:4b's answers averaged ~850 tokens and ROUGE-L 0.064 against one-sentence
references, even when the answer was right. The
datasets' system prompts now ask for answers about as long as the references, and
`tests/test_engine.py::test_rouge_penalizes_verbosity` pins the effect.

**Grounding (NLI) check.** The response is split into sentences. A local NLI model
([`cross-encoder/nli-deberta-v3-small`](https://huggingface.co/cross-encoder/nli-deberta-v3-small))
labels each sentence as *entailed by*, *contradicted by* or *neutral to* the premise.
Reported per response:

- **Contradiction flag** (`hallucination_flag`): at least one sentence contradicts the
  premise. This is the strict signal; "hallucination rate" in the UI is the share of
  responses with this flag.
- **Supported sentences** (`entailment_rate`): share of sentences the premise entails.
  For summarization, unsupported sentences are content not in the source. For QA a
  one-sentence reference can't support every true extra detail, so read it with care.

What it can't do: catch a false claim the premise says nothing about (that is
"neutral", not "contradiction"), or judge correctness beyond what the premise states.
`tests/test_nli_model.py` checks the real model on faithful, contradicting and
unsupported examples; CI runs it.

**Code.** The first fenced code block in the response is run with the sample's asserts
in a separate Python process (isolated mode, empty temp dir, stripped environment, 10 s
timeout, CPU and 1 GiB memory limits on Linux/macOS). That keeps accidents contained
but is **not a security sandbox**. Set `LLM_EVAL_DISABLE_CODE_EXEC=1` to turn
execution off, e.g. on a shared server. Every reference solution is checked against its
own tests in `tests/test_datasets.py`.

**Reproducibility.** Requests use greedy decoding (`temperature: 0`, `seed: 42`,
`num_predict: 1024`), and each result records the model's Ollama digest, so a score
change can be traced to a model change. Outputs can still differ across Ollama
versions or hardware.

**Latency.** `latency_ms` is the wall-clock time of the request, including model loading
on a cold start (recorded separately as `load_ms`). `generation_ms` is Ollama's own
prompt-processing + generation time. Models are queried **one at a time** by default.
`--parallel` sends all models' requests for a sample at once. That is faster when the
server can keep every model loaded, but one Ollama server queues requests, so wall-clock
latency then includes waiting time.

**Failures are recorded, not dropped.** A model call that errors (e.g. model not
pulled, timeout) is saved with its error. A run is `completed` (all calls scored),
`partial` (some failed) or `failed` (none succeeded), and the CLI exits non-zero.

**Scale.** 13 samples is a smoke-scale benchmark: good for regressions and comparing
setups, too small for fine-grained model rankings. Add samples with your own YAML or
`scripts/generate_tests.py`.

## Quick start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com), running (`ollama serve`)
- [uv](https://github.com/astral-sh/uv): `pip install uv`

```bash
git clone https://github.com/adityashah841/llm-eval-harness
cd llm-eval-harness
uv sync
ollama pull gemma3:4b
```

The first scored run downloads the NLI model from Hugging Face; it is cached afterwards.

### Run a benchmark from the CLI

```bash
# One model, one dataset
uv run python scripts/run_eval.py --dataset datasets/legal_qa --models gemma3:4b

# Several models (repeat -m), plus a JSON summary
uv run python scripts/run_eval.py -d datasets/code_gen -m gemma3:4b -m phi4 \
  --summary-json outputs/code_gen.json

# First 2 samples only, no MLflow
uv run python scripts/run_eval.py -d datasets/summarization -m gemma3:4b --smoke-test --no-mlflow
```

Per-sample results go to `outputs/results.csv` (change with `--output`).

### Launch the Streamlit dashboard

```bash
uv run streamlit run llm_eval/dashboard/app.py   # http://localhost:8501
```

### View experiment history (MLflow)

Runs log to `mlflow.db` in the directory you run from (MLflow's default):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

### Generate more test cases

```bash
uv run python scripts/generate_tests.py --domain legal_qa --count 10                    # uses llama3.1:70b
uv run python scripts/generate_tests.py --domain legal_qa --count 5 --model gemma3:4b   # smaller model
```

Cases are appended to `datasets/<domain>/<domain>_generated.yaml`. Malformed and
duplicate-id cases are dropped. Generated reference answers come from a model:
**review them** before trusting scores computed against them.

### Run with Docker

```bash
# Ollama must be running on the host, with the models pulled
docker compose up --build
```

This starts the Streamlit `dashboard` (8501), `mlflow` UI (5000), FastAPI `api` (8000)
and `frontend` (5173, served by nginx). `api` and `dashboard` reach the host's Ollama
through `host.docker.internal`. Run history is kept in `./data` and MLflow data in `./mlruns`.

## API

The FastAPI service exposes the same engine over REST.

```bash
uv run uvicorn api.main:app --reload --port 8000   # docs at http://localhost:8000/docs
```

| Method | Path | Description |
|---|---|---|
| `POST` | `/runs` | Start an eval run (dataset + models) in the background |
| `GET` | `/runs` | Recent runs, newest first |
| `GET` | `/runs/{id}` | Run status (`pending` / `running` / `completed` / `partial` / `failed`), failures, model digests |
| `GET` | `/runs/{id}/results` | Per-sample scores once the run has finished |
| `GET` | `/datasets` | Datasets that can be run, with sample counts and scoring type |
| `GET` | `/models` | Models available for evaluation, live from Ollama when reachable |
| `GET` | `/metrics/timeseries` | Per-`(run, model)` aggregates; filter by `model`, `dataset`, `eval_version` |

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"dataset": "legal_qa", "models": ["gemma3:4b"], "smoke_test": true}'
```

Behaviour worth knowing:

- Only folders inside `datasets/` can be run. Other paths are rejected with 404.
- Runs execute one at a time by default; extra runs wait as `pending`.
- Scoring runs off the event loop, so the API stays responsive during a run.
- Runs are saved to SQLite when they start and when they finish. A run left
  `running` by a crashed or restarted process is marked `failed` on the next startup.

Configuration (environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is |
| `OLLAMA_TIMEOUT_S` | `600` | Per-request timeout |
| `LLM_EVAL_DB_PATH` | `data/metrics.db` | SQLite run history |
| `LLM_EVAL_CORS_ORIGINS` | `http://localhost:5173,http://localhost:4173,http://localhost:3000` | Comma-separated browser origins allowed to call the API |
| `LLM_EVAL_API_KEY` | unset | If set, `POST /runs` requires `X-API-Key: <value>` (reads stay open) |
| `LLM_EVAL_MAX_CONCURRENT_RUNS` | `1` | Runs executing at once |
| `LLM_EVAL_DISABLE_CODE_EXEC` | unset | `1` skips running generated code |

The frontend never sends an API key. With `LLM_EVAL_API_KEY` set, trigger runs with
`curl`/scripts and use the UI to browse results.

## Frontend (Vite + React + TypeScript)

```bash
cd frontend
npm install
cp .env.example .env   # point VITE_API_BASE_URL at your API, if not localhost:8000
npm run dev            # http://localhost:5173
```

Pages:

- **Run configuration**: pick a dataset (from `GET /datasets`) and models (from
  `GET /models`), start a run and follow its status, including any failed model calls.
- **Results**: per-sample table for any past run, filterable by model and by
  outcome (contradiction / failed tests).
- **Model comparison**: side-by-side aggregates for a multi-model run (ROUGE and
  grounding for text, pass rate for code).
- **System health**: trends from `GET /metrics/timeseries` for one methodology
  version, model and dataset (see [Observability](#observability-dashboard)).

```bash
npm run build   # tsc -b && vite build
npm run lint    # oxlint
```

## Persistent metrics and the daily benchmark

Every API-triggered run, and every scheduled benchmark run, is stored in SQLite
(`data/metrics.db` locally, not committed). `/metrics/timeseries` reads from it.
Older databases are migrated in place on first use: their runs keep their scores and
are tagged `eval_version` 1.

`.github/workflows/scheduled-benchmark.yml` runs daily at 06:00 UTC (and on demand). It
evaluates `gemma3:4b` on all three datasets with `scripts/scheduled_benchmark.py` and
commits the updated database to the **`benchmark-data` branch**, not to `main`.

```bash
# Load the benchmark history locally
git fetch origin benchmark-data
mkdir -p data && git show origin/benchmark-data:metrics.db > data/metrics.db

# Run the same benchmark yourself
uv run python scripts/scheduled_benchmark.py
```

Scores are deterministic for a given model digest and Ollama version, so a score change
means something changed. Latency on GitHub's shared runners varies from machine to
machine, so treat the benchmark's latency trend as indicative only.

## Observability dashboard

The frontend's **System health** page (`/health`) shows one *methodology version × model
× dataset* slice at a time:

- **Accuracy trend and drift flag**: ROUGE-L (text) or unit-test pass rate (code) per
  run. A `drift detected` banner appears if the latest run falls more than 15% below the
  mean of the previous 5 runs. The banner also says when the model digest changed within
  that window, which usually explains the drift.
- **Grounding trend**: share of responses with a contradicted sentence, and share of
  supported sentences.
- **Latency percentiles**: p50 and p95 per run.
- A raw data table with every value and the model digest.

## Tests

```bash
uv run pytest                      # offline: Ollama and the NLI model are stubbed
uv run pytest -m requires_models   # real NLI model (downloads it from Hugging Face)
```

CI (`.github/workflows/eval.yml`) runs on every push and PR to `main`:

- `backend-tests`: the offline suite.
- `model-eval`: the real-NLI tests, then installs Ollama, pulls `gemma3:4b` and evaluates
  every dataset in full. Results are in the run's job summary and an `eval-results`
  artifact, and the job fails if any model call fails.
- `frontend-checks`: lint and build.
- `docker-build`: builds all three images and checks that the API and frontend serve.
- `deploy`: see below.

## Deployment (AWS)

On a push to `main`, after all checks pass, `deploy` builds `api/Dockerfile` and
`frontend/Dockerfile`, pushes both to ECR and redeploys two App Runner services. It
authenticates with GitHub OIDC (no long-lived AWS keys) and skips cleanly until the
setup below is done.

**One-time setup, outside CI:**

1. Create two ECR repositories: `llm-eval-harness/api`, `llm-eval-harness/frontend`.
2. Create two App Runner services from those repositories (port 8000 for the api, 80
   for the frontend). On the api service set:
   - `OLLAMA_BASE_URL` to an Ollama host it can reach (App Runner has no
     `host.docker.internal`; e.g. an EC2 instance in a VPC connected via a VPC connector)
   - `LLM_EVAL_CORS_ORIGINS` to the frontend service's URL
   - `LLM_EVAL_API_KEY` so the public URL can't be used to start runs
   - `LLM_EVAL_DISABLE_CODE_EXEC=1` unless you accept running model-generated code there
3. Add a GitHub OIDC identity provider to the AWS account and an IAM role that trusts
   `repo:adityashah841/llm-eval-harness:ref:refs/heads/main`, scoped to `ecr:*` on the
   two repositories and `apprunner:StartDeployment` on the two services.
4. Add repo secrets: `AWS_DEPLOY_ROLE_ARN`, `AWS_APPRUNNER_API_SERVICE_ARN`,
   `AWS_APPRUNNER_FRONTEND_SERVICE_ARN`, and `API_PUBLIC_URL` (baked into the frontend
   build as `VITE_API_BASE_URL`).

**Limitations:** App Runner's filesystem is temporary. The SQLite run history is lost on
every redeploy or restart, and the NLI model is re-downloaded on cold starts. For
durable history, run the API somewhere with a persistent volume (e.g. ECS with EFS, or
a VM). The api image also includes PyTorch's default Linux wheels (with CUDA libraries),
which makes it several GB.

## Project structure

```
llm_eval/
├── adapters/        # BaseAdapter + OllamaAdapter (deterministic options, digests, think-stripping)
├── evaluators/      # ROUGE, sentence-level NLI grounding, unit-test execution
├── runner/          # Dataset loader, PromptRunner (EvalResult/RunOutcome), shared aggregation
└── dashboard/       # Streamlit app
api/
├── main.py          # FastAPI app (CORS, startup migration/cleanup)
├── config.py        # Environment-variable settings
├── routes/          # runs, datasets, models, metrics
├── schemas.py       # Pydantic request/response models
├── store.py         # In-memory registry of runs in this process
├── db.py            # SQLite persistence + schema migration — the source for /metrics/timeseries
└── Dockerfile
frontend/
├── Dockerfile       # Multi-stage build → nginx-served static bundle
├── nginx.conf       # SPA fallback
└── src/
    ├── api/         # Typed fetch client
    ├── pages/       # Run configuration, Results, Comparison, System health
    ├── components/  # Nav, TrendChart, RunPicker, StatusPill, RunFailures
    ├── hooks/       # useRunData (debounced, abortable run loading)
    └── lib/         # stats.ts (same aggregation as llm_eval/runner/summary.py)
datasets/
├── legal_qa/        # 5 QA pairs
├── summarization/   # 4 passages with source context
└── code_gen/        # 4 tasks with unit tests
scripts/
├── run_eval.py             # CLI
├── generate_tests.py       # Draft new cases with a local model
└── scheduled_benchmark.py  # Daily benchmark → metrics DB
tests/               # Offline suite + requires_models checks
.github/workflows/
├── eval.yml                 # CI: tests, real-model eval, frontend, docker, deploy
└── scheduled-benchmark.yml  # Daily benchmark → benchmark-data branch
Dockerfile            # Streamlit dashboard image
docker-compose.yml    # dashboard + mlflow + api + frontend, wired to host Ollama
```

## Architecture

```
Dataset YAML (prompt, reference, optional source context / unit tests, system prompt)
    ↓
PromptRunner ──► OllamaAdapter (one model at a time by default; --parallel to overlap)
    ↓
Scoring (in a worker thread)
  text: ROUGE vs reference + sentence-level NLI vs source/reference
  code: run against unit tests
    ↓
RunOutcome (results + recorded failures)
    ↓
CSV / JSON summary · MLflow · SQLite (API + daily benchmark) → React + Streamlit UIs
```

## Methodology changes

- **v2 (current)**: short-answer system prompts; greedy decoding with a fixed seed;
  sentence-level NLI with the source passage as the premise for summaries; unit-test
  execution for code; failures recorded; model digest stored.
- **v1**: free-length answers; the first 500 characters of the response checked as one
  block against the reference; ROUGE used for code. In the 129 daily v1 runs that
  produced results, 251 of 258 responses were labelled "neutral" (1 "contradiction"), so the hallucination
  rate was close to 0% by construction rather than by model quality. v1 runs remain in
  the history, tagged `eval_version` 1.

## Built by

Aditya Shah — [LinkedIn](https://linkedin.com/in/aditya-r-shah26)
