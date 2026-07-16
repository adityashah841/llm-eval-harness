from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import metrics, models, runs

app = FastAPI(
    title="LLM Eval Harness API",
    description=(
        "REST layer over the local LLM evaluation engine "
        "(llm_eval/adapters, evaluators, runner) — no changes to that engine."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(models.router)
app.include_router(metrics.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
