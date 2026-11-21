from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm_eval.runner.prompt_runner import EVAL_VERSION

from .config import cors_origins
from .db import init_db, mark_interrupted_runs
from .routes import datasets, metrics, models, runs


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    mark_interrupted_runs()
    yield


app = FastAPI(
    title="LLM Eval Harness API",
    description="REST layer over the local LLM evaluation engine (llm_eval/).",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.include_router(runs.router)
app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(metrics.router)


@app.get("/health")
async def health():
    return {"status": "ok", "eval_version": EVAL_VERSION}
