"""Runtime settings, all read from environment variables.

LLM_EVAL_CORS_ORIGINS      Comma-separated origins allowed to call the API from
                           a browser. Default: the local Vite dev/preview and
                           docker-compose frontend origins.
LLM_EVAL_API_KEY           If set, POST /runs requires the header
                           `X-API-Key: <value>`. Read endpoints stay open.
LLM_EVAL_MAX_CONCURRENT_RUNS
                           Runs executing at once (default 1). Extra runs wait
                           in `pending`. One Ollama server serializes requests
                           anyway, so more mainly adds memory pressure.
LLM_EVAL_DATASETS_DIR      Directory holding the dataset folders (default
                           <repo>/datasets). Only folders inside it can be run.
"""
import os
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://localhost:4173,http://localhost:3000"


def cors_origins() -> List[str]:
    raw = os.environ.get("LLM_EVAL_CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


def api_key() -> Optional[str]:
    return os.environ.get("LLM_EVAL_API_KEY") or None


def max_concurrent_runs() -> int:
    try:
        return max(1, int(os.environ.get("LLM_EVAL_MAX_CONCURRENT_RUNS", "1")))
    except ValueError:
        return 1


def datasets_dir() -> Path:
    return Path(os.environ.get("LLM_EVAL_DATASETS_DIR", REPO_ROOT / "datasets")).resolve()
