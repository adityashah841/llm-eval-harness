from typing import List

import httpx
from fastapi import APIRouter

from ..schemas import ModelInfo

router = APIRouter(tags=["models"])

OLLAMA_BASE_URL = "http://localhost:11434"

# Mirrors llm_eval/dashboard/app.py::AVAILABLE_MODELS — used as a fallback
# when Ollama isn't reachable so the endpoint stays useful offline.
KNOWN_MODELS = ["llama3.1:70b", "deepseek-r1:14b", "phi4", "gemma3:4b"]


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List models available for evaluation, live from Ollama when reachable."""
    pulled = set()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
        pulled = {m["name"] for m in data.get("models", [])}
    except (httpx.HTTPError, KeyError, ValueError):
        pass

    if not pulled:
        return [ModelInfo(name=m, available=False) for m in KNOWN_MODELS]

    names = pulled | set(KNOWN_MODELS)
    return [ModelInfo(name=n, available=n in pulled) for n in sorted(names)]
