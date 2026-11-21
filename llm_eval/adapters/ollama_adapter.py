import os
import re
from typing import Optional

import httpx
from .base import BaseAdapter, ModelResponse

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "600"))

# Fixed decoding settings so repeated runs of the same model on the same
# prompts are comparable. temperature=0 makes decoding greedy; the seed pins
# any remaining sampling. num_predict caps runaway generations.
DEFAULT_OPTIONS = {"temperature": 0.0, "seed": 42, "num_predict": 1024}

# Reasoning models (e.g. deepseek-r1) may emit their chain of thought inline
# as <think>...</think>. That text is not the answer, so it is removed before
# scoring. (Newer Ollama versions return it in message.thinking instead,
# which is ignored for the same reason.)
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", flags=re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _ns_to_ms(value) -> Optional[float]:
    return round(value / 1e6, 2) if isinstance(value, (int, float)) else None


class OllamaError(RuntimeError):
    """Ollama answered with an error (e.g. model not pulled)."""


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.json().get("error") or response.text
    except ValueError:
        detail = response.text
    raise OllamaError(f"Ollama returned HTTP {response.status_code}: {str(detail).strip()[:300]}")


class OllamaAdapter(BaseAdapter):
    """Adapter for any model served locally by Ollama. Always free."""

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_BASE_URL,
        options: Optional[dict] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        super().__init__(model_name)
        self.base_url = base_url
        self.options = {**DEFAULT_OPTIONS, **(options or {})}
        self.timeout_s = timeout_s

    async def ask(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        start = self._start_timer()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": self.options,
                },
            )
            _raise_for_status(response)
            data = response.json()

        prompt_eval = _ns_to_ms(data.get("prompt_eval_duration"))
        evaluation = _ns_to_ms(data.get("eval_duration"))
        generation = (
            round((prompt_eval or 0.0) + (evaluation or 0.0), 2)
            if prompt_eval is not None or evaluation is not None
            else None
        )

        return ModelResponse(
            model_name=self.model_name,
            response_text=strip_thinking(data["message"]["content"]),
            latency_ms=self._elapsed_ms(start),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            estimated_cost_usd=0.0,
            load_ms=_ns_to_ms(data.get("load_duration")),
            generation_ms=generation,
            raw=data,
        )

    async def model_digest(self) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return None
        wanted = {self.model_name, f"{self.model_name}:latest"}
        for m in models:
            if m.get("name") in wanted or m.get("model") in wanted:
                return m.get("digest")
        return None
