import httpx
from .base import BaseAdapter, ModelResponse


class OllamaAdapter(BaseAdapter):
    """Adapter for any model served locally by Ollama. Always free."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url

    async def ask(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        start = self._start_timer()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

        return ModelResponse(
            model_name=self.model_name,
            response_text=data["message"]["content"],
            latency_ms=self._elapsed_ms(start),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            estimated_cost_usd=0.0,
            raw=data,
        )
