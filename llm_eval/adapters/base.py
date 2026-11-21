from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class ModelResponse:
    model_name: str
    response_text: str
    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    estimated_cost_usd: float = 0.0
    # Server-reported timings, when the backend provides them. `load_ms` is
    # the time spent loading the model into memory (a cold start), and
    # `generation_ms` is prompt processing + token generation only, so it
    # excludes queueing and loading.
    load_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    raw: Optional[dict] = None


class BaseAdapter(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def ask(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        pass

    async def model_digest(self) -> Optional[str]:
        """Content hash identifying the exact model weights, if the backend
        exposes one. Recorded with every run so a score change can be traced
        to a model change."""
        return None

    def _start_timer(self) -> float:
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)
