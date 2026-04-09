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
    raw: Optional[dict] = None


class BaseAdapter(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def ask(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        pass

    def _start_timer(self) -> float:
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)
