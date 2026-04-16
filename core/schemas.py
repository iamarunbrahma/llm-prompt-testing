from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o4-mini", "o3-mini"],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-4-5-20251001",
    ],
    "google": ["gemini/gemini-2.5-pro", "gemini/gemini-2.0-flash"],
    "ollama": ["ollama/llama3", "ollama/mistral", "ollama/codellama"],
}

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = DEFAULT_PROVIDER
    model_name: str = DEFAULT_MODEL
    api_key: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_base: Optional[str] = None


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    cached: bool = False


class EvalResult(BaseModel):
    metric_name: str
    score: Union[float, str, dict]
    details: Optional[str] = None


class RubricCriterion(BaseModel):
    name: str
    description: str
    scale_min: int = 1
    scale_max: int = 5


class ComparisonResult(BaseModel):
    winner: str  # "A", "B", or "tie"
    reasoning: str
    scores: dict[str, float] = Field(default_factory=dict)
