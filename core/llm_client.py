from __future__ import annotations

import os
import time

import litellm
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential

from core.cache import cache_key, get_cached, set_cached
from core.schemas import LLMConfig, LLMResponse

litellm.drop_params = True


def _set_api_key(config: LLMConfig) -> None:
    if config.provider == "openai":
        os.environ["OPENAI_API_KEY"] = config.api_key
    elif config.provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = config.api_key
    elif config.provider == "google":
        os.environ["GEMINI_API_KEY"] = config.api_key


def _build_params(config: LLMConfig) -> dict:
    params: dict = {
        "model": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
    }
    if config.frequency_penalty != 0.0:
        params["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty != 0.0:
        params["presence_penalty"] = config.presence_penalty
    if config.api_base:
        params["api_base"] = config.api_base
    return params


@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(4))
def get_completion(
    config: LLMConfig,
    system_prompt: str,
    user_message: str,
    use_cache: bool = True,
) -> LLMResponse:
    if use_cache:
        key = cache_key(config, system_prompt, user_message)
        cached = get_cached(key)
        if cached is not None:
            return cached

    _set_api_key(config)
    params = _build_params(config)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    start = time.perf_counter()
    response = litellm.completion(messages=messages, **params)
    elapsed_ms = (time.perf_counter() - start) * 1000

    content = response.choices[0].message.content or ""
    usage = response.usage or litellm.Usage()
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = 0.0

    result = LLMResponse(
        content=content.strip(),
        model=response.model or config.model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round(elapsed_ms, 1),
        estimated_cost_usd=round(cost, 6),
    )

    if use_cache:
        set_cached(key, result)

    return result


EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "anthropic": "text-embedding-3-small",  # Anthropic has no embeddings; use OpenAI
    "google": "gemini/text-embedding-004",
    "ollama": "ollama/nomic-embed-text",
}


@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(4))
def get_embedding(
    text: str,
    config: LLMConfig,
    model: str | None = None,
) -> list[float]:
    if model is None:
        model = EMBEDDING_MODELS.get(config.provider, "text-embedding-3-small")
    _set_api_key(config)
    # For providers without native embeddings (Anthropic), ensure
    # the OpenAI key is set since we fall back to OpenAI embeddings
    if config.provider == "anthropic" and model.startswith("text-embedding"):
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not openai_key:
            os.environ["OPENAI_API_KEY"] = config.api_key
    response = litellm.embedding(model=model, input=[text])
    return response.data[0]["embedding"]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.asarray(vec_a)
    b = np.asarray(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def validate_api_key(
    provider: str, api_key: str, model: str
) -> tuple[bool, str]:
    try:
        config = LLMConfig(provider=provider, model_name=model, api_key=api_key)
        get_completion(
            config,
            system_prompt="Say OK",
            user_message="Test",
            use_cache=False,
        )
        return True, ""
    except Exception as e:
        return False, str(e)
