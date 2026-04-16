from __future__ import annotations

import hashlib
import json
from typing import Optional

import streamlit as st

from core.schemas import LLMConfig, LLMResponse

CACHE_KEY = "response_cache"


def _ensure_cache() -> dict[str, LLMResponse]:
    if CACHE_KEY not in st.session_state:
        st.session_state[CACHE_KEY] = {}
    return st.session_state[CACHE_KEY]


def cache_key(config: LLMConfig, system_prompt: str, user_message: str) -> str:
    payload = json.dumps(
        {
            "model": config.model_name,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "system_prompt": system_prompt,
            "user_message": user_message,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def get_cached(key: str) -> Optional[LLMResponse]:
    cache = _ensure_cache()
    return cache.get(key)


def set_cached(key: str, response: LLMResponse) -> None:
    cache = _ensure_cache()
    cache[key] = response
