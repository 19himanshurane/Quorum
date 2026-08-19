"""Builds `instructor`-patched clients for the live LLM providers.

Groq, Mistral, and NVIDIA NIM (and OpenAI and Ollama, for anyone who wants them)
all expose an OpenAI-compatible `/chat/completions` endpoint, so a single client
construction path covers all of them - only the base URL and API key differ.
Only imported/used when a critic's provider is not "mock", keeping the mock demo
path free of any SDK/network dependency.
"""
from __future__ import annotations

import os
import threading

import instructor

# (base_url, api_key env var name). `api_key_env=None` means no key is required
# (Ollama). Ollama's base_url is resolved from Settings instead, since it's
# typically a local/self-hosted address rather than a fixed public one.
_PROVIDER_ENDPOINTS: dict[str, tuple[str | None, str | None]] = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),  # None -> openai SDK's own default base_url
    "ollama": (None, None),
}

# Constructing an OpenAI/httpx client is not just cheap object setup - it lazily
# initializes an SSL context, which touches the OS certificate store. Doing that
# from multiple threads at once (exactly what LangGraph's parallel critic
# dispatch does on first run) can deadlock on Windows. Caching + a lock means
# only one thread ever constructs a given provider's client, and every other
# caller (concurrent or not) just reuses it - httpx clients are safe to share
# and use concurrently for requests once built.
_client_cache: dict[tuple[str, str | None], object] = {}
_client_cache_lock = threading.Lock()


def _require_env(name: str, provider: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (required for provider={provider!r} in live mode)")
    return value


def instructor_client_for(config, settings):
    if config.provider not in _PROVIDER_ENDPOINTS:
        raise ValueError(f"No live client available for provider={config.provider!r}")

    cache_key = (config.provider, settings.ollama_base_url if config.provider == "ollama" else None)

    with _client_cache_lock:
        cached = _client_cache.get(cache_key)
        if cached is not None:
            return cached

        from openai import OpenAI

        base_url, api_key_env = _PROVIDER_ENDPOINTS[config.provider]
        if config.provider == "ollama":
            base_url = settings.ollama_base_url
            api_key = "ollama"  # Ollama ignores the key but the SDK requires a non-empty string
        else:
            api_key = _require_env(api_key_env, config.provider)

        # Explicit timeout: the openai SDK's own default is ~10 minutes, which
        # would otherwise let one unresponsive provider hang each retry attempt
        # that long.
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=settings.request_timeout_seconds)
        # JSON mode (prompted structured output) rather than native tool-calling:
        # it's the one extraction strategy that works consistently across every
        # provider/model in this registry, regardless of each one's function-calling support.
        instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)
        _client_cache[cache_key] = instructor_client
        return instructor_client


def structured_completion(client, config, response_model, system_prompt: str, user_prompt: str):
    return client.chat.completions.create(
        model=config.model,
        response_model=response_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
