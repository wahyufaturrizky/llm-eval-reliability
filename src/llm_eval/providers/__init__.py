"""LLM providers and a small factory to build them from config."""

from __future__ import annotations

from typing import Any

from .base import LLMProvider, LLMResponse, ProviderError
from .hosted import AnthropicProvider, GeminiProvider, OllamaProvider, OpenAIProvider
from .mock import MockProvider

REGISTRY: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "mock": MockProvider,
}


def build_provider(spec: dict[str, Any]) -> LLMProvider:
    """Build a provider from a config dict like ``{"type": "gemini", "model": "gemini-2.5-flash"}``."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in REGISTRY:
        raise ValueError(f"Unknown provider type '{kind}'. Choose from: {', '.join(REGISTRY)}")
    return REGISTRY[kind](**spec)


__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "LLMProvider",
    "LLMResponse",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "build_provider",
]
