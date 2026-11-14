"""Hosted and local LLM providers, implemented over plain HTTPS (no vendor SDKs).

Keeping every provider on the same tiny HTTP layer makes error handling,
timeouts and retries behave identically across vendors.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import LLMProvider, LLMResponse, ProviderError


def _require_env(var: str, provider: str) -> str:
    value = os.getenv(var)
    if not value:
        raise ProviderError(f"{provider}: environment variable {var} is not set", retryable=False)
    return value


class GeminiProvider(LLMProvider):
    """Google Gemini via the Generative Language REST API (free tier friendly)."""

    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str = "gemini-2.5-flash", *, api_key: str | None = None,
                 client: httpx.Client | None = None, timeout_s: float = 30.0):
        super().__init__(model, client=client, timeout_s=timeout_s)
        self._api_key = api_key

    def _generate(self, prompt: str, *, system: str | None, temperature: float, max_tokens: int) -> LLMResponse:
        api_key = self._api_key or _require_env("GEMINI_API_KEY", self.name)
        config: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if "flash" in self.model and "2.5" in self.model:
            # Disable "thinking" so the token budget goes to the answer (cheaper, faster, deterministic).
            config["thinkingConfig"] = {"thinkingBudget": 0}
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        data = self._post(
            f"{self.base_url}/models/{self.model}:generateContent",
            json=body,
            headers={"x-goog-api-key": api_key},
        )
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderError(f"gemini returned no answer ({reason})", retryable=False)
        parts = (candidates[0].get("content") or {}).get("parts") or []
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text="".join(p.get("text", "") for p in parts).strip(),
            provider=self.name,
            model=self.model,
            input_tokens=int(usage.get("promptTokenCount", 0)),
            output_tokens=int(usage.get("candidatesTokenCount", 0)),
            raw=data,
        )


class OpenAIProvider(LLMProvider):
    """OpenAI (or any OpenAI-compatible endpoint, e.g. Azure OpenAI proxy, vLLM)."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None,
                 base_url: str | None = None, client: httpx.Client | None = None, timeout_s: float = 30.0):
        super().__init__(model, client=client, timeout_s=timeout_s)
        self._api_key = api_key
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    def _generate(self, prompt: str, *, system: str | None, temperature: float, max_tokens: int) -> LLMResponse:
        api_key = self._api_key or _require_env("OPENAI_API_KEY", self.name)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = self._post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return LLMResponse(
            text=((choice.get("message") or {}).get("content") or "").strip(),
            provider=self.name,
            model=self.model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            raw=data,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic Claude via the Messages API."""

    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"

    def __init__(self, model: str = "claude-haiku-4-5", *, api_key: str | None = None,
                 client: httpx.Client | None = None, timeout_s: float = 30.0):
        super().__init__(model, client=client, timeout_s=timeout_s)
        self._api_key = api_key

    def _generate(self, prompt: str, *, system: str | None, temperature: float, max_tokens: int) -> LLMResponse:
        api_key = self._api_key or _require_env("ANTHROPIC_API_KEY", self.name)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        data = self._post(
            f"{self.base_url}/messages",
            json=body,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )
        usage = data.get("usage") or {}
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return LLMResponse(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            raw=data,
        )


class OllamaProvider(LLMProvider):
    """Local open-weight models (Llama, Mistral, Qwen...) served by Ollama."""

    name = "ollama"

    def __init__(self, model: str = "llama3.2", *, base_url: str | None = None,
                 client: httpx.Client | None = None, timeout_s: float = 120.0):
        super().__init__(model, client=client, timeout_s=timeout_s)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")

    def _generate(self, prompt: str, *, system: str | None, temperature: float, max_tokens: int) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = self._post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        return LLMResponse(
            text=((data.get("message") or {}).get("content") or "").strip(),
            provider=self.name,
            model=self.model,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            raw=data,
        )
