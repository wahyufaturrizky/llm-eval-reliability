"""Provider abstraction shared by every LLM backend."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResponse:
    """Normalised response returned by every provider."""

    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class ProviderError(Exception):
    """Raised when a provider call fails.

    ``retryable`` tells the reliability layer whether a retry can help
    (rate limits, timeouts, 5xx) or not (bad request, auth error).
    """

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class LLMProvider(ABC):
    """Base class for an LLM backend."""

    name: str = "base"

    def __init__(self, model: str, *, client: httpx.Client | None = None, timeout_s: float = 30.0):
        self.model = model
        self.timeout_s = timeout_s
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def id(self) -> str:
        return f"{self.name}:{self.model}"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.perf_counter()
        response = self._generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        response.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return response

    @abstractmethod
    def _generate(
        self, prompt: str, *, system: str | None, temperature: float, max_tokens: int
    ) -> LLMResponse:
        ...

    # ------------------------------------------------------------------ helpers
    def _post(self, url: str, *, json: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        """POST JSON and map transport/HTTP failures to :class:`ProviderError`."""
        try:
            resp = self._client.post(url, json=json, headers=headers, timeout=self.timeout_s)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.id} timed out: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"{self.id} transport error: {exc}", retryable=True) from exc

        if resp.status_code >= 400:
            body = resp.text[:300]
            raise ProviderError(
                f"{self.id} HTTP {resp.status_code}: {body}",
                retryable=resp.status_code in RETRYABLE_STATUS,
                status_code=resp.status_code,
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self.id} returned non-JSON body", retryable=False) from exc
