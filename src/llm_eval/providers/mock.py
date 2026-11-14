"""Deterministic mock provider with fault injection.

Lets the whole framework (and CI) run with zero API keys and zero cost, and
lets you rehearse failure modes (timeouts, 429s, outages) on purpose.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .base import LLMProvider, LLMResponse, ProviderError


@dataclass
class MockRule:
    match: str
    response: str


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(
        self,
        model: str = "mock-1",
        *,
        rules: list[MockRule] | None = None,
        responses_file: str | Path | None = None,
        default_response: str = "I'm sorry, I don't have that information. Please contact support.",
        failure_rate: float = 0.0,
        fail_status: int = 503,
        latency_ms: float = 0.0,
        always_fail: bool = False,
        seed: int = 42,
        **_: object,
    ):
        super().__init__(model)
        self.rules = list(rules or [])
        if responses_file:
            self.rules.extend(self.load_rules(responses_file))
        self.default_response = default_response
        self.failure_rate = failure_rate
        self.fail_status = fail_status
        self.latency_ms = latency_ms
        self.always_fail = always_fail
        self._rng = random.Random(seed)
        self.calls = 0

    @staticmethod
    def load_rules(path: str | Path) -> list[MockRule]:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return [MockRule(match=r["match"].lower(), response=r["response"].strip()) for r in data.get("rules", [])]

    def _generate(self, prompt: str, *, system: str | None, temperature: float, max_tokens: int) -> LLMResponse:
        self.calls += 1
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        if self.always_fail or (self.failure_rate and self._rng.random() < self.failure_rate):
            raise ProviderError(
                f"{self.id} injected failure (HTTP {self.fail_status})",
                retryable=self.fail_status in {429, 500, 502, 503, 504},
                status_code=self.fail_status,
            )

        lowered = prompt.lower()
        text = next((r.response for r in self.rules if r.match in lowered), self.default_response)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=max(1, len(((system or "") + prompt).split())),
            output_tokens=max(1, len(text.split())),
        )
