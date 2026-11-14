"""Production resilience for LLM calls.

* Retries with exponential backoff + full jitter (only for retryable errors)
* Per-provider circuit breaker (closed -> open -> half-open)
* Ordered model fallback chain (e.g. Gemini -> Claude -> local Llama)
* Client-side rate limiting (keeps free tiers under their RPM quota)
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .providers.base import LLMProvider, LLMResponse, ProviderError
from .tracing import Tracer

Clock = Callable[[], float]
Sleeper = Callable[[float], None]


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: bool = True

    def delay(self, attempt: int, rng: random.Random) -> float:
        """Backoff before retry number ``attempt`` (1-based)."""
        capped = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        return rng.uniform(0, capped) if self.jitter else capped


class CircuitBreaker:
    """Stops hammering a provider that keeps failing, then probes it again later."""

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int = 3, recovery_timeout_s: float = 30.0, clock: Clock = time.monotonic):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_unlocked()

    def _state_unlocked(self) -> str:
        if self._opened_at is None:
            return self.CLOSED
        if self._clock() - self._opened_at >= self.recovery_timeout_s:
            return self.HALF_OPEN
        return self.OPEN

    def allow_request(self) -> bool:
        return self.state != self.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            if self._state_unlocked() == self.HALF_OPEN:
                self._opened_at = self._clock()  # probe failed: open again
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = self._clock()


class RateLimiter:
    """Sliding-window limiter: at most ``rpm`` calls in any 60-second window."""

    def __init__(self, rpm: int | None, clock: Clock = time.monotonic, sleep: Sleeper = time.sleep):
        self.rpm = rpm
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if not self.rpm:
            return
        with self._lock:
            now = self._clock()
            while self._calls and now - self._calls[0] >= 60:
                self._calls.popleft()
            if len(self._calls) >= self.rpm:
                self._sleep(60 - (now - self._calls[0]) + 0.05)
                self._calls.popleft()
            self._calls.append(self._clock())


class AllProvidersFailed(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("All providers in the fallback chain failed: " + " | ".join(errors))
        self.errors = errors


@dataclass
class CallResult:
    response: LLMResponse
    attempts: int
    fallback_used: bool
    errors: list[str] = field(default_factory=list)


class ReliableLLM:
    """Wraps an ordered list of providers with retries, breakers, fallback and rate limits."""

    def __init__(
        self,
        providers: list[LLMProvider],
        *,
        retry: RetryPolicy | None = None,
        breaker_factory: Callable[[], CircuitBreaker] | None = None,
        rate_limit_rpm: int | None = None,
        tracer: Tracer | None = None,
        sleep: Sleeper = time.sleep,
        seed: int = 7,
    ):
        if not providers:
            raise ValueError("ReliableLLM needs at least one provider")
        self.providers = providers
        self.retry = retry or RetryPolicy()
        factory = breaker_factory or CircuitBreaker
        self.breakers = {p.id: factory() for p in providers}
        self.limiter = RateLimiter(rate_limit_rpm, sleep=sleep)
        self.tracer = tracer or Tracer()
        self._sleep = sleep
        self._rng = random.Random(seed)

    def generate(self, prompt: str, **kwargs) -> CallResult:
        errors: list[str] = []
        attempts = 0
        for index, provider in enumerate(self.providers):
            breaker = self.breakers[provider.id]
            if not breaker.allow_request():
                errors.append(f"{provider.id}: circuit open, skipped")
                continue

            for attempt in range(1, self.retry.max_attempts + 1):
                attempts += 1
                self.limiter.acquire()
                with self.tracer.span("llm.call", provider=provider.id, attempt=attempt) as span:
                    try:
                        response = provider.generate(prompt, **kwargs)
                    except ProviderError as exc:
                        breaker.record_failure()
                        span.fail(str(exc))
                        errors.append(f"{provider.id} attempt {attempt}: {exc}")
                        if exc.retryable and attempt < self.retry.max_attempts and breaker.allow_request():
                            self._sleep(self.retry.delay(attempt, self._rng))
                            continue
                        break  # move to next provider in the chain
                    breaker.record_success()
                    span.set(
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        latency_ms=response.latency_ms,
                    )
                    return CallResult(response=response, attempts=attempts, fallback_used=index > 0, errors=errors)
        raise AllProvidersFailed(errors)
