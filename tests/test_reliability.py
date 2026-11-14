import pytest

from llm_eval.providers import MockProvider
from llm_eval.providers.base import LLMProvider, LLMResponse, ProviderError
from llm_eval.reliability import AllProvidersFailed, CircuitBreaker, RateLimiter, ReliableLLM, RetryPolicy


class ScriptedProvider(LLMProvider):
    """Fails according to a script, then succeeds."""

    name = "scripted"

    def __init__(self, model: str, script: list[ProviderError | str]):
        super().__init__(model)
        self.script = list(script)
        self.calls = 0

    def _generate(self, prompt, *, system, temperature, max_tokens):
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if isinstance(step, ProviderError):
            raise step
        return LLMResponse(text=step, provider=self.name, model=self.model)


def retryable():
    return ProviderError("503", retryable=True, status_code=503)


def fatal():
    return ProviderError("401", retryable=False, status_code=401)


NO_SLEEP = {"sleep": lambda _s: None}


def test_retries_then_succeeds():
    p = ScriptedProvider("a", [retryable(), retryable(), "hello"])
    llm = ReliableLLM([p], retry=RetryPolicy(max_attempts=3), **NO_SLEEP)
    result = llm.generate("hi")
    assert result.response.text == "hello"
    assert result.attempts == 3
    assert not result.fallback_used


def test_non_retryable_error_skips_retries_and_falls_back():
    primary = ScriptedProvider("a", [fatal()])
    backup = ScriptedProvider("b", ["from backup"])
    llm = ReliableLLM([primary, backup], retry=RetryPolicy(max_attempts=3), **NO_SLEEP)
    result = llm.generate("hi")
    assert primary.calls == 1
    assert result.response.text == "from backup"
    assert result.fallback_used


def test_all_providers_failed():
    llm = ReliableLLM([MockProvider(always_fail=True)], retry=RetryPolicy(max_attempts=2), **NO_SLEEP)
    with pytest.raises(AllProvidersFailed) as err:
        llm.generate("hi")
    assert len(err.value.errors) == 2


def test_backoff_is_exponential_and_capped():
    import random

    policy = RetryPolicy(base_delay_s=1, max_delay_s=5, jitter=False)
    rng = random.Random(0)
    assert [policy.delay(n, rng) for n in (1, 2, 3, 4)] == [1, 2, 4, 5]


def test_circuit_breaker_opens_half_opens_and_closes():
    now = [0.0]
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_s=10, clock=lambda: now[0])
    cb.record_failure()
    assert cb.state == "closed"
    cb.record_failure()
    assert cb.state == "open" and not cb.allow_request()
    now[0] = 11
    assert cb.state == "half_open" and cb.allow_request()
    cb.record_failure()  # failed probe re-opens
    assert cb.state == "open"
    now[0] = 22
    cb.record_success()
    assert cb.state == "closed"


def test_open_breaker_routes_straight_to_fallback():
    primary = ScriptedProvider("a", [retryable()] * 10)
    backup = ScriptedProvider("b", ["ok"] * 10)
    llm = ReliableLLM(
        [primary, backup],
        retry=RetryPolicy(max_attempts=2),
        breaker_factory=lambda: CircuitBreaker(failure_threshold=2, recovery_timeout_s=999),
        **NO_SLEEP,
    )
    llm.generate("1")  # 2 failures -> breaker opens
    calls_before = primary.calls
    result = llm.generate("2")
    assert primary.calls == calls_before  # primary skipped entirely
    assert result.fallback_used


def test_rate_limiter_waits_when_window_is_full():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    limiter = RateLimiter(rpm=2, clock=lambda: now[0], sleep=sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # third call in the same minute must wait
    assert slept and slept[0] == pytest.approx(60.05)
