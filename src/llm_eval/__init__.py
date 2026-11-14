"""llm-eval-reliability: golden-dataset evaluation, regression gates and resilience for LLM apps."""

from .evaluators import CheckResult, run_checks
from .reliability import AllProvidersFailed, CircuitBreaker, ReliableLLM, RetryPolicy
from .runner import load_config, run_suite

__version__ = "0.1.0"

__all__ = [
    "AllProvidersFailed",
    "CheckResult",
    "CircuitBreaker",
    "ReliableLLM",
    "RetryPolicy",
    "load_config",
    "run_checks",
    "run_suite",
]
