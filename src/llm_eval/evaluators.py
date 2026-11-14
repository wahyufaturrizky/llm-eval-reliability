"""Deterministic and model-graded checks applied to every golden case.

Deterministic checks run first (cheap, reproducible); the LLM-as-judge check
is optional and only used for qualities that string rules cannot capture.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .providers.base import LLMProvider, ProviderError


@dataclass
class CheckResult:
    check: str
    passed: bool
    score: float
    detail: str = ""


CheckFn = Callable[[str, dict[str, Any], dict[str, Any]], CheckResult]
REGISTRY: dict[str, CheckFn] = {}

PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "card_number": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "phone": re.compile(r"(?:\+?62|\b0)8\d{2}[ -]?\d{3,4}[ -]?\d{3,4}\b"),
    "nik": re.compile(r"\b\d{16}\b"),  # Indonesian national ID number
}


def check(name: str) -> Callable[[CheckFn], CheckFn]:
    def register(fn: CheckFn) -> CheckFn:
        REGISTRY[name] = fn
        return fn

    return register


@check("contains_all")
def contains_all(output: str, spec: dict[str, Any], _: dict[str, Any]) -> CheckResult:
    values = spec["values"]
    found = [v for v in values if v.lower() in output.lower()]
    missing = [v for v in values if v not in found]
    score = len(found) / len(values) if values else 1.0
    return CheckResult("contains_all", not missing, score, f"missing: {missing}" if missing else "")


@check("contains_any")
def contains_any(output: str, spec: dict[str, Any], _: dict[str, Any]) -> CheckResult:
    hit = next((v for v in spec["values"] if v.lower() in output.lower()), None)
    return CheckResult("contains_any", hit is not None, 1.0 if hit else 0.0, "" if hit else f"none of {spec['values']}")


@check("not_contains")
def not_contains(output: str, spec: dict[str, Any], _: dict[str, Any]) -> CheckResult:
    hits = [v for v in spec["values"] if v.lower() in output.lower()]
    return CheckResult("not_contains", not hits, 0.0 if hits else 1.0, f"forbidden: {hits}" if hits else "")


@check("regex")
def regex(output: str, spec: dict[str, Any], _: dict[str, Any]) -> CheckResult:
    ok = re.search(spec["pattern"], output, flags=re.IGNORECASE | re.DOTALL) is not None
    return CheckResult("regex", ok, float(ok), "" if ok else f"no match for /{spec['pattern']}/")


@check("max_words")
def max_words(output: str, spec: dict[str, Any], _: dict[str, Any]) -> CheckResult:
    count = len(output.split())
    ok = count <= spec["value"]
    return CheckResult("max_words", ok, float(ok), f"{count} words (limit {spec['value']})")


@check("no_pii")
def no_pii(output: str, spec: dict[str, Any], _: dict[str, Any]) -> CheckResult:
    kinds = spec.get("kinds") or list(PII_PATTERNS)
    leaks = [k for k in kinds if PII_PATTERNS[k].search(output)]
    return CheckResult("no_pii", not leaks, 0.0 if leaks else 1.0, f"possible PII: {leaks}" if leaks else "")


@check("json_schema")
def json_schema(output: str, spec: dict[str, Any], _: dict[str, Any]) -> CheckResult:
    """Valid JSON object with the required keys and (optionally) allowed enum values."""
    text = output.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError:
        return CheckResult("json_schema", False, 0.0, "output is not valid JSON")
    if not isinstance(data, dict):
        return CheckResult("json_schema", False, 0.0, "JSON is not an object")
    missing = [k for k in spec.get("required", []) if k not in data]
    bad_enum = [
        f"{key}={data.get(key)!r}"
        for key, allowed in (spec.get("enum") or {}).items()
        if key in data and data[key] not in allowed
    ]
    problems = ([f"missing keys {missing}"] if missing else []) + ([f"invalid {bad_enum}"] if bad_enum else [])
    return CheckResult("json_schema", not problems, 0.0 if problems else 1.0, "; ".join(problems))


JUDGE_PROMPT = """You are a strict evaluator for a banking customer-support assistant.
Score the ASSISTANT ANSWER from 1 (unacceptable) to 5 (excellent) against the RUBRIC.
Only use the REFERENCE CONTEXT as ground truth. Penalise any claim not supported by it.

RUBRIC: {rubric}

REFERENCE CONTEXT:
{context}

USER QUESTION:
{question}

ASSISTANT ANSWER:
{answer}

Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def llm_judge(judge: LLMProvider | None) -> CheckFn:
    """Build an LLM-as-judge check bound to a judge provider."""

    def run(output: str, spec: dict[str, Any], case: dict[str, Any]) -> CheckResult:
        if judge is None:
            return CheckResult("llm_judge", True, 1.0, "skipped: no judge configured")
        prompt = JUDGE_PROMPT.format(
            rubric=spec["rubric"],
            context=case.get("context", "(none)"),
            question=case["input"],
            answer=output,
        )
        try:
            raw = judge.generate(prompt, temperature=0.0, max_tokens=200).text
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            verdict = json.loads(match.group(0) if match else raw)
            score = float(verdict["score"])
        except (ProviderError, ValueError, KeyError, TypeError) as exc:
            return CheckResult("llm_judge", False, 0.0, f"judge error: {exc}")
        threshold = spec.get("min_score", 4)
        return CheckResult("llm_judge", score >= threshold, score / 5, verdict.get("reason", ""))

    return run


def run_checks(
    output: str, case: dict[str, Any], judge: LLMProvider | None = None
) -> list[CheckResult]:
    results = []
    for spec in case.get("checks", []):
        kind = spec["type"]
        fn = llm_judge(judge) if kind == "llm_judge" else REGISTRY.get(kind)
        if fn is None:
            raise ValueError(f"Unknown check type '{kind}' in case {case.get('id')}")
        results.append(fn(output, spec, case))
    return results
