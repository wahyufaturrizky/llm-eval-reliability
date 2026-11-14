"""Runs a golden dataset through a reliable LLM chain and gates on budgets + regressions."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .evaluators import CheckResult, run_checks
from .providers import build_provider
from .providers.base import LLMProvider
from .reliability import AllProvidersFailed, CircuitBreaker, ReliableLLM, RetryPolicy
from .tracing import Tracer


# --------------------------------------------------------------------------- data
@dataclass
class CaseResult:
    id: str
    tags: list[str]
    passed: bool
    output: str
    provider: str | None
    attempts: int
    fallback_used: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    suite: str
    started_at: str
    duration_s: float
    summary: dict[str, Any]
    gates: list[Gate]
    cases: list[CaseResult]

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


# --------------------------------------------------------------------------- helpers
def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_base_dir"] = str(path.parent.parent if path.parent.name == "configs" else path.parent)
    return config


def _resolve(config: dict[str, Any], value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(config["_base_dir"]) / p


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("//"):
            try:
                cases.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case ids in {path}")
    return cases


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 2)


def cost_usd(pricing: dict[str, Any], model: str, input_tokens: int, output_tokens: int) -> float:
    price = pricing.get(model) or {}
    return round(
        input_tokens / 1e6 * price.get("input_per_1m", 0.0) + output_tokens / 1e6 * price.get("output_per_1m", 0.0),
        8,
    )


def _build_provider(config: dict[str, Any], spec: dict[str, Any]) -> LLMProvider:
    spec = dict(spec)
    if "responses_file" in spec:
        spec["responses_file"] = str(_resolve(config, spec["responses_file"]))
    return build_provider(spec)


def build_llm(config: dict[str, Any], tracer: Tracer) -> ReliableLLM:
    rel = config.get("reliability", {})
    breaker = rel.get("circuit_breaker", {})
    return ReliableLLM(
        [_build_provider(config, s) for s in config["providers"]],
        retry=RetryPolicy(**rel.get("retry", {})),
        breaker_factory=lambda: CircuitBreaker(**breaker),
        rate_limit_rpm=rel.get("rate_limit_rpm"),
        tracer=tracer,
    )


# --------------------------------------------------------------------------- run
def run_suite(config: dict[str, Any], *, out_dir: str | Path = "reports") -> Report:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(out_dir / "traces.jsonl", otel=config.get("tracing", {}).get("otel", False))
    llm = build_llm(config, tracer)
    judge = _build_provider(config, config["judge"]) if config.get("judge") else None
    cases = load_dataset(_resolve(config, config["dataset"]))
    gen = config.get("generation", {})
    template = config.get("prompt_template", "{input}")
    pricing = config.get("pricing", {})

    def run_case(case: dict[str, Any]) -> CaseResult:
        prompt = template.format(input=case["input"], context=case.get("context", ""))
        with tracer.trace(), tracer.span("eval.case", case_id=case["id"]) as span:
            try:
                result = llm.generate(
                    prompt,
                    system=config.get("system_prompt"),
                    temperature=gen.get("temperature", 0.0),
                    max_tokens=gen.get("max_tokens", 400),
                )
            except AllProvidersFailed as exc:
                span.fail(str(exc))
                return CaseResult(case["id"], case.get("tags", []), False, "", None, len(exc.errors),
                                  False, 0.0, 0, 0, 0.0, error=str(exc))
            r = result.response
            checks = run_checks(r.text, case, judge)
            passed = all(c.passed for c in checks)
            span.set(passed=passed, provider=f"{r.provider}:{r.model}")
            return CaseResult(
                id=case["id"],
                tags=case.get("tags", []),
                passed=passed,
                output=r.text,
                provider=f"{r.provider}:{r.model}",
                attempts=result.attempts,
                fallback_used=result.fallback_used,
                latency_ms=r.latency_ms,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                cost_usd=cost_usd(pricing, r.model, r.input_tokens, r.output_tokens),
                checks=checks,
            )

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(config.get("concurrency", 1)))) as pool:
        results = list(pool.map(run_case, cases))
    duration = round(time.time() - started, 2)

    summary = summarize(results)
    report = Report(
        suite=config.get("name", "llm-eval"),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        duration_s=duration,
        summary=summary,
        gates=budget_gates(summary, config.get("budgets", {})),
        cases=results,
    )
    return report


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    ok = [r for r in results if r.error is None]
    latencies = [r.latency_ms for r in ok]
    by_check: dict[str, list[bool]] = {}
    by_tag: dict[str, list[bool]] = {}
    for r in results:
        for c in r.checks:
            by_check.setdefault(c.check, []).append(c.passed)
        for t in r.tags:
            by_tag.setdefault(t, []).append(r.passed)
    rate = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0
    return {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": rate([r.passed for r in results]),
        "errors": sum(r.error is not None for r in results),
        "fallbacks": sum(r.fallback_used for r in results),
        "retries": sum(max(0, r.attempts - 1) for r in ok),
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "tokens": {"input": sum(r.input_tokens for r in ok), "output": sum(r.output_tokens for r in ok)},
        "cost_usd": {
            "total": round(sum(r.cost_usd for r in ok), 6),
            "per_case": round(sum(r.cost_usd for r in ok) / len(ok), 6) if ok else 0.0,
        },
        "pass_rate_by_check": {k: rate(v) for k, v in sorted(by_check.items())},
        "pass_rate_by_tag": {k: rate(v) for k, v in sorted(by_tag.items())},
        "case_status": {r.id: r.passed for r in results},
    }


def budget_gates(summary: dict[str, Any], budgets: dict[str, Any]) -> list[Gate]:
    gates = []
    if "min_pass_rate" in budgets:
        v, lim = summary["pass_rate"], budgets["min_pass_rate"]
        gates.append(Gate("min_pass_rate", v >= lim, f"{v:.1%} (min {lim:.0%})"))
    if "max_p95_latency_ms" in budgets:
        v, lim = summary["latency_ms"]["p95"], budgets["max_p95_latency_ms"]
        gates.append(Gate("max_p95_latency_ms", v <= lim, f"{v:.0f} ms (max {lim} ms)"))
    if "max_cost_per_case_usd" in budgets:
        v, lim = summary["cost_usd"]["per_case"], budgets["max_cost_per_case_usd"]
        gates.append(Gate("max_cost_per_case_usd", v <= lim, f"${v:.6f} (max ${lim})"))
    if "max_error_rate" in budgets:
        v = summary["errors"] / summary["cases"] if summary["cases"] else 0.0
        lim = budgets["max_error_rate"]
        gates.append(Gate("max_error_rate", v <= lim, f"{v:.1%} (max {lim:.0%})"))
    return gates


def regression_gates(summary: dict[str, Any], baseline: dict[str, Any], tolerance: dict[str, Any]) -> list[Gate]:
    """Compare against a stored baseline so a prompt/model change can't silently get worse."""
    gates = []
    drop = baseline["pass_rate"] - summary["pass_rate"]
    max_drop = tolerance.get("max_pass_rate_drop", 0.0)
    gates.append(Gate("regression.pass_rate", drop <= max_drop + 1e-9,
                      f"{baseline['pass_rate']:.1%} -> {summary['pass_rate']:.1%} (max drop {max_drop:.0%})"))

    newly_failing = sorted(
        cid for cid, ok in baseline.get("case_status", {}).items()
        if ok and summary["case_status"].get(cid) is False
    )
    if tolerance.get("fail_on_newly_failing_cases", True):
        gates.append(Gate("regression.newly_failing_cases", not newly_failing,
                          ", ".join(newly_failing) if newly_failing else "none"))

    base_p95 = baseline["latency_ms"]["p95"]
    max_growth = tolerance.get("max_p95_latency_increase", 0.25)
    if base_p95 > 0:
        growth = summary["latency_ms"]["p95"] / base_p95 - 1
        gates.append(Gate("regression.p95_latency", growth <= max_growth,
                          f"{base_p95:.0f} -> {summary['latency_ms']['p95']:.0f} ms (max +{max_growth:.0%})"))

    base_cost = baseline["cost_usd"]["per_case"]
    max_cost_growth = tolerance.get("max_cost_increase", 0.25)
    if base_cost > 0:
        growth = summary["cost_usd"]["per_case"] / base_cost - 1
        gates.append(Gate("regression.cost_per_case", growth <= max_cost_growth,
                          f"${base_cost:.6f} -> ${summary['cost_usd']['per_case']:.6f} (max +{max_cost_growth:.0%})"))
    return gates


# --------------------------------------------------------------------------- output
def write_reports(report: Report, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_markdown(report: Report) -> str:
    s = report.summary
    lines = [
        f"# LLM Eval Report: {report.suite}",
        "",
        f"**Result:** {'✅ PASSED' if report.passed else '❌ FAILED'} · {report.started_at} · {report.duration_s}s",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pass rate | {s['pass_rate']:.1%} ({s['passed']}/{s['cases']}) |",
        f"| p50 / p95 latency | {s['latency_ms']['p50']:.0f} / {s['latency_ms']['p95']:.0f} ms |",
        f"| Tokens (in / out) | {s['tokens']['input']} / {s['tokens']['output']} |",
        f"| Cost (total / per case) | ${s['cost_usd']['total']:.6f} / ${s['cost_usd']['per_case']:.6f} |",
        f"| Retries · Fallbacks · Errors | {s['retries']} · {s['fallbacks']} · {s['errors']} |",
        "",
        "## Gates",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
        *[f"| {g.name} | {'✅' if g.passed else '❌'} | {g.detail} |" for g in report.gates],
        "",
        "## Pass rate by check",
        "",
        "| Check | Pass rate |",
        "|---|---|",
        *[f"| {k} | {v:.1%} |" for k, v in s["pass_rate_by_check"].items()],
        "",
        "## Failing cases",
        "",
    ]
    failing = [c for c in report.cases if not c.passed]
    if not failing:
        lines.append("None 🎉")
    for c in failing:
        reasons = c.error or "; ".join(f"{x.check}: {x.detail}" for x in c.checks if not x.passed)
        lines.append(f"- **{c.id}** ({', '.join(c.tags)}): {reasons}")
    return "\n".join(lines) + "\n"
