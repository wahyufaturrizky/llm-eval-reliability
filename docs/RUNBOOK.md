# Runbook: when an LLM eval gate fails

Use this when `llm-eval` exits with code 1 in CI, or when the live eval shows degraded results.

## 1. Read the report first

- `reports/report.md` lists the failed gates and, for every failing case, the check that failed and why.
- `reports/traces.jsonl` has one span per attempt: provider, attempt number, latency, tokens and error.

```bash
# Which provider calls failed, and why?
jq -c 'select(.status=="error") | {provider: .attributes.provider, attempt: .attributes.attempt, error}' reports/traces.jsonl
```

## 2. Triage by gate

| Failed gate | Likely cause | First action |
|---|---|---|
| `regression.newly_failing_cases` | Prompt or model change broke specific behaviour | Diff the outputs of those case ids against the last green run; revert or fix the prompt |
| `min_pass_rate` / `regression.pass_rate` | Broad quality drop | Check whether a fallback model served most answers (`fallbacks` in the summary) |
| `max_error_rate` | Provider outage, bad key, or quota exhausted | Look for 401 (key), 429 (quota) or 5xx (outage) in traces |
| `max_p95_latency_ms` / `regression.p95_latency` | Slower model, longer outputs, or retries | Compare `output_tokens`; lower `max_tokens`; review retry settings |
| `max_cost_per_case_usd` / `regression.cost_per_case` | Fallback to a pricier model or longer prompts | Check which provider served the cases; trim context |

## 3. Safety failures are release blockers

Failures in cases tagged `safety`, `pii` or `prompt-injection` must not be waived with a baseline update. Fix the system prompt or add a guardrail, then re-run.

## 4. Intentional changes

If the change is an improvement (for example, a stricter prompt now passes more cases):

```bash
make baseline            # writes baselines/demo.json from this run
git add baselines/ && git commit -m "eval: refresh baseline after <change>"
```

Reviewers then see the metric changes in the baseline diff.

## 5. Provider incident checklist

1. The circuit breaker should already route traffic to the fallback model. Confirm `fallbacks > 0` in the report.
2. If the fallback model is also failing, check the provider status pages and your quotas.
3. For long incidents, reorder the `providers:` chain so the healthy model is primary.
