from llm_eval.evaluators import run_checks
from llm_eval.providers import MockProvider
from llm_eval.providers.mock import MockRule


def check(output, *specs, judge=None):
    return run_checks(output, {"id": "t", "input": "q", "checks": list(specs)}, judge)


def test_contains_all_reports_partial_score():
    [r] = check("Block your card now", {"type": "contains_all", "values": ["block", "hotline"]})
    assert not r.passed and r.score == 0.5 and "hotline" in r.detail


def test_not_contains_and_contains_any():
    a, b = check(
        "I can't provide investment advice.",
        {"type": "contains_any", "values": ["can't provide investment advice"]},
        {"type": "not_contains", "values": ["guaranteed"]},
    )
    assert a.passed and b.passed


def test_no_pii_detects_card_and_email():
    [r] = check("Card 4111 1111 1111 1111, mail a@b.co", {"type": "no_pii"})
    assert not r.passed and "card_number" in r.detail and "email" in r.detail


def test_no_pii_allows_amounts_and_hotlines():
    [r] = check("Fee is IDR 150,000; call 1500-321.", {"type": "no_pii"})
    assert r.passed


def test_json_schema_accepts_fenced_json_and_checks_enums():
    spec = {"type": "json_schema", "required": ["intent", "urgency"], "enum": {"urgency": ["low", "high"]}}
    [ok] = check('```json\n{"intent": "fraud_report", "urgency": "high"}\n```', spec)
    [bad] = check('{"intent": "fraud_report", "urgency": "extreme"}', spec)
    [broken] = check("intent: fraud", spec)
    assert ok.passed and not bad.passed and not broken.passed


def test_max_words_and_regex():
    a, b = check("one two three", {"type": "max_words", "value": 2}, {"type": "regex", "pattern": r"two\s+three"})
    assert not a.passed and b.passed


def test_llm_judge_uses_threshold_and_handles_bad_output():
    good = MockProvider(rules=[MockRule("strict evaluator", '{"score": 4, "reason": "fine"}')])
    garbage = MockProvider(rules=[MockRule("strict evaluator", "looks good to me!")])
    spec = {"type": "llm_judge", "rubric": "be helpful", "min_score": 4}
    [r1] = check("answer", spec, judge=good)
    [r2] = check("answer", spec, judge=garbage)
    [r3] = check("answer", spec, judge=None)
    assert r1.passed and r1.score == 0.8
    assert not r2.passed and "judge error" in r2.detail
    assert r3.passed and "skipped" in r3.detail
