import json
from pathlib import Path

from llm_eval.cli import main
from llm_eval.runner import load_config, percentile, regression_gates, run_suite

ROOT = Path(__file__).resolve().parents[1]


def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([], 95) == 0.0


def test_demo_suite_passes_all_budget_gates(tmp_path):
    report = run_suite(load_config(ROOT / "configs/demo.yaml"), out_dir=tmp_path)
    assert report.passed, [g for g in report.gates if not g.passed]
    assert report.summary["pass_rate"] == 1.0
    assert report.summary["retries"] > 0  # fault injection was exercised
    traces = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert any(json.loads(t)["name"] == "llm.call" for t in traces)


def test_regressed_model_is_blocked_by_gates(tmp_path):
    baseline = json.loads((ROOT / "baselines/demo.json").read_text())
    report = run_suite(load_config(ROOT / "configs/regressed.yaml"), out_dir=tmp_path)
    gates = {g.name: g for g in report.gates + regression_gates(report.summary, baseline, {})}
    assert not gates["min_pass_rate"].passed
    assert not gates["regression.newly_failing_cases"].passed
    assert "pii-echo-007" in gates["regression.newly_failing_cases"].detail


def test_cli_exit_codes(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    ok = main(["run", "-c", "configs/demo.yaml", "-b", "baselines/demo.json", "-o", str(tmp_path / "ok")])
    bad = main(["run", "-c", "configs/regressed.yaml", "-b", "baselines/demo.json", "-o", str(tmp_path / "bad")])
    assert (ok, bad) == (0, 1)
    assert (tmp_path / "bad" / "report.md").exists()
