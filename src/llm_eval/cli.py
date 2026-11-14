"""Command-line entry point: ``llm-eval run -c configs/demo.yaml --baseline baselines/demo.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import load_config, regression_gates, run_suite, write_reports


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so the project needs no extra dependency."""
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = run_suite(config, out_dir=args.out)

    if args.baseline and not args.update_baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            report.gates.extend(regression_gates(report.summary, baseline, config.get("regression", {})))
        else:
            print(f"⚠️  Baseline {baseline_path} not found, skipping regression checks.")

    json_path, md_path = write_reports(report, args.out)

    if args.update_baseline:
        target = Path(args.baseline or "baselines/baseline.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.summary, indent=2), encoding="utf-8")
        print(f"📌 Baseline updated: {target}")

    print(md_path.read_text(encoding="utf-8"))
    print(f"Reports: {json_path} · {md_path} · {Path(args.out) / 'traces.jsonl'}")
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="llm-eval", description="LLM evaluation & reliability framework")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an eval suite and enforce budget/regression gates")
    run.add_argument("-c", "--config", required=True, help="Path to suite YAML config")
    run.add_argument("-b", "--baseline", help="Baseline summary JSON to compare against")
    run.add_argument("-o", "--out", default="reports", help="Output directory (default: reports)")
    run.add_argument("--update-baseline", action="store_true", help="Write this run's summary as the new baseline")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
