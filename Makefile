.PHONY: install test lint eval eval-regressed eval-gemini baseline

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

eval:            ## offline demo (no API key): passes all gates
	llm-eval run -c configs/demo.yaml -b baselines/demo.json

eval-regressed:  ## same suite with a regressed model: gates block the release (exit code 1)
	llm-eval run -c configs/regressed.yaml -b baselines/demo.json -o reports/regressed

eval-gemini:     ## real run on Gemini free tier (needs GEMINI_API_KEY in .env)
	llm-eval run -c configs/gemini.yaml -b baselines/gemini.json -o reports/gemini

baseline:        ## refresh the demo baseline after an intentional change
	llm-eval run -c configs/demo.yaml -b baselines/demo.json --update-baseline
