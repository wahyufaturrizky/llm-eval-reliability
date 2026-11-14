"""Provider adapters are tested against recorded-shape responses via httpx.MockTransport (no network)."""

import json

import httpx
import pytest

from llm_eval.providers import AnthropicProvider, GeminiProvider, OllamaProvider, OpenAIProvider, build_provider
from llm_eval.providers.base import ProviderError


def client(status: int, payload: dict, seen: list | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gemini_parses_text_and_usage_and_sends_system_prompt():
    seen = []
    payload = {
        "candidates": [{"content": {"parts": [{"text": "Hello "}, {"text": "world"}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 2},
    }
    p = GeminiProvider("gemini-2.5-flash", api_key="k", client=client(200, payload, seen))
    r = p.generate("hi", system="be nice")
    body = json.loads(seen[0].content)
    assert r.text == "Hello world" and (r.input_tokens, r.output_tokens) == (11, 2)
    assert seen[0].headers["x-goog-api-key"] == "k"
    assert body["systemInstruction"]["parts"][0]["text"] == "be nice"
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_openai_parses_response():
    payload = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    r = OpenAIProvider(api_key="k", client=client(200, payload)).generate("hi")
    assert (r.text, r.input_tokens, r.output_tokens) == ("ok", 5, 1)


def test_anthropic_parses_response_and_headers():
    seen = []
    payload = {"content": [{"type": "text", "text": "hi there"}], "usage": {"input_tokens": 7, "output_tokens": 3}}
    r = AnthropicProvider(api_key="k", client=client(200, payload, seen)).generate("hi", system="sys")
    assert r.text == "hi there" and r.input_tokens == 7
    assert seen[0].headers["anthropic-version"] == "2023-06-01"
    assert json.loads(seen[0].content)["system"] == "sys"


def test_ollama_parses_response():
    payload = {"message": {"content": "local"}, "prompt_eval_count": 9, "eval_count": 4}
    r = OllamaProvider(client=client(200, payload)).generate("hi")
    assert (r.text, r.output_tokens) == ("local", 4)


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (400, False), (401, False)])
def test_http_errors_are_classified(status, retryable):
    p = OpenAIProvider(api_key="k", client=client(status, {"error": "x"}))
    with pytest.raises(ProviderError) as err:
        p.generate("hi")
    assert err.value.retryable is retryable and err.value.status_code == status


def test_missing_api_key_is_not_retryable(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError) as err:
        GeminiProvider(client=client(200, {})).generate("hi")
    assert not err.value.retryable


def test_build_provider_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_provider({"type": "nope"})
