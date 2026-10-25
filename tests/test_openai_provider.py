import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google import genai
from hcaptcha_challenger.models import ChallengeRouterResult
from pydantic import SecretStr

from extensions.llm_adapter import (
    OpenAICompatibleGenAIClient,
    _GLMAsyncModels,
    _GLMAsyncNamespace,
    _resolve_openai_endpoint,
    apply_llm_patch,
)
from settings import EpicSettings


def _openai_settings(**overrides):
    base = dict(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY=SecretStr("test-key"),
        OPENAI_BASE_URL="https://api.example.invalid/v1",
        OPENAI_MODEL="step-3.7-flash",
        OPENAI_MAX_TOKENS=32000,
        OPENAI_REQUEST_TIMEOUT_SECONDS=45,
        GLM_API_KEY=None,
        GEMINI_API_KEY=None,
        GLM_BASE_URL="https://open.bigmodel.cn/api/paas/v4",
        GLM_MODEL="glm-4.6v",
        GLM_REQUEST_TIMEOUT_SECONDS=50.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _config(schema=None):
    return SimpleNamespace(
        response_schema=schema, system_instruction=None, temperature=None, thinking_config=None
    )


class _FakeResponse:
    def __init__(self, payload: dict, *, is_error: bool = False, status_code: int = 200):
        self._payload = payload
        self.is_error = is_error
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise AssertionError("raise_for_status should not be reached")


def _capture_client(monkeypatch, responses):
    captured = {}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured.setdefault("payloads", []).append(json)
            return responses[len(captured["payloads"]) - 1]

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: Client())
    return captured


def test_apply_llm_patch_openai_swaps_genai_client(monkeypatch):
    monkeypatch.setattr(genai, "Client", genai.Client)
    apply_llm_patch(_openai_settings())
    assert genai.Client is OpenAICompatibleGenAIClient


def test_apply_llm_patch_openai_without_key_keeps_client(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(genai, "Client", sentinel)
    apply_llm_patch(_openai_settings(OPENAI_API_KEY=None))
    assert genai.Client is sentinel


def test_openai_endpoint_resolution():
    endpoint = _resolve_openai_endpoint(_openai_settings())
    assert endpoint.base_url == "https://api.example.invalid/v1"
    assert endpoint.api_key == "test-key"
    assert endpoint.timeout == 45.0
    assert endpoint.label == "OpenAI"
    assert endpoint.max_tokens == 32000


def test_openai_payload_carries_max_tokens():
    """Reasoning models need a large budget or `content` comes back empty."""
    settings = _openai_settings()
    models = _GLMAsyncModels(settings, {}, endpoint=_resolve_openai_endpoint(settings))
    payload = models._build_payload(
        model="step-3.7-flash", contents=[], config=_config(schema=ChallengeRouterResult), kwargs={}
    )
    assert payload["max_tokens"] == 32000


def test_glm_payload_has_no_max_tokens():
    """GLM route behavior must stay unchanged."""
    models = _GLMAsyncModels(settings=None, storage={})
    payload = models._build_payload(
        model="glm-4.6v", contents=[], config=_config(schema=ChallengeRouterResult), kwargs={}
    )
    assert "max_tokens" not in payload


def test_openai_namespace_wires_endpoint():
    settings = _openai_settings()
    namespace = _GLMAsyncNamespace(settings, {}, endpoint=_resolve_openai_endpoint(settings))
    assert namespace.models._endpoint.label == "OpenAI"


def test_openai_client_reads_settings_singleton(monkeypatch):
    fake_module = types.ModuleType("settings")
    fake_module.settings = _openai_settings()
    monkeypatch.setitem(sys.modules, "settings", fake_module)

    client = OpenAICompatibleGenAIClient()
    assert client.aio.models._endpoint.label == "OpenAI"
    assert client.aio.models._endpoint.base_url == "https://api.example.invalid/v1"


def test_generate_content_posts_to_openai_endpoint(monkeypatch):
    responses = [_FakeResponse({"choices": [{"message": {"content": '{"answer":"ok"}'}}]})]
    captured = _capture_client(monkeypatch, responses)

    settings = _openai_settings()
    models = _GLMAsyncModels(settings, {}, endpoint=_resolve_openai_endpoint(settings))
    response = asyncio.run(models.generate_content("step-3.7-flash", [], config=_config()))

    assert captured["url"] == "https://api.example.invalid/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert response.text == '{"answer":"ok"}'


def test_generate_content_retries_once_without_json_mode(monkeypatch):
    reject = _FakeResponse(
        {"error": {"code": "20024", "message": "json mode is not supported"}},
        is_error=True,
        status_code=400,
    )
    accept = _FakeResponse(
        {"choices": [{"message": {"content": '{"challenge_type":"image_label_single_select"}'}}]}
    )
    captured = _capture_client(monkeypatch, [reject, accept])

    settings = _openai_settings()
    models = _GLMAsyncModels(settings, {}, endpoint=_resolve_openai_endpoint(settings))
    response = asyncio.run(
        models.generate_content("step-3.7-flash", [], config=_config(schema=ChallengeRouterResult))
    )

    assert len(captured["payloads"]) == 2
    assert captured["payloads"][0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in captured["payloads"][1]
    assert "step-3.7-flash" in models._models_without_json_mode
    assert response.parsed.challenge_type.value == "image_label_single_select"


def test_other_400_errors_are_not_retried(monkeypatch):
    reject = _FakeResponse(
        {"error": {"message": "invalid image message"}}, is_error=True, status_code=400
    )
    captured = _capture_client(monkeypatch, [reject])

    settings = _openai_settings()
    models = _GLMAsyncModels(settings, {}, endpoint=_resolve_openai_endpoint(settings))
    with pytest.raises(Exception):
        asyncio.run(
            models.generate_content(
                "step-3.7-flash", [], config=_config(schema=ChallengeRouterResult)
            )
        )

    assert len(captured["payloads"]) == 1


def test_extract_text_falls_back_to_reasoning_content():
    client = _GLMAsyncModels(settings=None, storage={})
    data = {"choices": [{"message": {"content": "", "reasoning_content": '{"answer":"ok"}'}}]}
    assert client._extract_text(data) == '{"answer":"ok"}'


def test_extract_text_raises_when_content_and_reasoning_empty():
    client = _GLMAsyncModels(settings=None, storage={})
    with pytest.raises(ValueError, match="content is empty"):
        client._extract_text({"choices": [{"message": {"content": "  "}}]})


def test_epic_settings_openai_route_defaults():
    settings = EpicSettings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="step-3.7-flash",
        EPIC_EMAIL="user@example.com",
        EPIC_PASSWORD="password",
    )

    assert settings.LLM_PROVIDER == "openai"
    assert settings.llm_configuration_error is None
    assert settings.CHALLENGE_CLASSIFIER_MODEL == "step-3.7-flash"
    assert settings.IMAGE_CLASSIFIER_MODEL == "step-3.7-flash"
    assert settings.SPATIAL_POINT_REASONER_MODEL == "step-3.7-flash"
    assert settings.SPATIAL_PATH_REASONER_MODEL == "step-3.7-flash"
    # hcaptcha-challenger base config still expects GEMINI_API_KEY to be present.
    assert settings.GEMINI_API_KEY is not None


def test_epic_settings_openai_route_requires_key():
    settings = EpicSettings(
        LLM_PROVIDER="openai",
        OPENAI_MODEL="step-3.7-flash",
        EPIC_EMAIL="user@example.com",
        EPIC_PASSWORD="password",
    )
    assert settings.llm_configuration_error is not None
    assert "OPENAI_API_KEY" in settings.llm_configuration_error


def test_workflow_passes_openai_env():
    workflow = Path(__file__).parent.parent / ".github" / "workflows" / "epic-gamer.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}" in text
    assert "OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}" in text
    assert "OPENAI_MODEL: ${{ vars.OPENAI_MODEL || secrets.OPENAI_MODEL }}" in text
