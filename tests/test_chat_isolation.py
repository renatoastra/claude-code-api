"""Endpoint tests: real usage passthrough and tool isolation outside tool mode."""

import claude_code_api.api.chat as chat_module
from claude_code_api.core.config import settings
from tests.model_utils import get_test_model_id


def _request(**extra):
    body = {
        "model": get_test_model_id(),
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    body.update(extra)
    return body


def test_non_streaming_usage_comes_from_cli_result(test_client):
    # claude_stream_simple.jsonl: usage input 12 / output 8, cost_usd 0.00002.
    resp = test_client.post("/v1/chat/completions", json=_request())
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] == 12
    assert usage["completion_tokens"] == 8
    assert usage["total_tokens"] == 20
    assert usage["total_cost_usd"] == 0.00002


def _capture_create_session(monkeypatch, test_client):
    captured = {}
    manager = test_client.app.state.claude_manager
    original = manager.create_session

    async def spy(**kwargs):
        captured.update(kwargs)
        return await original(**kwargs)

    monkeypatch.setattr(manager, "create_session", spy)
    return captured


def test_isolate_tools_is_off_by_default(test_client, monkeypatch):
    captured = _capture_create_session(monkeypatch, test_client)
    resp = test_client.post("/v1/chat/completions", json=_request())
    assert resp.status_code == 200, resp.text
    assert captured["isolate_tools"] is False
    assert captured["json_schema"] is None


def test_isolate_tools_request_field_isolates_without_tool_mode(
    test_client, monkeypatch
):
    captured = _capture_create_session(monkeypatch, test_client)
    resp = test_client.post("/v1/chat/completions", json=_request(isolate_tools=True))
    assert resp.status_code == 200, resp.text
    assert captured["isolate_tools"] is True
    # Plain single-turn call: no StructuredOutput schema, so `--tools ""` applies.
    assert captured["json_schema"] is None


def test_isolate_tools_default_setting_applies_and_request_overrides(
    test_client, monkeypatch
):
    captured = _capture_create_session(monkeypatch, test_client)
    monkeypatch.setattr(settings, "isolate_tools_default", True)

    resp = test_client.post("/v1/chat/completions", json=_request())
    assert resp.status_code == 200, resp.text
    assert captured["isolate_tools"] is True

    resp = test_client.post("/v1/chat/completions", json=_request(isolate_tools=False))
    assert resp.status_code == 200, resp.text
    assert captured["isolate_tools"] is False


def test_tool_mode_always_isolates(test_client, monkeypatch):
    captured = _capture_create_session(monkeypatch, test_client)
    tools = [
        {
            "type": "function",
            "function": {"name": "noop", "parameters": {"type": "object"}},
        }
    ]
    resp = test_client.post(
        "/v1/chat/completions",
        json=_request(tools=tools, tool_choice="auto", isolate_tools=False),
    )
    assert resp.status_code == 200, resp.text
    assert captured["isolate_tools"] is True
    assert captured["json_schema"] is chat_module.TOOL_RESPONSE_SCHEMA
