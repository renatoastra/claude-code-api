"""Endpoint tests for OpenAI function calling via the mock Claude binary."""

import json

from tests.model_utils import get_test_model_id

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_index_repository",
            "description": "Start the OpenWiki indexing pipeline.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _request(stream: bool):
    return {
        "model": get_test_model_id(),
        "messages": [
            {"role": "system", "content": "You are Eve."},
            {"role": "user", "content": "Start the OpenWiki indexing pipeline."},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "stream": stream,
    }


def test_non_streaming_tool_call(test_client):
    resp = test_client.post("/v1/chat/completions", json=_request(stream=False))
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] == "Starting the indexing pipeline."
    assert len(message["tool_calls"]) == 1
    call = message["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "run_index_repository"
    assert json.loads(call["function"]["arguments"]) == {}


def test_streaming_tool_call(test_client):
    resp = test_client.post("/v1/chat/completions", json=_request(stream=True))
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert deltas[0]["role"] == "assistant"
    tool_deltas = [d for d in deltas if "tool_calls" in d]
    assert len(tool_deltas) == 1
    call = tool_deltas[0]["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"]["name"] == "run_index_repository"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_keeps_plain_text_path(test_client):
    body = _request(stream=False)
    body["tool_choice"] = "none"
    body["messages"][-1]["content"] = "Hi"
    resp = test_client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert not choice["message"].get("tool_calls")
