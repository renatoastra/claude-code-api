"""Unit tests for OpenAI-style function calling emulation."""

import json

from claude_code_api.models.openai import ChatMessage, ToolDefinition
from claude_code_api.utils.tool_calling import (
    TOOL_RESPONSE_SCHEMA,
    build_tools_system_prompt,
    extract_tool_response,
    render_conversation,
    tools_enabled,
)


def _tools():
    return [
        ToolDefinition(
            type="function",
            function={
                "name": "run_index_repository",
                "description": "Start indexing.",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
        ToolDefinition(
            type="function",
            function={
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        ),
    ]


def test_tools_enabled_respects_tool_choice_none():
    assert tools_enabled(_tools(), None) is True
    assert tools_enabled(_tools(), "auto") is True
    assert tools_enabled(_tools(), "none") is False
    assert tools_enabled(None, "auto") is False
    assert tools_enabled([], "auto") is False


def test_system_prompt_lists_tools_and_keeps_base():
    prompt = build_tools_system_prompt(_tools(), None, "Base instructions.")
    assert prompt.startswith("Base instructions.")
    assert "run_index_repository" in prompt
    assert "Start indexing." in prompt
    assert '"path"' in prompt
    assert "tool_calls" in prompt


def test_system_prompt_tool_choice_required_and_specific():
    required = build_tools_system_prompt(_tools(), "required", None)
    assert "must call at least one tool" in required
    specific = build_tools_system_prompt(
        _tools(), {"type": "function", "function": {"name": "read_file"}}, None
    )
    assert "must call the tool `read_file`" in specific


def test_render_conversation_includes_history_tool_calls_and_results():
    messages = [
        ChatMessage(role="system", content="ignored here"),
        ChatMessage(role="user", content="Index the repo"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_index_repository", "arguments": "{}"},
                }
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_1", content='{"status":"ok"}'),
    ]
    rendered = render_conversation(messages)
    assert "ignored here" not in rendered
    assert "[user]: Index the repo" in rendered
    assert "run_index_repository" in rendered and "call_1" in rendered
    assert '{"status":"ok"}' in rendered
    assert rendered.index("[user]") < rendered.index("[tool result")


def test_extract_tool_response_from_structured_output():
    messages = [
        {"type": "system", "subtype": "init"},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "StructuredOutput",
                        "input": {
                            "content": "Starting.",
                            "tool_calls": [
                                {"name": "read_file", "arguments": {"path": "a.md"}},
                                {"name": "run_index_repository", "arguments": {}},
                            ],
                        },
                    }
                ],
            },
        },
        {"type": "result", "subtype": "success"},
    ]
    content, tool_calls = extract_tool_response(messages)
    assert content == "Starting."
    assert len(tool_calls) == 2
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["id"].startswith("call_")
    assert tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"path": "a.md"}
    assert tool_calls[1]["function"]["arguments"] == "{}"
    assert tool_calls[0]["id"] != tool_calls[1]["id"]


def test_extract_tool_response_without_structured_output_returns_none():
    messages = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
            },
        }
    ]
    assert extract_tool_response(messages) is None


def test_schema_shape():
    assert TOOL_RESPONSE_SCHEMA["required"] == ["content", "tool_calls"]
