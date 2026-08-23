"""OpenAI-style function calling emulated on top of `claude -p`.

The Claude CLI cannot receive client-defined tools directly. Instead we:

1. Describe the client's tools in the system prompt.
2. Render the whole OpenAI conversation (including prior tool calls and
   tool results) into the prompt, because every CLI run is stateless.
3. Force the CLI to answer through ``--json-schema`` with a
   ``{content, tool_calls}`` shape. The CLI surfaces that answer as a
   ``tool_use`` block named ``StructuredOutput``.
4. Translate that block back into OpenAI ``tool_calls``.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from claude_code_api.models.openai import ChatMessage, ToolChoice, ToolDefinition
from claude_code_api.utils.parser import ClaudeOutputParser, normalize_claude_message

STRUCTURED_OUTPUT_TOOL = "StructuredOutput"

TOOL_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {
            "type": ["string", "null"],
            "description": "Assistant text for the user, or null when only calling tools.",
        },
        "tool_calls": {
            "type": "array",
            "description": "Tools to invoke now. Empty when no tool is needed.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["content", "tool_calls"],
}

ToolChoiceValue = Optional[Union[str, ToolChoice, Dict[str, Any]]]


def tools_enabled(
    tools: Optional[Sequence[ToolDefinition]], tool_choice: ToolChoiceValue
) -> bool:
    """Whether the request should run in function-calling mode."""
    if not tools:
        return False
    return tool_choice != "none"


def _forced_tool_name(tool_choice: ToolChoiceValue) -> Optional[str]:
    if isinstance(tool_choice, ToolChoice):
        return tool_choice.function.name
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function") or {}
        return function.get("name") if isinstance(function, dict) else None
    return None


def build_tools_system_prompt(
    tools: Sequence[ToolDefinition],
    tool_choice: ToolChoiceValue,
    base_system_prompt: Optional[str],
) -> str:
    """Append the tool catalog and calling rules to the system prompt."""
    lines: List[str] = []
    if base_system_prompt:
        lines.append(base_system_prompt.rstrip())
        lines.append("")

    lines.append(
        "You have access to external tools provided by the client application. "
        "These tools are executed by the client, not by you."
    )
    lines.append("Available tools:")
    for tool in tools:
        fn = tool.function
        description = f" — {fn.description}" if fn.description else ""
        params = json.dumps(fn.parameters or {"type": "object"}, ensure_ascii=False)
        lines.append(f"- {fn.name}{description}")
        lines.append(f"  Parameters (JSON Schema): {params}")

    lines.append("")
    lines.append("Rules for responding:")
    lines.append(
        '- To call one or more tools, list them in "tool_calls" (name + arguments '
        'object matching the schema) and set "content" to null or a brief note. '
        "Never describe a tool call in prose instead of calling it."
    )
    lines.append(
        '- When the task is complete and no tool is needed, leave "tool_calls" '
        'empty and answer in "content".'
    )
    lines.append(
        "- Tool results appear in the conversation as [tool result for <id>]. "
        "Use them to continue; do not repeat a call whose result you already have."
    )

    forced = _forced_tool_name(tool_choice)
    if forced:
        lines.append(f"- For this turn you must call the tool `{forced}`.")
    elif tool_choice == "required":
        lines.append("- For this turn you must call at least one tool.")

    return "\n".join(lines)


def _render_assistant(msg: ChatMessage) -> List[str]:
    out: List[str] = []
    text = msg.get_text_content().strip()
    if text:
        out.append(f"[assistant]: {text}")
    for call in msg.tool_calls or []:
        out.append(
            f"[assistant] called tool {call.function.name} (id={call.id}) "
            f"with arguments: {call.function.arguments}"
        )
    return out


def render_conversation(messages: Sequence[ChatMessage]) -> str:
    """Render the full OpenAI message history into a single CLI prompt."""
    lines: List[str] = ["Conversation so far:"]
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            lines.append(f"[user]: {msg.get_text_content().strip()}")
        elif msg.role == "assistant":
            lines.extend(_render_assistant(msg))
        elif msg.role == "tool":
            call_id = msg.tool_call_id or "unknown"
            lines.append(
                f"[tool result for {call_id}]: {msg.get_text_content().strip()}"
            )
    lines.append("")
    lines.append("Respond to the latest state of the conversation.")
    return "\n".join(lines)


def _structured_payloads(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    parser = ClaudeOutputParser()
    payloads: List[Dict[str, Any]] = []
    for raw in messages:
        normalized = normalize_claude_message(raw)
        if not normalized or not parser.is_assistant_message(normalized):
            continue
        for tool_use in parser.extract_tool_uses(normalized):
            if tool_use.name == STRUCTURED_OUTPUT_TOOL and isinstance(
                tool_use.input, dict
            ):
                payloads.append(tool_use.input)
    return payloads


def _to_openai_tool_call(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = call.get("name")
    if not name:
        return None
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {} if arguments is None else {"input": arguments}
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments, separators=(",", ":"), ensure_ascii=False
            ),
        },
    }


def extract_tool_response(
    messages: Sequence[Any],
) -> Optional[Tuple[Optional[str], List[Dict[str, Any]]]]:
    """Return ``(content, openai_tool_calls)`` from the StructuredOutput block.

    Returns ``None`` when the CLI produced no structured output (caller should
    fall back to plain text handling).
    """
    payloads = _structured_payloads(messages)
    if not payloads:
        return None
    payload = payloads[-1]

    content = payload.get("content")
    if content is not None and not isinstance(content, str):
        content = str(content)

    tool_calls: List[Dict[str, Any]] = []
    raw_calls = payload.get("tool_calls") or []
    if isinstance(raw_calls, list):
        for call in raw_calls:
            if isinstance(call, dict):
                converted = _to_openai_tool_call(call)
                if converted:
                    tool_calls.append(converted)

    return content or None, tool_calls
