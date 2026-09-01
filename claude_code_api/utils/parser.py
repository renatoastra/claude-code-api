"""JSONL parser for Claude Code output."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

import structlog

from claude_code_api.models.claude import ClaudeMessage, ClaudeToolResult, ClaudeToolUse
from claude_code_api.utils.time import utc_now

logger = structlog.get_logger()


def _normalize_text_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "text" in value:
        return str(value.get("text", ""))
    return str(value)


def _text_from_mapping(payload: Dict[str, Any]) -> Optional[str]:
    if "text" in payload:
        return _normalize_text_value(payload.get("text"))
    if "content" in payload:
        return _normalize_text_value(payload.get("content"))
    return None


def _text_from_part(part: Any) -> Optional[str]:
    if isinstance(part, dict):
        if part.get("type") == "text" and "text" in part:
            return _normalize_text_value(part.get("text"))
        return _text_from_mapping(part)
    if isinstance(part, str):
        return part
    return None


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class ClaudeOutputParser:
    """Parser for Claude Code JSONL output."""

    def __init__(self):
        self.session_id: Optional[str] = None
        self.model: Optional[str] = None
        self.input_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0
        self.message_count = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.output_tokens
        )

    def parse_line(self, line: str) -> Optional[ClaudeMessage]:
        """Parse a single JSONL line."""
        if not line.strip():
            return None

        try:
            data = json.loads(line.strip())
            message = ClaudeMessage(**data)
            return self.parse_message(message)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSONL line", line=line[:100], error=str(e))
            return None
        except Exception as e:
            logger.error("Error parsing message", line=line[:100], error=str(e))
            return None

    def parse_message(self, message: ClaudeMessage) -> Optional[ClaudeMessage]:
        """Parse a ClaudeMessage and update metrics."""
        if not message:
            return None

        # Extract session info on first message
        if message.session_id and not self.session_id:
            self.session_id = message.session_id

        if message.model and not self.model:
            self.model = message.model

        # Only the `result` line carries top-level usage, and it is already the
        # aggregate of the run. With prompt caching nearly every prompt token is
        # reported under cache_creation/cache_read, not input_tokens.
        if message.usage:
            self.input_tokens += _as_int(message.usage.get("input_tokens"))
            self.cache_creation_input_tokens += _as_int(
                message.usage.get("cache_creation_input_tokens")
            )
            self.cache_read_input_tokens += _as_int(
                message.usage.get("cache_read_input_tokens")
            )
            self.output_tokens += _as_int(message.usage.get("output_tokens"))

        cost = message.total_cost_usd
        if cost is None:
            cost = message.cost_usd
        if cost:
            self.total_cost += cost

        if message.type in ["user", "assistant"]:
            self.message_count += 1

        return message

    def parse_stream(self, lines: List[str]) -> Generator[ClaudeMessage, None, None]:
        """Parse multiple JSONL lines."""
        for line in lines:
            message = self.parse_line(line)
            if message:
                yield message

    def extract_text_content(self, message: ClaudeMessage) -> str:
        """Extract text content from a message."""
        if not message.message:
            return ""

        content = message.message.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return _text_from_mapping(content) or ""
        if isinstance(content, list):
            text_parts = []
            for part in content:
                text = _text_from_part(part)
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts)
        return str(content)

    def extract_tool_uses(self, message: ClaudeMessage) -> List[ClaudeToolUse]:
        """Extract tool uses from a message."""
        if not message.message:
            return []

        content = message.message.get("content", [])
        if not isinstance(content, list):
            return []

        tool_uses = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                try:
                    tool_use = ClaudeToolUse(
                        id=part.get("id", ""),
                        name=part.get("name", ""),
                        input=part.get("input", {}),
                    )
                    tool_uses.append(tool_use)
                except Exception as e:
                    logger.warning("Failed to parse tool use", part=part, error=str(e))

        return tool_uses

    def extract_tool_results(self, message: ClaudeMessage) -> List[ClaudeToolResult]:
        """Extract tool results from a message."""
        if not message.message:
            return []

        content = message.message.get("content", [])
        if not isinstance(content, list):
            return []

        tool_results = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_result":
                try:
                    tool_result = ClaudeToolResult(
                        tool_use_id=part.get("tool_use_id", ""),
                        content=part.get("content", ""),
                        is_error=part.get("is_error", False),
                    )
                    tool_results.append(tool_result)
                except Exception as e:
                    logger.warning(
                        "Failed to parse tool result", part=part, error=str(e)
                    )

        return tool_results

    def is_system_message(self, message: ClaudeMessage) -> bool:
        """Check if message is a system message."""
        return message.type == "system"

    def is_user_message(self, message: ClaudeMessage) -> bool:
        """Check if message is from user."""
        return message.type == "user" or (
            message.message and message.message.get("role") == "user"
        )

    def is_assistant_message(self, message: ClaudeMessage) -> bool:
        """Check if message is from assistant."""
        return message.type == "assistant" or (
            message.message and message.message.get("role") == "assistant"
        )

    def is_final_message(self, message: ClaudeMessage) -> bool:
        """Check if this is a final result message."""
        return message.type == "result"

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of parsed session."""
        return {
            "session_id": self.session_id,
            "model": self.model,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "message_count": self.message_count,
        }

    def reset(self):
        """Reset parser state."""
        self.session_id = None
        self.model = None
        self.input_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0
        self.message_count = 0


class OpenAIConverter:
    """Converts Claude messages to OpenAI format."""

    @staticmethod
    def claude_message_to_openai(message: ClaudeMessage) -> Optional[Dict[str, Any]]:
        """Convert Claude message to OpenAI chat format."""
        parser = ClaudeOutputParser()
        if parser.is_system_message(message):
            return {"role": "system", "content": parser.extract_text_content(message)}

        if parser.is_user_message(message):
            return {"role": "user", "content": parser.extract_text_content(message)}

        if parser.is_assistant_message(message):
            content = parser.extract_text_content(message)
            if content:
                return {"role": "assistant", "content": content}

        return None

    @staticmethod
    def claude_stream_to_openai_chunk(
        message: ClaudeMessage, chunk_id: str, model: str, created: int
    ) -> Optional[Dict[str, Any]]:
        """Convert Claude stream message to OpenAI chunk format."""
        parser = ClaudeOutputParser()
        if not parser.is_assistant_message(message):
            return None

        content = parser.extract_text_content(message)
        if not content:
            return None

        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }

    @staticmethod
    def create_final_chunk(
        chunk_id: str, model: str, created: int, finish_reason: str = "stop"
    ) -> Dict[str, Any]:
        """Create final chunk to end streaming."""
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }

    @staticmethod
    def calculate_usage(parser: ClaudeOutputParser) -> Dict[str, Any]:
        """OpenAI usage block from the CLI result usage.

        prompt_tokens is everything the model read (fresh + cache write + cache
        read); `prompt_tokens_details.cached_tokens` mirrors OpenAI's field for
        the cache-read share, and the two extra keys expose what OpenAI has no
        slot for.
        """
        prompt_tokens = (
            parser.input_tokens
            + parser.cache_creation_input_tokens
            + parser.cache_read_input_tokens
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": parser.output_tokens,
            "total_tokens": prompt_tokens + parser.output_tokens,
            "prompt_tokens_details": {"cached_tokens": parser.cache_read_input_tokens},
            "cache_creation_input_tokens": parser.cache_creation_input_tokens,
            "total_cost_usd": parser.total_cost,
        }


class MessageAggregator:
    """Aggregates streaming messages into complete responses."""

    def __init__(self):
        self.messages: List[ClaudeMessage] = []
        self.current_assistant_content = ""
        self.parser = ClaudeOutputParser()

    def add_message(self, message: Any):
        """Add message to aggregator."""
        normalized = normalize_claude_message(message)
        if not normalized:
            return
        self.messages.append(normalized)
        self.parser.parse_message(normalized)

        # Aggregate assistant content for complete response
        if self.parser.is_assistant_message(normalized):
            content = self.parser.extract_text_content(normalized)
            if content:
                self.current_assistant_content += content

    def get_complete_response(self) -> str:
        """Get complete aggregated response."""
        return self.current_assistant_content

    def get_messages(self) -> List[ClaudeMessage]:
        """Get all messages."""
        return self.messages

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        return self.parser.get_session_summary()

    def clear(self):
        """Clear aggregator state."""
        self.messages.clear()
        self.current_assistant_content = ""
        self.parser.reset()


def sanitize_content(content: str) -> str:
    """Sanitize content for safe transmission."""
    if not content:
        return ""

    # Remove null bytes
    content = content.replace("\x00", "")

    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Ensure valid UTF-8
    try:
        content.encode("utf-8")
    except UnicodeEncodeError:
        # Replace invalid characters
        content = content.encode("utf-8", errors="replace").decode("utf-8")

    return content


def extract_error_from_message(message: ClaudeMessage) -> Optional[str]:
    """Extract error information from Claude message."""
    if message.error:
        return message.error

    if message.type == "result" and not message.result:
        return "Execution completed without result"

    # Check for error in tool results
    tool_results = ClaudeOutputParser().extract_tool_results(message)
    for result in tool_results:
        if result.is_error:
            return str(result.content)

    return None


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count."""
    # Very rough estimation: ~4 characters per token
    return max(1, len(text) // 4)


def normalize_claude_message(raw: Any) -> Optional[ClaudeMessage]:
    """Normalize a raw Claude output object into a ClaudeMessage."""
    if isinstance(raw, ClaudeMessage):
        return raw
    if isinstance(raw, dict):
        try:
            return ClaudeMessage(**raw)
        except (TypeError, ValueError) as e:
            logger.warning("Failed to normalize Claude message", error=str(e))
            return None
    return None


def tool_use_to_openai_call(tool_use: ClaudeToolUse) -> Dict[str, Any]:
    """Convert a Claude tool use to an OpenAI tool call object."""
    tool_id = tool_use.id or f"call_{uuid.uuid4().hex}"
    try:
        arguments = json.dumps(
            tool_use.input or {}, separators=(",", ":"), ensure_ascii=False
        )
    except TypeError:
        arguments = json.dumps(
            {"input": str(tool_use.input)}, separators=(",", ":"), ensure_ascii=False
        )
    return {
        "id": tool_id,
        "type": "function",
        "function": {"name": tool_use.name, "arguments": arguments},
    }


def format_timestamp(timestamp: Optional[str]) -> str:
    """Format timestamp for display."""
    if not timestamp:
        return utc_now().isoformat()

    try:
        # Try parsing ISO format
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.isoformat()
    except (ValueError, TypeError):
        return timestamp
