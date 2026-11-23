"""Provider-agnostic conversation types for tool use.

The agent loop speaks only these types. The Anthropic adapter translates them
to and from the SDK, and they serialise to plain JSON so a conversation can be
stored in Postgres and resumed days later.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


Block = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: list[Block]

    def to_json(self) -> dict[str, Any]:
        return {"role": self.role, "content": [_block_to_json(b) for b in self.content]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Message":
        return cls(role=data["role"], content=[_block_from_json(b) for b in data["content"]])

    @classmethod
    def user_text(cls, text: str) -> "Message":
        return cls("user", [TextBlock(text)])


def _block_to_json(b: Block) -> dict[str, Any]:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return {
        "type": "tool_result",
        "tool_use_id": b.tool_use_id,
        "content": b.content,
        "is_error": b.is_error,
    }


def _block_from_json(d: dict[str, Any]) -> Block:
    if d["type"] == "text":
        return TextBlock(d["text"])
    if d["type"] == "tool_use":
        return ToolUseBlock(d["id"], d["name"], dict(d["input"]))
    if d["type"] == "tool_result":
        return ToolResultBlock(d["tool_use_id"], d["content"], bool(d.get("is_error", False)))
    raise ValueError(f"unknown block type: {d['type']}")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call. ``input_schema`` is JSON Schema."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """One model response in a tool-using conversation."""

    content: list[Block]
    stop_reason: str | None
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    prompt_name: str
    prompt_version: str
    extra: dict[str, Any] = field(default_factory=dict)
    # Prompt-cache tokens, billed apart from input_tokens (which excludes them).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock)).strip()

    def as_message(self) -> Message:
        return Message("assistant", list(self.content))
