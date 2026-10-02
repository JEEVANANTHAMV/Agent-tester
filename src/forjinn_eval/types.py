"""Shared, dependency-light data primitives for forjinn_eval.

These types intentionally avoid heavy dependencies (no pydantic) so the library
installs and runs in the smallest possible footprint. Everything is a plain
``dataclass`` that round-trips through ``asdict`` / from-dict helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ForjinnError(Exception):
    """Raised for client-level problems (transport, decode, bad status)."""


class NodeStatusError(ForjinnError):
    """Raised when a captured node failed or the run did not reach a terminal state."""


@dataclass
class ToolCall:
    """A single tool invocation observed on an agent node.

    Forjinn's ``output.calledTools`` entries are free-form; we normalise the most
    common shapes (``{name, arguments/input, output}``) while preserving the raw
    dict so nothing is lost.
    """

    name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any) -> "ToolCall":
        if not isinstance(d, dict):
            return cls(raw=d if isinstance(d, dict) else {}, )
        name = d.get("name") or d.get("tool") or d.get("toolName")
        args = d.get("arguments") or d.get("args") or d.get("input") or d.get("parameters") or {}
        out = d.get("output") if "output" in d else d.get("result")
        return cls(name=name, arguments=args if isinstance(args, dict) else {}, output=out, raw=d)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "output": self.output, "raw": self.raw}

    # Equality over (name, arguments) so set/F1 tool metrics are stable.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return NotImplemented
        return self.name == other.name and self.arguments == other.arguments

    def __hash__(self) -> int:
        return hash((self.name, _hashable(self.arguments)))


def _hashable(x: Any) -> Any:
    if isinstance(x, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_hashable(v) for v in x)
    return x


@dataclass
class UsageMetadata:
    """Token accounting reported by the Forjinn agent node / top-level result."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_call_tokens: int = 0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "UsageMetadata":
        d = d or {}
        inp = int(d.get("input_tokens") or d.get("prompt_tokens") or 0)
        out = int(d.get("output_tokens") or d.get("completion_tokens") or 0)
        total = d.get("total_tokens")
        total = int(total) if total not in (None, 0) else (inp + out)
        return cls(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=int(total),
            tool_call_tokens=int(d.get("tool_call_tokens") or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "tool_call_tokens": self.tool_call_tokens,
        }


@dataclass
class TimeMetadata:
    start_ms: Optional[float] = None
    end_ms: Optional[float] = None
    delta_ms: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.delta_ms is not None:
            return self.delta_ms / 1000.0
        if self.start_ms and self.end_ms is not None:
            return (self.end_ms - self.start_ms) / 1000.0
        return None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "TimeMetadata":
        d = d or {}
        return cls(
            start_ms=d.get("start"),
            end_ms=d.get("end"),
            delta_ms=d.get("delta"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start_ms, "end": self.end_ms, "delta": self.delta_ms}


# ---- Multi-turn conversation primitives -----------------------------------
@dataclass
class Message:
    """One turn in a multi-turn conversation.

    ``role`` is one of ``human`` (user), ``ai`` / ``assistant`` (agent) or
    ``system``. ``tool_calls`` holds the :class:`ToolCall`s the agent issued
    during an ``ai`` turn (Forjinn surfaces these on the agent node's
    ``output.calledTools``).
    """

    role: str = "human"
    content: str = ""
    tool_calls: List["ToolCall"] = field(default_factory=list)
    retrieval_contexts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Any) -> "Message":
        if not isinstance(d, dict):
            return cls(role=str(d) if d else "", content=str(d))
        role = str(d.get("role", "human")).lower()
        return cls(
            role=role,
            content=str(d.get("content", "")),
            tool_calls=[ToolCall.from_dict(t) for t in (d.get("tool_calls") or d.get("calledTools") or [])],
            retrieval_contexts=[str(c) for c in (d.get("retrieval_contexts") or d.get("retrieved_contexts") or [])],
            metadata=dict(d.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "retrieval_contexts": list(self.retrieval_contexts),
            "metadata": self.metadata,
        }

    @property
    def is_human(self) -> bool:
        return self.role in {"human", "user"}

    @property
    def is_ai(self) -> bool:
        return self.role in {"ai", "assistant"}


@dataclass
class Conversation:
    """An ordered multi-turn transcript.

    Built either from a list of :class:`Message` (or dicts) or from one or more
    :class:`~forjinn_eval.capture.AgentRun` objects (each run contributes its
    question + final answer + tool calls).
    """

    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_messages(cls, messages) -> "Conversation":
        conv = cls()
        for m in messages or []:
            conv.append(m)
        return conv

    @classmethod
    def from_runs(cls, *runs) -> "Conversation":
        from .capture import AgentRun  # local import avoids a cycle during package init

        conv = cls()
        for r in runs:
            if not isinstance(r, AgentRun):
                continue
            if r.question:
                conv.messages.append(Message(role="human", content=r.question))
            tool_calls = list(r.all_tool_calls)
            if tool_calls or r.text:
                conv.messages.append(Message(role="ai", content=r.text, tool_calls=tool_calls))
        return conv

    def append(self, message: "Message | Dict[str, Any]") -> "Message":
        m = message if isinstance(message, Message) else Message.from_dict(message)
        self.messages.append(m)
        return m

    def add_human(self, content: str) -> "Message":
        return self.append(Message(role="human", content=content))

    def add_ai(self, content: str, tool_calls=None) -> "Message":
        return self.append(Message(role="ai", content=content, tool_calls=list(tool_calls or [])))

    @property
    def turns(self) -> List[Message]:
        return list(self.messages)

    @property
    def human_turns(self) -> List[Message]:
        return [m for m in self.messages if m.is_human]

    @property
    def ai_turns(self) -> List[Message]:
        return [m for m in self.messages if m.is_ai]

    @property
    def tool_calls(self) -> List[ToolCall]:
        out: List[ToolCall] = []
        for m in self.messages:
            out.extend(m.tool_calls)
        return out

    def retrieval_contexts(self, upto: Optional[int] = None) -> List[str]:
        """Cumulative retrieved contexts across turns (``None`` = all)."""
        out: List[str] = []
        for m in self.messages[:upto]:
            out.extend(m.retrieval_contexts)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"messages": [m.to_dict() for m in self.messages], "metadata": self.metadata}


# ---- Canonical Forjinn node status values ---------------------------------
STATUS_INPROGRESS = "INPROGRESS"
STATUS_FINISHED = "FINISHED"
STATUS_FAILED = "FAILED"
TERMINAL_STATUSES = {STATUS_FINISHED, STATUS_FAILED}

# ---- Canonical SSE event names --------------------------------------------
EV_FLOW_EVENT = "agentFlowEvent"
EV_NEXT_AGENT_FLOW = "nextAgentFlow"
EV_FLOW_EXECUTED_DATA = "agentFlowExecutedData"
EV_TOKEN = "token"
EV_CALLED_TOOLS = "calledTools"
EV_USAGE = "usageMetadata"
EV_METADATA = "metadata"
EV_END = "end"
SSE_END_MARK = "[DONE]"
