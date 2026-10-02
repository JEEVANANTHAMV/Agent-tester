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
