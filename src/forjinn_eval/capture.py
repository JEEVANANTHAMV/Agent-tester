"""Execution-capture data model for Forjinn agent runs.

Everything is derived from the live Forjinn API on the builder host (e.g.
172.16.34.7). Two source shapes exist and both are normalised into ``AgentRun``:

* **Non-streaming** ``POST /api/v1/prediction/<chatflowId>`` -> a single JSON
  ``AgentResult`` (top-level: ``text``, ``question``, ``chatId``, ``chatMessageId``,
  ``executionId``, ``agentFlowExecutedData``, ``usageMetadata``, ``calledTools``,
  ``sessionId``).
* **Streaming** (same call + ``"streaming": true``) -> SSE frames
  ``message:`` / ``data:{"event":..., "data":...}`` with events:
  ``agentFlowEvent``, ``nextAgentFlow``, ``agentFlowExecutedData``, ``token``,
  ``calledTools``, ``usageMetadata``, ``metadata``, ``end`` ``= "[DONE]"``.

``AgentRun`` is the single artefact the whole library evaluates: it always
carries the final text, the ordered node-by-node trace, tool calls, token/timing
usage, and the raw SSE event sequence when streamed.
"""
from __future__ import annotations

import codecs
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .types import (
    EV_CALLED_TOOLS,
    EV_END,
    EV_FLOW_EVENT,
    EV_FLOW_EXECUTED_DATA,
    EV_METADATA,
    EV_NEXT_AGENT_FLOW,
    EV_TOKEN,
    EV_USAGE,
    STATUS_FINISHED,
    ToolCall,
    TimeMetadata,
    UsageMetadata,
)


def _status_of(raw: str) -> str:
    s = (raw or "").upper()
    if s in ("FINISHED", "SUCCESS", "COMPLETE", "COMPLETED"):
        return STATUS_FINISHED
    if s in ("FAILED", "ERROR", "ERRORS", "EXCEPTION"):
        return "FAILED"
    if s in ("INPROGRESS", "IN_PROGRESS", "RUNNING", "PENDING"):
        return "INPROGRESS"
    return s or "UNKNOWN"


@dataclass
class Node:
    """One canvas node as it appears in ``agentFlowExecutedData``.

    ``input``/``output`` are kept as free-form dicts because the fields differ per
    node type (start / agent / retrieval / tool / ...). Only the handful of fields
    common to the *agent* (LLM+tools) node are hoisted onto explicit attributes so
    evaluators can read them without re-parsing.
    """

    node_id: str
    node_label: str
    name: str  # data.name e.g. "startAgentflow", "agentAgentflow"
    status: str
    input: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    previous_node_ids: List[str] = field(default_factory=list)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    model_name: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    called_tools: List[ToolCall] = field(default_factory=list)
    available_tools: List[Dict[str, Any]] = field(default_factory=list)
    usage: UsageMetadata = field(default_factory=UsageMetadata)
    time: TimeMetadata = field(default_factory=TimeMetadata)

    @property
    def is_agent(self) -> bool:
        return self.name.startswith("agent") or self.name.startswith("agentAgentflow")

    @property
    def is_start(self) -> bool:
        return self.name.startswith("start")

    @property
    def content(self) -> str:
        out = self.output.get("content")
        if out is None:
            out = self.output.get("text")
        return out if out is not None else ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        data = d.get("data") or {}
        output = dict(data.get("output") or {})
        inp = dict(data.get("input") or {})
        node = cls(
            node_id=d.get("nodeId") or data.get("id") or "",
            node_label=d.get("nodeLabel") or data.get("name") or "",
            name=data.get("name") or d.get("nodeId") or "",
            status=_status_of(d.get("status") or data.get("status") or ""),
            input=inp,
            output=output,
            state=dict(data.get("state") or {}),
            previous_node_ids=list(d.get("previousNodeIds") or []),
            chat_history=list(data.get("chatHistory") or []),
        )
        # Agent-node hoisted fields (safe for any node type: .get() -> default)
        mcfg = inp.get("agentModelConfig") or {}
        node.model_name = mcfg.get("modelName") or inp.get("agentModel")
        node.messages = list(inp.get("messages") or [])
        node.called_tools = [ToolCall.from_dict(t) for t in (output.get("calledTools") or [])]
        node.available_tools = list(output.get("availableTools") or [])
        node.usage = UsageMetadata.from_dict(output.get("usageMetadata"))
        node.time = TimeMetadata.from_dict(output.get("timeMetadata"))
        return node

    def node_type(self) -> str:
        prefix = self.name.split("Agentflow")[0] or self.name
        return prefix


@dataclass
class StreamEvent:
    """One decoded SSE data frame."""

    event: str
    data: Any
    index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"event": self.event, "data": self.data}


@dataclass
class AgentRun:
    """The single artefact an evaluation operates on.

    Populated by :meth:`from_nonstream` or :meth:`from_stream`.
    """

    # --- identity ----------------------------------------------------------------
    chatflow_id: str
    question: str = ""
    chat_id: Optional[str] = None
    chat_message_id: Optional[str] = None
    execution_id: Optional[str] = None
    session_id: Optional[str] = None
    # --- core outputs ----------------------------------------------------------
    text: str = ""
    nodes: List[Node] = field(default_factory=list)
    called_tools: List[ToolCall] = field(default_factory=list)
    usage: UsageMetadata = field(default_factory=UsageMetadata)
    # --- provenance ----------------------------------------------------------
    streaming: bool = False
    stream_events: List[StreamEvent] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    flow_status: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_nonstream(cls, chatflow_id: str, payload: Dict[str, Any]) -> "AgentRun":
        run = cls(
            chatflow_id=chatflow_id,
            streaming=False,
            raw=payload,
            question=payload.get("question") or "",
            chat_id=payload.get("chatId"),
            chat_message_id=payload.get("chatMessageId"),
            execution_id=payload.get("executionId"),
            session_id=payload.get("sessionId"),
            text=payload.get("text") or "",
            called_tools=[ToolCall.from_dict(t) for t in (payload.get("calledTools") or [])],
            usage=UsageMetadata.from_dict(payload.get("usageMetadata")),
        )
        for nd in payload.get("agentFlowExecutedData") or []:
            run.nodes.append(Node.from_dict(nd))
        # Fallback: some payloads only carry the text on the agent node.
        if not run.text:
            for nd in run.nodes:
                if nd.content:
                    run.text = nd.content
        run._roll_up_usage()
        return run

    @classmethod
    def from_stream(
        cls,
        chatflow_id: str,
        events: List[StreamEvent],
    ) -> "AgentRun":
        run = cls(chatflow_id=chatflow_id, streaming=True, stream_events=list(events))
        last_executed: List[Dict[str, Any]] = []
        for ev in events:
            if ev.event == EV_FLOW_EXECUTED_DATA and isinstance(ev.data, list):
                last_executed = ev.data
            elif ev.event == EV_TOKEN and isinstance(ev.data, str):
                run.tokens.append(ev.data)
            elif ev.event == EV_CALLED_TOOLS and isinstance(ev.data, list):
                run.called_tools = [ToolCall.from_dict(t) for t in ev.data]
            elif ev.event == EV_USAGE and isinstance(ev.data, dict):
                run.usage = UsageMetadata.from_dict(ev.data)
            elif ev.event == EV_METADATA and isinstance(ev.data, dict):
                run.chat_id = ev.data.get("chatId", run.chat_id)
                run.chat_message_id = ev.data.get("chatMessageId", run.chat_message_id)
                run.question = ev.data.get("question", run.question)
                run.session_id = ev.data.get("sessionId", run.session_id)
            elif ev.event == EV_END and ev.data == "[DONE]":
                run.flow_status = "FINISHED"
            elif ev.event == EV_FLOW_EVENT and isinstance(ev.data, str):
                run.flow_status = _status_of(ev.data)
        for nd in last_executed:
            run.nodes.append(Node.from_dict(nd))
        if not run.text:
            run.text = "".join(run.tokens)
        # The final executed-data payload's agent node carries content + usage.
        for nd in run.nodes:
            if nd.content and not run.text:
                run.text = nd.content
            if nd.usage.total_tokens and not run.usage.total_tokens:
                run.usage = nd.usage
        run._roll_up_usage()
        return run

    def _roll_up_usage(self) -> None:
        if not self.usage.total_tokens and self.nodes:
            acc = UsageMetadata()
            for nd in self.nodes:
                acc.input_tokens += nd.usage.input_tokens
                acc.output_tokens += nd.usage.output_tokens
                acc.tool_call_tokens += nd.usage.tool_call_tokens
            acc.total_tokens = acc.input_tokens + acc.output_tokens + acc.tool_call_tokens
            self.usage = acc

    # ---- accessors -----------------------------------------------------------
    @property
    def agent_node(self) -> Optional[Node]:
        for nd in self.nodes:
            if nd.is_agent:
                return nd
        return None

    @property
    def final_agent_node(self) -> Optional[Node]:
        agents = [nd for nd in self.nodes if nd.is_agent]
        return agents[-1] if agents else None

    @property
    def all_tool_calls(self) -> List[ToolCall]:
        if self.called_tools:
            return self.called_tools
        out: List[ToolCall] = []
        for nd in self.nodes:
            out.extend(nd.called_tools)
        return out

    @property
    def tool_names(self) -> List[str]:
        return [t.name for t in self.all_tool_calls if t.name]

    @property
    def node_names(self) -> List[str]:
        return [nd.name for nd in self.nodes]

    def nodes_by_type(self, prefix: str) -> List[Node]:
        return [nd for nd in self.nodes if nd.name.startswith(prefix)]

    def agent_nodes(self) -> List[Node]:
        return [nd for nd in self.nodes if nd.is_agent]

    def system_prompt(self) -> str:
        node = self.final_agent_node
        if not node:
            return ""
        for m in node.messages:
            if m.get("role") == "system":
                return m.get("content") or ""
        return ""

    @property
    def all_nodes_finished(self) -> bool:
        return bool(self.nodes) and all(nd.status == STATUS_FINISHED for nd in self.nodes)

    # ---- evaluation-context accessors --------------------------------
    # These expose the fields the metric catalog needs. Forjinn carries them in
    # different places depending on agent type, so this centralises the lookup:
    #  * retrieval contexts : node `output.context` / `data.metadata.context`
    #    (knowledge-base / document nodes) or `raw.retrieved_contexts`
    #  * reference (ground truth) : `raw.reference`
    #  * agent config      : `agentModelConfig` from the final agent node
    def retrieved_context(self) -> List[str]:
        """Context chunks the agent saw (knowledge bases / document stores)."""
        raw = self.raw or {}
        explicit = raw.get("retrieved_contexts")
        if isinstance(explicit, list) and explicit:
            return [str(c) for c in explicit]
        out: List[str] = []
        for nd in self.nodes:
            ctx = nd.output.get("context") or nd.input.get("context")
            if isinstance(ctx, str) and ctx.strip():
                out.append(ctx)
            elif isinstance(ctx, list) and ctx:
                out.extend(str(c) for c in ctx)
            meta = nd.state.get("context") or (nd.output.get("metadata") or {}).get("context")
            if isinstance(meta, str) and meta.strip():
                out.append(meta)
            elif isinstance(meta, list) and meta:
                out.extend(str(c) for c in meta)
        return out

    @property
    def reference(self) -> Optional[str]:
        """Ground-truth answer, if the caller attached one via run.raw."""
        r = (self.raw or {}).get("reference")
        return str(r) if r not in (None, "") else None

    def agent_config(self) -> Dict[str, Any]:
        node = self.final_agent_node
        if not node:
            return {}
        return dict(node.input.get("agentModelConfig") or {})

    @property
    def allow_image_uploads(self) -> bool:
        return bool(self.agent_config().get("allowImageUploads"))

    @property
    def available_tool_names(self) -> List[str]:
        node = self.final_agent_node
        if not node:
            return []
        return [str(t.get("name")) for t in node.available_tools if t.get("name")]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chatflow_id": self.chatflow_id,
            "question": self.question,
            "chat_id": self.chat_id,
            "session_id": self.session_id,
            "text": self.text,
            "streaming": self.streaming,
            "flow_status": self.flow_status,
            "usage": self.usage.to_dict(),
            "called_tools": [t.to_dict() for t in self.all_tool_calls],
            "nodes": [
                {
                    "node_id": nd.node_id,
                    "node_label": nd.node_label,
                    "name": nd.name,
                    "status": nd.status,
                    "model_name": nd.model_name,
                    "previous_node_ids": nd.previous_node_ids,
                    "usage": nd.usage.to_dict(),
                    "time": nd.time.to_dict(),
                    "content": nd.content,
                    "called_tools": [t.to_dict() for t in nd.called_tools],
                    "available_tools": nd.available_tools,
                }
                for nd in self.nodes
            ],
            "stream_event_count": len(self.stream_events),
        }


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------
def _event_from_line(line: str) -> Optional[StreamEvent]:
    """Convert a single ``data:`` SSE line to a StreamEvent (or None)."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return StreamEvent(event=parsed.get("event", "event"), data=parsed.get("data"))


def iter_sse_events(lines: Iterator[Any]) -> Iterator[StreamEvent]:
    """Yield :class:`StreamEvent`s from an iterator of bytes/str lines.

    Works for ``requests.Response.iter_lines()`` (bytes) or any string-lines
    iterator. ``message:`` lines and blank separators are ignored; only
    ``data:``-prefixed lines that parse as JSON with ``event``/``data`` are kept.
    ``end`` events carry the literal ``"[DONE]"`` marker.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for raw in lines:
        if raw is None:
            continue
        if isinstance(raw, (bytes, bytearray)):
            raw = decoder.decode(bytes(raw))
        elif isinstance(raw, str) and raw and raw[0] == "\ufeff":
            raw = raw[1:]
        for line in raw.splitlines():
            ev = _event_from_line(line)
            if ev is not None:
                yield ev


def parse_sse_text(text: str) -> List[StreamEvent]:
    """Parse a saved SSE body (the kind captured to a file) into StreamEvents."""
    events = [ev for ev in iter_sse_events([text]) ]
    for i, ev in enumerate(events):
        ev.index = i
    return events
