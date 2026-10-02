"""Helpers for the offline tests of the expanded metric catalog + web-scenarios.

Provides a :func:`canned_judge` factory (a MockJudge pre-seeded with one canned
response per call) and :func:`multi_turn_run` (builds an :class:`AgentRun` from
a list of (human, ai) turns, optionally with per-turn retrieval contexts and
tool calls).
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from forjinn_eval import AgentRun
from forjinn_eval.judge import MockJudge
from forjinn_eval.types import Conversation, Message, ToolCall


def canned_judge(*responses: Any) -> MockJudge:
    """A MockJudge whose successive calls return *responses[0], [1], ...*.

    Each response may be a plain dict/str (returned as-is) or a callable
    ``fn(question) -> dict``. When the queue is exhausted it answers ``{}``.
    """
    j = MockJudge()
    for r in responses:
        j.add(r)
    return j


def _tool(name: str, args: Optional[dict] = None) -> ToolCall:
    return ToolCall(name=name, arguments=args or {})


def multi_turn_run(
    turns: Sequence[Sequence],
    reference: Optional[str] = None,
    chatflow_id: str = "conv",
) -> AgentRun:
    """Build a multi-turn :class:`AgentRun` from ``turns``.

    ``turns`` is a list where each entry is one of:
      * ``"hi"``                        - a human-only prompt
      * ``("hi", "hello")``             - human prompt + ai reply
      * ``("hi", "hello", ["ctx"], [tool_calls])``  - with retrieval contexts
        and tool calls on the ai turn (the last two are optional).
    """
    conv = Conversation()
    for t in turns:
        if isinstance(t, str):
            conv.add_human(t)
            continue
        human = str(t[0])
        ai = str(t[1]) if len(t) > 1 and t[1] else ""
        ctx = list(t[2]) if len(t) > 2 and t[2] else []
        tools = list(t[3]) if len(t) > 3 and t[3] else []
        conv.add_human(human)
        conv.messages.append(Message(role="ai", content=ai,
                                     retrieval_contexts=ctx,
                                     tool_calls=[_tool(x) if isinstance(x, str) else x
                                                 for x in tools]))
    if reference is not None:
        conv.metadata["reference"] = reference
    conv.metadata["chatflow_id"] = chatflow_id
    return AgentRun.from_conversation(conv)


def single_turn_run(question: str, text: str, reference: Optional[str] = None,
                    contexts: Optional[List[str]] = None) -> AgentRun:
    run = AgentRun(chatflow_id="single", question=question, text=text)
    if contexts:
        run.raw["retrieved_contexts"] = contexts
    if reference is not None:
        run.raw["reference"] = reference
    return run


__all__ = ["canned_judge", "multi_turn_run", "single_turn_run", "_tool"]
