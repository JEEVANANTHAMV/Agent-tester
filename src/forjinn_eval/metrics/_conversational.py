"""Shared helpers for the multi-turn / conversational metric family.

Every conversational metric operates on a :class:`~forjinn_eval.Types.Conversation`
extracted off the :class:`~forjinn_eval.capture.AgentRun` (which carries the full
transcript when built from :meth:`AgentRun.from_conversation`). These helpers
window and render the transcript for the per-turn rubrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..capture import AgentRun, Conversation


@dataclass
class Turn:
    """A single (human, ai) exchange in a conversation.

    ``idx`` is the 1-based turn number. ``human`` may be ``""`` for a seeded
    assistant-only start; ``ai`` is the agent's reply.
    """

    idx: int
    human: str = ""
    ai: str = ""
    tool_calls: List[Any] = field(default_factory=list)
    retrieval_contexts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def conversation_of(run: AgentRun) -> Conversation:
    return run.conversation


def to_turns(conversation: Conversation) -> List[Turn]:
    """Pair human->ai messages into :class:`Turn` objects.

    Consecutive human messages (before an ai reply) are joined with a newline;
    an ai message with no preceding human still yields a turn (``human=""``).
    """
    turns: List[Turn] = []
    pending: List[str] = []
    for m in conversation.turns:
        if m.is_human:
            if m.content:
                pending.append(m.content)
        elif m.is_ai:
            turns.append(Turn(
                idx=len(turns) + 1,
                human="\n".join(pending),
                ai=m.content,
                tool_calls=list(m.tool_calls),
                retrieval_contexts=list(m.retrieval_contexts),
                metadata=dict(m.metadata),
            ))
            pending = []
    if pending:
        turns.append(Turn(idx=len(turns) + 1, human="\n".join(pending), ai="", tool_calls=[], retrieval_contexts=[]))
    return turns


def turns_of(run: AgentRun) -> List[Turn]:
    return to_turns(run.conversation)


def render_history(turns: List[Turn], upto: int) -> str:
    """Render turns ``1..upto`` (inclusive) as a transcript block.

    The human input of turn ``upto`` is shown as the *current* question; the
    preceding human/ai pairs are the history context.
    """
    if not turns:
        return ""
    upto = max(1, min(upto, len(turns)))
    parts: List[str] = []
    for t in turns[:upto]:
        if t.human:
            parts.append(f"Human: {t.human}")
        if t.ai:
            parts.append(f"AI: {t.ai}")
    return "\n".join(parts) or "\n".join(f"AI: {t.ai}" for t in turns[:upto])


def context_for(conversation: Conversation, upto: int) -> List[str]:
    return conversation.retrieval_contexts(upto=upto)


def tool_names_of(tc_list) -> List[str]:
    return [str(t.name) for t in tc_list if getattr(t, "name", None)]


__all__ = ["Turn", "context_for", "conversation_of", "render_history", "to_turns", "tool_names_of", "turns_of"]
