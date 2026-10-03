"""multi_turn.py - conversational (multi-turn) testing.

RUN IT (fully offline - a hand-built transcript, no network):

    forjinn-eval run examples/multi_turn.py

A multi-turn transcript is a ``Conversation`` of ``Message``s. Two ways to get it:

A) **Hand-build it** (this example) - no network, full control:

       conv = Conversation()
       conv.add_human("…")
       conv.add_ai("…", tool_calls=[…], retrieval_contexts=[…])

B) **Drive a live agent** - its memory carries the history:

       run = ForjinnClient(HOST).predict_multi_turn(
           AGT, ["What is the refund policy?", "And exchanges?"])

Either way ``AgentRun.from_conversation(conv)`` turns the transcript into the
*same* ``AgentRun`` the single-turn catalog reads, so you can mix conversational
metrics (``TurnFaithfulness``, ``MultiTurnTopicAdherence``) with the deterministic
ones (``AllNodesFinished``, ``TokenBudget``) in one case.
"""
from __future__ import annotations

import os

from forjinn_eval import (
    AgentRun,
    Conversation,
    Message,
    MultiTurnTopicAdherence,
    OutputNotEmpty,
    ToolCall,
    TurnFaithfulness,
    TurnRelevancy,
    agent_test,
)

OFFLINE = os.environ.get("FORJINN_OFFLINE", "0") == "1"
# Use the default (permissive offline) judge when not running a live judge.
_judge = None
if not OFFLINE and os.environ.get("FORJINN_JUDGE_CHATFLOW"):
    from forjinn_eval import JudgeClient

    _judge = JudgeClient(
        judge_chatflow=os.environ["FORJINN_JUDGE_CHATFLOW"],
        base_url=os.environ.get("FORJINN_JUDGE_BASE_URL", os.environ.get("FORJINN_HOST", "https://172.16.34.7")),
        streaming=os.environ.get("FORJINN_JUDGE_STREAMING", "0") == "1",
    )
else:
    _judge = None


def _refund_thread() -> AgentRun:
    """A two-turn refund-policy conversation with per-turn retrieved context."""
    conv = Conversation()
    conv.add_human("What is our refund policy?")
    conv.append(Message(
        role="ai",
        content="Refunds are available within 30 days of purchase, provided the item is unused.",
        retrieval_contexts=["Policy: refunds within 30 days if unused."],
    ))
    conv.add_human("So I can return it in a month if I don't use it?")
    conv.append(Message(
        role="ai",
        content="Yes - within 30 days, and only if it is unused.",
        tool_calls=[ToolCall(name="search_db", arguments={"query": "refund window"})],
    ))
    conv.metadata["chatflow_id"] = "refund-thread"
    run = AgentRun.from_conversation(conv)
    return run


@agent_test("refund-thread", tags=["conversation", "multi-turn"])
def refund_thread():
    """A two-turn conversation stays on-topic and coherent."""
    run = _refund_thread()
    # A hand-built conversational AgentRun has no canvas-node trace, so we assert
    # on the text and (with a judge) the conversational metrics instead of
    # AllNodesFinished.
    evs = [OutputNotEmpty()]
    if _judge is not None:
        evs += [
            TurnFaithfulness(judge=_judge, threshold=0.5),
            TurnRelevancy(judge=_judge, threshold=0.4),
            MultiTurnTopicAdherence(reference_topics=["refunds", "return window"],
                                    judge=_judge, threshold=0.4),
        ]
    return run, evs


if __name__ == "__main__":
    from forjinn_eval import run_registered

    print(run_registered(suite_name="multi-turn").to_markdown())
