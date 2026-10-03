"""Deterministic agent-loop detector (deepeval ``AgentLoopDetectionMetric``).

Detects infinite-loop / cyclical failure modes in an :class:`AgentRun` trace
using three independent, LLM-free sub-signals:

1. **Tool-call repetition** - the same ``(name, arguments)`` pair invoked
   ``>= repetition_threshold`` times.
2. **Reasoning stagnation** - consecutive agent outputs that are near-identical
   (token-overlap similarity ``>= similarity_threshold``).
3. **Call-graph cycles** - a node reaching ``FAILED`` status or the start node
   being re-entered (the Forjinn trace re-emits ``nextAgentFlow``/``
   agentFlowExecutedData`` per node; a repeating node label is a cycle).

The final score is in ``[0, 1]`` where ``1.0`` = clean, ``0.0`` = severe loop.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Set

from ..capture import AgentRun
from ._common import _cr, _nan_or_skip
from .base import CheckResult, Evaluator


def _bigram_set(text: str) -> Set[frozenset]:
    toks = [w.lower() for w in (text or "").split()]
    if len(toks) < 2:
        return {frozenset({t}) for t in toks}
    return {frozenset((toks[i], toks[i + 1])) for i in range(len(toks) - 1)}


def _overlap(a: str, b: str) -> float:
    if not a or not b or a.strip() == b.strip():
        return 1.0 if a.strip() == b.strip() else (1.0 if a == b else 0.0)
    sa, sb = _bigram_set(a), _bigram_set(b)
    if not sa or not sb:
        return 1.0 / 1
    return len(sa & sb) / len(sa | sb)


class AgentLoopDetection(Evaluator):
    name = "agent_loop_detection"
    kind = "agent"

    def __init__(self, repetition_threshold: int = 3, similarity_threshold: float = 0.85,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.repetition_threshold = repetition_threshold
        self.similarity_threshold = similarity_threshold
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not run.nodes and not run.text:
            return _nan_or_skip(self.name, "no trace to analyse")

        penalties: List[float] = []

        # 1. repeated tool calls
        pairs = [(t.name or "?", _hashable_args(t.arguments)) for t in run.all_tool_calls]
        if pairs:
            rep = Counter(pairs)
            max_rep = max(rep.values())
            if max_rep >= self.repetition_threshold:
                penalties.append(min(1.0, (max_rep - 1) / (self.repetition_threshold * 2)))

        # 2. reasoning stagnation: consecutive agent outputs
        outputs = [nd.content for nd in run.nodes if nd.is_agent and nd.content]
        for i in range(len(outputs) - 1):
            if _overlap(outputs[i], outputs[i + 1]) >= self.similarity_threshold:
                penalties.append(0.8)
                break

        # 3. node re-entry / failed node
        labels = [nd.node_label or nd.name for nd in run.nodes if nd.node_label or nd.name]
        if labels:
            c = Counter(labels)
            if any(v >= 2 for v in c.values()):
                penalties.append(0.6)
        if any(nd.status == "FAILED" for nd in run.nodes):
            penalties.append(0.7)

        if not penalties:
            return _cr(self.name, 1.0, self.threshold, details={"clean": True})
        score = 1.0 - min(1.0, max(penalties))
        return _cr(self.name, score, self.threshold,
                   details={"max_tool_repetition": (max(Counter(pairs).values()) if pairs else 0),
                            "node_reenters": bool(labels),
                            "has_failed_node": any(nd.status == "FAILED" for nd in run.nodes)})


def _hashable_args(args: Dict) -> tuple:
    from ..types import _hashable

    return tuple(sorted((k, _hashable(v)) for k, v in (args or {}).items()))


__all__ = ["AgentLoopDetection"]
