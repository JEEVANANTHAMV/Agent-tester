"""Deterministic agent-tool-call metrics (ragas ``ToolCall*`` family).

These compare the tools an agent *actually called* against a *reference* set of
tool calls, fully offline (no LLM):

* :class:`ToolCallAccuracy` - order-sensitive (or sequence-aligned) accuracy of
  the tool-call sequence vs the reference.
* :class:`ToolCallF1`       - set-based F1 (exact name+args match for a TP).

The reference is read from :attr:`AgentRun.reference_tool_calls` or passed
explicitly.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import List, Optional

from ..capture import AgentRun
from ..types import ToolCall
from ._common import _cr, _nan_or_skip
from .base import CheckResult, Evaluator


def _key(tc: ToolCall) -> tuple:
    from ..types import _hashable

    return (tc.name or "", tuple(sorted((k, _hashable(v)) for k, v in (tc.arguments or {}).items())))


def _normalize(raw: Sequence) -> List[ToolCall]:
    out: List[ToolCall] = []
    for r in raw or []:
        if isinstance(r, ToolCall):
            out.append(r)
        elif isinstance(r, str):
            out.append(ToolCall(name=r))
        elif isinstance(r, (tuple, list)):
            out.append(ToolCall(name=r[0], arguments=dict(r[1]) if len(r) > 1 else {}))
        elif isinstance(r, dict):
            out.append(ToolCall.from_dict(r))
    return out


def _lcs_len(a: List[tuple], b: List[tuple]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len(a)][len(b)]


class ToolCallAccuracy(Evaluator):
    """Order-sensitive tool-call sequence accuracy vs the reference."""

    name = "tool_call_accuracy"
    kind = "tool"

    def __init__(self, reference_tool_calls: Optional[Sequence] = None, strict_order: bool = True,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.reference_tool_calls = _normalize(reference_tool_calls) if reference_tool_calls is not None else None
        self.strict_order = strict_order
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference_tool_calls if self.reference_tool_calls is not None else run.reference_tool_calls
        if not ref:
            return _nan_or_skip(self.name, "no reference tool calls")
        got = run.all_tool_calls
        got_keys = [_key(t) for t in got]
        ref_keys = [_key(t) for t in ref]
        if self.strict_order:
            n = len(got_keys)
            correct = sum(1 for i in range(min(n, len(ref_keys))) if got_keys[i] == ref_keys[i])
            score = correct / len(ref_keys)
        else:
            # multiset overlap (order-insensitive)
            gc, rc = defaultdict(int), defaultdict(int)
            for k in got_keys:
                gc[k] += 1
            for k in ref_keys:
                rc[k] += 1
            overlap = sum(min(gc[k], rc[k]) for k in rc)
            score = overlap / len(ref_keys)
        return _cr(self.name, score, self.threshold,
                   details={"expected": [t.name for t in ref], "got": [t.name for t in got],
                            "strict_order": self.strict_order})


class ToolCallF1(Evaluator):
    """Set-based F1 over (name, args) tool calls."""

    name = "tool_call_f1"
    kind = "tool"

    def __init__(self, reference_tool_calls: Optional[Sequence] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        self.reference_tool_calls = _normalize(reference_tool_calls) if reference_tool_calls is not None else None
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference_tool_calls if self.reference_tool_calls is not None else run.reference_tool_calls
        if not ref:
            return _nan_or_skip(self.name, "no reference tool calls")
        got = {_key(t) for t in run.all_tool_calls}
        refset = {_key(t) for t in ref}
        tp = len(got & refset)
        fp = len(got - refset)
        fn = len(refset - got)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        return _cr(self.name, f1, self.threshold,
                   details={"precision": p, "recall": r, "f1": f1,
                            "expected": [t.name for t in ref], "got": [t.name for t in run.all_tool_calls]})


__all__ = ["ToolCallAccuracy", "ToolCallF1"]
