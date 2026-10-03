"""Tool-evaluation evaluators (deterministic)."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, Union

from ..capture import AgentRun
from .base import CheckResult, Evaluator

__all__ = [
    "AvailableToolsExposed",
    "NoToolsExpected",
    "OnlyAllowedTools",
    "ToolCallCount",
    "ToolCallOrder",
    "ToolCallSetF1",
]


class ToolCallOrder(Evaluator):
    """Deterministic tool-order check: ``run.all_tool_calls`` order must equal
    the expected sequence (``expected`` may be a list of tool names or
    ``(name, args)`` tuples)."""

    name = "tool_call_order"
    kind = "tool"

    def __init__(self, expected: Iterable[Union[str, tuple]]):
        self.expected = list(expected)

    def evaluate(self, run: AgentRun) -> CheckResult:
        got = [(t.name, t.arguments) for t in run.all_tool_calls if t.name]
        exp_pairs = []
        for e in self.expected:
            if isinstance(e, (tuple, list)):
                exp_pairs.append((e[0], e[1]))
            else:
                exp_pairs.append((e, None))
        # Each expected entry is matched positionally:
        #   - a name (str)      -> the tool name must equal
        #   - a (name, args)    -> name equals and args are a subset of actual
        name_only = all(v is None for _, v in exp_pairs)
        if name_only:
            ok = [g[0] for g in got] == [e[0] for e in exp_pairs]
        else:
            if len(got) != len(exp_pairs):
                ok = False
            else:
                ok = True
                for (gn, gargs), (en, eargs) in zip(got, exp_pairs):
                    if gn != en:
                        ok = False
                        break
                    if eargs is not None and not all(
                        gargs.get(k) == v for k, v in eargs.items()
                    ):
                        ok = False
                        break
        if ok:
            return CheckResult.pass_(self.name, f"tool order matches {self.expected}", score=1.0)
        got_names = [t.name for t in run.all_tool_calls]
        return CheckResult.fail_(
            self.name,
            f"tool order mismatch: expected={self.expected}, got={got_names}",
            score=0.0,
            details={"expected": self.expected, "got": got_names},
        )


class ToolCallSetF1(Evaluator):
    """Deterministic unordered set-match F1 over tool call *names*."""

    name = "tool_call_f1"
    kind = "tool"
    threshold: float

    def __init__(self, expected: Iterable[str], threshold: float = 1.0):
        self.expected = list(expected)
        self.threshold = threshold

    def evaluate(self, run: AgentRun) -> CheckResult:
        got = set(run.tool_names)
        exp = set(self.expected)
        tp = len(got & exp)
        fp = len(got - exp)
        fn = len(exp - got)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if f1 >= self.threshold:
            return CheckResult.pass_(self.name, f"tool F1={f1:.2f} >= {self.threshold}", score=f1)
        return CheckResult.fail_(
            self.name,
            f"tool F1={f1:.2f} < {self.threshold} (exp={exp}, got={got})",
            score=f1,
            details={"precision": precision, "recall": recall, "f1": f1, "expected": sorted(exp), "got": sorted(got)},
        )


class OnlyAllowedTools(Evaluator):
    """Deterministic gate: every called tool name must be in ``allowed``.

    A useful anti-prompt-injection gate.
    """

    name = "only_allowed_tools"
    kind = "tool"

    def __init__(self, allowed: Optional[Iterable[str]] = None):
        self.allowed = set(allowed) if allowed is not None else None

    def evaluate(self, run: AgentRun) -> CheckResult:
        if self.allowed is None:
            # If no explicit whitelist, use the run's own available_tools.
            node = run.final_agent_node
            if not node:
                return CheckResult.skip_(self.name, "no agent node / no configured whitelist")
            self.allowed = {t.get("name") for t in node.available_tools if t.get("name")}
            if not self.allowed:
                return CheckResult.skip_(self.name, "no configured whitelist (agent has no available_tools)")
        got = set(run.tool_names)
        bad = got - self.allowed
        if bad:
            return CheckResult.fail_(
                self.name,
                f"forbidden tools called: {sorted(bad)}",
                details={"forbidden": sorted(bad), "allowed": sorted(self.allowed or [])},
            )
        return CheckResult.pass_(self.name, f"only allowed tools called: {sorted(got)}")


class ToolCallCount(Evaluator):
    """Deterministic bound on the number of tool invocations (0, 1, 2..)."""

    name = "tool_call_count"
    kind = "tool"

    def __init__(self, min_calls: int = 0, max_calls: Optional[int] = None):
        self.min_calls = min_calls
        self.max_calls = max_calls

    def evaluate(self, run: AgentRun) -> CheckResult:
        n = len(run.all_tool_calls)
        lo_ok = n >= self.min_calls
        hi_ok = self.max_calls is None or n <= self.max_calls
        if lo_ok and hi_ok:
            return CheckResult.pass_(self.name, f"calls={n}", score=1.0)
        rng = f"min={self.min_calls}, max={self.max_calls if self.max_calls is not None else 'inf'}"
        return CheckResult.fail_(self.name, f"calls={n} outside [{rng}]", score=0.0)


class NoToolsExpected(Evaluator):
    """Sanity: the run should NOT have called any tool."""

    name = "no_tools_expected"
    kind = "tool"

    def evaluate(self, run: AgentRun) -> CheckResult:
        if run.all_tool_calls:
            return CheckResult.fail_(
                self.name,
                f"expected no tools, got {len(run.all_tool_calls)}: {run.tool_names}",
            )
        return CheckResult.pass_(self.name, "no tools called")


class AvailableToolsExposed(Evaluator):
    """If the agent has MCP tools wired, they must be exposed in ``available_tools``."""

    name = "available_tools_exposed"
    kind = "tool"

    def __init__(self, min_exposed: int = 1):
        self.min_exposed = min_exposed

    def evaluate(self, run: AgentRun) -> CheckResult:
        node = run.final_agent_node
        if not node:
            return CheckResult.error_(self.name, "no agent node")
        if not node.input.get("agentTools"):
            return CheckResult.skip_(self.name, "agent has no agentTools configured")
        exposed = node.available_tools or []
        if len(exposed) >= self.min_exposed:
            return CheckResult.pass_(self.name, f"{len(exposed)} available tools exposed", score=len(exposed))
        return CheckResult.fail_(
            self.name,
            f"agentTools configured but only {len(exposed)} exposed (expected >= {self.min_exposed})",
            score=len(exposed) / self.min_exposed if self.min_exposed else 0.0,
        )
