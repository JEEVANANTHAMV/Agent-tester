"""Performance evaluators: token budgets + latency budgets."""
from __future__ import annotations

from typing import List, Optional

from ..capture import AgentRun
from .base import CheckResult, Evaluator

__all__ = ["LatencyBudget", "TokenBudget"]


class TokenBudget(Evaluator):
    """Cumulative token budget: input/output/total must stay within the limits.

    ``score`` is a fraction of the tightest budget (1.0 = well inside; 0.0 = over).
    """

    name = "token_budget"
    kind = "performance"

    def __init__(
        self,
        input_tokens: Optional[float] = None,
        output_tokens: Optional[float] = None,
        total_tokens: Optional[float] = None,
    ):
        self.limits = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def evaluate(self, run: AgentRun) -> CheckResult:
        u = run.usage
        over: List[str] = []
        worst = 1.0
        for key, limit in self.limits.items():
            if limit is None:
                continue
            val = getattr(u, key)
            ratio = val / limit if limit else 1.0
            worst = min(worst, max(0.0, 1.0 - ratio))
            if val > limit:
                over.append(f"{key}={val} > {limit}")
        if over:
            return CheckResult.fail_(self.name, "; ".join(over), score=0.0)
        return CheckResult.pass_(self.name, f"within budget {self.limits}", score=round(worst, 3))


class LatencyBudget(Evaluator):
    """Cumulative agent-node latency budget (ms)."""

    name = "latency_budget"
    kind = "performance"

    def __init__(self, max_ms: float, node: str = "final_agent"):
        self.max_ms = max_ms
        self.node = node

    def evaluate(self, run: AgentRun) -> CheckResult:
        target = run.final_agent_node if self.node == "final_agent" else run.agent_node
        if not target:
            return CheckResult.error_(self.name, "no agent node to time")
        delta = target.time.delta_ms
        if delta is None:
            return CheckResult.error_(self.name, "agent node has no timeMetadata.delta")
        if delta <= self.max_ms:
            return CheckResult.pass_(self.name, f"agent delta={delta:.0f}ms <= {self.max_ms}ms", score=1.0)
        ratio = delta / self.max_ms
        return CheckResult.fail_(
            self.name,
            f"agent delta={delta:.0f}ms exceeds {self.max_ms}ms",
            score=max(0.0, 1.0 - (ratio - 1.0)),
            details={"delta_ms": delta, "budget_ms": self.max_ms},
        )
