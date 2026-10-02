"""Base evaluator.

Every evaluator (deterministic or LLM-judged) is a plain object with ``name``
and ``kind`` and implements ``evaluate(run) -> CheckResult``. This mirrors the
RAGAS "metric is the atom" pattern: one reusable object, a declared input
contract read directly off :class:`~forjinn_eval.capture.AgentRun`, and a
single-sample 4-state verdict.
"""
from __future__ import annotations

from ..results import CheckResult

__all__ = ["Evaluator", "CheckResult"]


class Evaluator:
    """Base evaluator. Subclasses must define ``name`` and implement ``evaluate``."""

    name: str = "base"
    kind: str = "base"

    def evaluate(self, run) -> CheckResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} {self.name}>"
