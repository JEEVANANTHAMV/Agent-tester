"""Result + 4-state semantics for forjinn_eval.

Borrowed deliberately from Giskard's v3 result ladder:

* ``PASS``  - the evaluator reached a verdict and it held.
* ``FAIL``  - the evaluator reached a verdict and it did NOT hold.
* ``ERROR`` - no verdict could be produced (broken input, missing field, crash).
* ``SKIP``  - the evaluator does not apply (e.g. no tools configured).

Rollup priority: ``ERROR > FAIL > PASS > SKIP``. This is critical for LLM-style
evaluation: a broken key must not read *green*, and a legitimately-inapplicable
check must not read *red*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"


_PRIORITY = {Status.ERROR: 3, Status.FAIL: 2, Status.PASS: 1, Status.SKIP: 0}


def rollup(statuses: List[Status]) -> Status:
    """Combine a list of statuses using the priority (error beats fail, etc)."""
    if not statuses:
        return Status.SKIP
    return max(statuses, key=lambda s: _PRIORITY[s])


@dataclass
class Metric:
    """A single named numeric value attached to a CheckResult (e.g. latency_ms)."""

    name: str
    value: float
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit}


@dataclass
class CheckResult:
    """Outcome of one evaluator applied to one :class:`AgentRun`."""

    name: str  # evaluator name / label shown in the report
    status: Status
    reason: str = ""
    passed: Optional[bool] = None  # verdict where one exists (None if ERROR/SKIP)
    score: Optional[float] = None  # 0.0 - 1.0 where the evaluator produces a magnitude
    metrics: List[Metric] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pass_(cls, name: str, reason: str = "", **kw: Any) -> "CheckResult":
        passed = kw.pop("passed", True)
        return cls(name=name, status=Status.PASS, reason=reason, passed=True, **kw)

    @classmethod
    def fail_(cls, name: str, reason: str = "", score: Optional[float] = None, **kw: Any) -> "CheckResult":
        return cls(name=name, status=Status.FAIL, reason=reason, passed=False, score=score, **kw)

    @classmethod
    def error_(cls, name: str, reason: str = "", **kw: Any) -> "CheckResult":
        return cls(name=name, status=Status.ERROR, reason=reason, passed=None, **kw)

    @classmethod
    def skip_(cls, name: str, reason: str = "", **kw: Any) -> "CheckResult":
        return cls(name=name, status=Status.SKIP, reason=reason, passed=None, **kw)

    @property
    def ok(self) -> bool:
        """True only if the verdict was PASS."""
        return self.status == Status.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "passed": self.passed,
            "score": self.score,
            "metrics": [m.to_dict() for m in self.metrics],
            "details": self.details,
        }
