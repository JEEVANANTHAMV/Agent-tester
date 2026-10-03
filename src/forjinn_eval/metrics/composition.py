"""Composable evaluators (AllOf / AnyOf / Not) - Giskard-style Boolean logic.

These wrap one or more :class:`Evaluator`s and combine their 4-state verdicts
using the Giskard rollup priority (``ERROR > FAIL > PASS > SKIP``):

* :class:`AllOf` - passes only if every inner check passes (AND).
* :class:`AnyOf` - passes if at least one inner check passes (OR).
* :class:`Not`   - inverts an inner check's PASS/FAIL; ERROR/SKIP pass through
  unchanged so a broken key can never be laundered into a pass.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional

from ..capture import AgentRun
from ..results import Status
from .base import CheckResult, Evaluator


class _Composed(Evaluator):
    name = "composed"
    kind = "composite"

    def evaluate(self, run: AgentRun) -> CheckResult:
        results = [c.evaluate(run) for c in self.inner]
        return self._combine(results)

    def _combine(self, results: List[CheckResult]) -> CheckResult:
        raise NotImplementedError


class AllOf(_Composed):
    """AND: PASS only if all inner checks PASS; any ERROR wins, then any FAIL."""

    name = "all_of"

    def __init__(self, checks: Sequence[Evaluator], name: Optional[str] = None):
        self.inner = list(checks)
        if name:
            self.name = name

    def _combine(self, results: List[CheckResult]) -> CheckResult:
        if any(r.status == Status.ERROR for r in results):
            return CheckResult.error_(self.name, "one or more inner checks errored")
        if any(r.status == Status.FAIL for r in results):
            failed = [r.name for r in results if r.status == Status.FAIL]
            return CheckResult.fail_(self.name, f"failed inner checks: {failed}",
                                     details={"failed": failed})
        all_pass = all(r.status == Status.PASS for r in results)
        if all_pass:
            return CheckResult.pass_(self.name, "all inner checks passed",
                                     details={"checked": len(results)})
        return CheckResult.skip_(self.name, "no inner check produced a decisive verdict")


class AnyOf(_Composed):
    """OR: PASS if at least one inner check PASSes; ERROR only if all error."""

    name = "any_of"

    def __init__(self, checks: Sequence[Evaluator], name: Optional[str] = None):
        self.inner = list(checks)
        if name:
            self.name = name

    def _combine(self, results: List[CheckResult]) -> CheckResult:
        if any(r.status == Status.PASS for r in results):
            passed = [r.name for r in results if r.status == Status.PASS]
            return CheckResult.pass_(self.name, f"passed via: {passed}", details={"passed_via": passed})
        if any(r.status == Status.ERROR for r in results):
            errored = [r.name for r in results if r.status == Status.ERROR]
            return CheckResult.error_(self.name, f"errored inner checks: {errored}", details={"errored": errored})
        if any(r.status == Status.FAIL for r in results):
            return CheckResult.fail_(self.name, "no inner check passed", details={})
        return CheckResult.skip_(self.name, "no decisive inner verdict")


class Not(_Composed):
    """Invert one check's PASS<->FAIL; ERROR/SKIP pass through unchanged."""

    name = "not"

    def __init__(self, check: Evaluator, name: Optional[str] = None):
        self.inner = [check]
        if name:
            self.name = name

    def _combine(self, results: List[CheckResult]) -> CheckResult:
        r = results[0]
        if r.status == Status.PASS:
            return CheckResult.fail_(self.name, "inner check passed (negated)", score=r.score,
                                     details={"inner": r.name})
        if r.status == Status.FAIL:
            return CheckResult.pass_(self.name, "inner check failed (negated)", score=r.score,
                                     details={"inner": r.name})
        # ERROR / SKIP pass through (never laundered)
        return r


__all__ = ["AllOf", "AnyOf", "Not"]
