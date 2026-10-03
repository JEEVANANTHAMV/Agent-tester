"""giskard Conformity: judge the whole trace against a plain-text ``rule``."""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default, as_bool
from ._common import _cr, _judge_user, _trace_text
from .base import CheckResult, Evaluator


class Conformity(Evaluator):
    name = "conformity"
    kind = "llm"

    def __init__(self, judge: Optional[JudgeClient] = None, rule: str = "",
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.rule = rule
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not self.rule:
            return CheckResult.error_(self.name, "no rule supplied")
        prompt = (
            "Evaluate whether the agent interaction conforms to this rule. Judge the outputs/metadata; "
            "use inputs as reference. Return JSON {\"reason\": str, \"passed\": true|false}\n"
            f"RULE: {self.rule}\nTRACE:\n{_trace_text(run)}\nJSON:"
        )
        out = self.judge.complete_json(_judge_user(prompt))
        if not out or "passed" not in out:
            return CheckResult.error_(self.name, "conformity judge returned no verdict")
        passed = as_bool(out.get("passed"))
        reason = str(out.get("reason", ""))
        return _cr(self.name, 1.0 if passed else 0.0, 1.0,
                   reason=reason, details={"reason": reason, "rule": self.rule})


__all__ = ["Conformity"]
