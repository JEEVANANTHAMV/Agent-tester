"""giskard LLMJudge: a fully custom rubric. Pass a natural-language
``instruction``; the judge returns ``{"reason": str, "passed": bool}``."""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, as_bool, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _judge_user, _trace_text


class LLMJudge(Evaluator):
    name = "llm_judge"
    kind = "llm"

    _TEMPLATE = (
        "{instruction}\nEvaluate the interaction below and return JSON "
        '{{"reason": str, "passed": true|false}}.\nInteraction trace:\n{trace}\nJSON:'
    )

    def __init__(self, judge: Optional[JudgeClient] = None, instruction: str = "",
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.instruction = instruction
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not self.instruction:
            return CheckResult.error_(self.name, "no instruction supplied")
        prompt = self._TEMPLATE.format(instruction=self.instruction, trace=_trace_text(run))
        out = self.judge.complete_json(_judge_user(prompt))
        if not out or "passed" not in out:
            return CheckResult.error_(self.name, "judge returned no {reason,passed} object")
        passed = as_bool(out.get("passed"))
        reason = str(out.get("reason", ""))
        if passed:
            return CheckResult.pass_(self.name, reason, details={"reason": reason})
        return CheckResult.fail_(self.name, reason, details={"reason": reason})


__all__ = ["LLMJudge"]
