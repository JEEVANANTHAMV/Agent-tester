"""giskard AnswerRelevance: is the answer relevant to the question (in the
full-history context)?"""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, as_bool, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _judge_user


class AnswerRelevance(Evaluator):
    name = "answer_relevance"
    kind = "rag"

    _PROMPT = (
        "Evaluate whether the AI assistant's ANSWER is relevant to the QUESTION. Return JSON "
        '{"reason": str, "passed": true|false}\nQUESTION: {question}\nANSWER: {answer}\nJSON:'
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 1.0,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not run.text:
            return CheckResult.error_(self.name, "no answer text")
        out = self.judge.complete_json(_judge_user(_f(
            self._PROMPT, question=run.question, answer=run.text)))
        if not out or "passed" not in out:
            return CheckResult.error_(self.name, "relevance judge returned no verdict")
        passed = as_bool(out.get("passed"))
        reason = str(out.get("reason", ""))
        return _cr(self.name, 1.0 if passed else 0.0, self.threshold,
                   reason=reason, details={"reason": reason})


__all__ = ["AnswerRelevance"]
