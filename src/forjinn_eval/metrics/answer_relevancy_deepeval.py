"""deepeval AnswerRelevancy: fraction of output statements relevant to the
input (yes + borderline count as passing)."""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from ._common import _cr, _fmt_list, _judge_user
from .base import CheckResult, Evaluator


class AnswerRelevancyDeepeval(Evaluator):
    name = "answer_relevancy_deepeval"
    kind = "rag"

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        o = self.judge.complete_json(_judge_user(
            'Break the text into atomic statements. Return JSON {"statements": [str]}.\n'
            "Text: " + run.text + "\nJSON:"))
        statements = (o or {}).get("statements", [])
        if not statements:
            return _cr(self.name, 1.0, self.threshold, details={"statements": []})
        n = len(statements)
        prompt = (
            f"For each of these {n} statements, decide if it is relevant to addressing the input. "
            f'Return JSON key "verdicts" = a list of {n} objects '
            f'{{"verdict": "yes"|"no"|"borderline", "reason": str}}.\nInput: {run.question}\n'
            f"Statements: {_fmt_list(statements)}\nJSON:"
        )
        out = self.judge.complete_json(_judge_user(prompt))
        verdicts = (out or {}).get("verdicts", [])
        if not isinstance(verdicts, list) or not verdicts:
            return CheckResult.error_(self.name, "relevancy judge returned no verdicts")
        good = sum(1 for v in verdicts if str(v.get("verdict", "")).lower() in {"yes", "borderline"})
        score = good / len(verdicts)
        return _cr(self.name, score, self.threshold, details={"good": good, "total": len(verdicts)})


__all__ = ["AnswerRelevancyDeepeval"]
