"""deepeval Summarization: quality of a summary (``actual_output``) of a source
(``input``). Two sub-scores - coverage (does the summary answer questions drawn
from the source) and faithfulness (does the summary avoid contradicting the
source) - averaged into ``[0, 1]``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default, as_int01  # as_int01 kept for faithful check
from ._common import _cr, _f, _judge_user, _nan_or_skip, _verdict_list
from .base import CheckResult, Evaluator


class Summarization(Evaluator):
    name = "summarization"
    kind = "content"

    _GEN_Q = (
        "Given the source document, generate {n} key questions a good summary MUST answer. "
        'Return JSON {"questions": [str, ...]}.\nSource: {source}\nJSON:'
    )
    _COVERAGE = (
        "For each question, decide whether the SUMMARy answers it. Return JSON "
        '{"answered": [0|1, ...]} one per question, in order.\n'
        "Summary: {summary}\nQuestions:\n{questions}\nJSON:"
    )
    _FAITHFUL = (
        "Decide whether the summary contains any claim NOT supported by (or contradicting) the "
        "source. Return JSON {\"faithful\": 0|1, \"reason\": str}.\n"
        "Source: {source}\nSummary: {summary}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 n: int = 5, assessment_questions: Optional[Sequence[str]] = None,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.n = n
        self.assessment_questions = list(assessment_questions) if assessment_questions else None
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        source, summary = run.question, run.text
        if not summary:
            return _nan_or_skip(self.name, "no summary output")
        if not source:
            return _nan_or_skip(self.name, "no source input")
        questions = self.assessment_questions
        if not questions:
            out = self.judge.complete_json(_judge_user(_f(self._GEN_Q, n=self.n, source=source)))
            questions = (out or {}).get("questions", [])
        if not questions:
            return CheckResult.error_(self.name, "no assessment questions generated")
        cov = self.judge.complete_json(_judge_user(_f(
            self._COVERAGE, summary=summary,
            questions="\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)))))
        answered = _verdict_list((cov or {}).get("answered"))
        if not answered:
            return CheckResult.error_(self.name, "coverage judge returned nothing")
        coverage = sum(answered) / len(answered)
        f = self.judge.complete_json(_judge_user(_f(self._FAITHFUL, source=source, summary=summary)))
        if f:
            fv = as_int01(f.get("faithful"))
            faithful = 0.0 if fv in ("", None) else (max(0.0, min(1.0, float(fv))) if isinstance(fv, (int, float)) else (1.0 if fv else 0.0))
        else:
            faithful = 0.0
        score = 0.5 * coverage + 0.5 * faithful
        return _cr(self.name, score, self.threshold,
                   details={"coverage": coverage, "faithful": faithful, "questions": questions})


__all__ = ["Summarization"]
