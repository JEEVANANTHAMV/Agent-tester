"""Answer-quality LLM metrics (ragas ``NoiseSensitivity`` + ``AnswerAccuracy``).

* :class:`NoiseSensitivity` - how often the answer introduces claims NOT in the
  reference (a noise/hallucination signal). Lower is better; the score is
  ``1 - (unsupported claims / all claims)``.
* :class:`AnswerAccuracy`    - NVIDIA-style dual-judge accuracy: the judge
  averages two 0/2/4 ratings of the answer vs the reference, normalised to [0,1].
"""
from __future__ import annotations

from typing import List, Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _fmt_list, _judge_user, _nan_or_skip, _verdict_list


class NoiseSensitivity(Evaluator):
    name = "noise_sensitivity"
    kind = "rag"

    _DECOMP = (
        "Decompose the answer into standalone statements. Return JSON "
        '{"statements": [str, ...]}.\nAnswer: {answer}\nJSON:'
    )
    _NLI = (
        "For each statement, say whether it is supported by the reference (ground truth). "
        'Return JSON {"verdicts": [0|1, ...]} one per statement, in order.\n'
        "Reference: {reference}\nStatements: {statements}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, reference: Optional[str] = None,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.reference = reference
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference")
        out = self.judge.complete_json(_judge_user(_f(self._DECOMP, answer=run.text or "")))
        statements = (out or {}).get("statements", [])
        if not statements:
            return _cr(self.name, 1.0, self.threshold, details={"statements": 0})
        nli = self.judge.complete_json(_judge_user(_f(
            self._NLI, reference=ref, statements=_fmt_list(statements))))
        verdicts = _verdict_list((nli or {}).get("verdicts"))
        if not verdicts:
            return CheckResult.error_(self.name, "NLI judge returned no verdicts")
        unsupported = sum(1 for v in verdicts if not v)
        noise = unsupported / len(verdicts)
        score = 1.0 - noise
        return _cr(self.name, score, self.threshold,
                   details={"statements": len(verdicts), "unsupported": unsupported, "noise": noise})


class AnswerAccuracy(Evaluator):
    name = "answer_accuracy"
    kind = "rag"

    _JUDGE = (
        "Rate how correctly the answer responds to the question, given the ground-truth reference. "
        "Use this rubric:\n"
        "<1> The answer is inaccurate and provides no useful information.\n"
        "<2> The answer is vague or contains minor errors but is still helpful.\n"
        "<4> The answer is very good and contains almost all information from the reference.\n"
        "Return JSON {\"rating\": 1|2|4, \"reason\": str}.\n"
        "Question: {question}\nGround-truth: {reference}\nAnswer: {answer}\nJSON:"
    )

    RATING_MAP = {1: 0.0, 2: 1.0 / 3, 4: 1.0}
    _MAX_RATING = 10

    def __init__(self, judge: Optional[JudgeClient] = None, reference: Optional[str] = None,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.reference = reference
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference")
        ratings: List[float] = []
        # two judge samples, averaged (NVIDIA dual-judge style)
        for _ in range(2):
            out = self.judge.complete_json(_judge_user(_f(
                self._JUDGE, question=run.question, reference=ref, answer=run.text or "")))
            if not out:
                continue
            r = out.get("rating")
            if isinstance(r, (int, float)):
                r = float(r)
                # rubric uses 1/2/4, but a loose judge may emit 0-10 - normalise
                ratings.append(max(0.0, min(1.0, r / self._MAX_RATING)) if r not in (1, 2, 4)
                               else self.RATING_MAP[int(round(r))])
            elif str(r).strip() in self.RATING_MAP:
                ratings.append(self.RATING_MAP[int(r.strip())])
        if not ratings:
            return CheckResult.error_(self.name, "no accuracy ratings produced")
        score = sum(ratings) / len(ratings)
        return _cr(self.name, score, self.threshold, details={"ratings": ratings})


__all__ = ["NoiseSensitivity", "AnswerAccuracy"]
