"""ragas AnswerCorrectness: ``0.75*F1(claims) + 0.25*similarity`` vs a reference
answer. Reference comes from ``run.reference`` (set via ``run.raw['reference']``);
without one the metric SKIPs."""
from __future__ import annotations

from typing import List, Optional, Sequence

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _fbeta, _fmt_list, _judge_user, _nan_or_skip, _string_similarity


class AnswerCorrectness(Evaluator):
    name = "answer_correctness"
    kind = "rag"

    DECOMPOSE_STATEMENTS = (
        "Given a question and an answer, break down each sentence in the answer into one or more "
        "fully understandable, standalone statements. Replace all pronouns with full antecedents. "
        'Return JSON with a single key "statements" that is a list of strings.\n'
        "Question: {question}\nAnswer: {answer}\nJSON:"
    )

    CLASSIFY_TPFP_FN = (
        "Given a ground truth and a list of answer statements, classify each answer statement as TP "
        "(present in the answer AND directly supported by the ground truth), FP (present in the answer "
        "but NOT supported by the ground truth) or FN (in the ground truth but absent from the answer). "
        'Each statement belongs to exactly one category. Return JSON with keys "TP","FP","FN", each '
        'a list of {"statement","reason"}.\nQuestion: {question}\nGround truth statements: {ref}\n'
        "Answer statements: {answer}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 weights: Sequence[float] = (0.75, 0.25), name: Optional[str] = None,
                 kind: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.weights = list(weights)
        if name:
            self.name = name
        if kind:
            self.kind = kind

    def _claims(self, question: str, text: str) -> List[str]:
        out = self.judge.complete_json(_judge_user(_f(
            self.DECOMPOSE_STATEMENTS, question=question, answer=text)))
        return (out or {}).get("statements", []) if out else []

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = run.reference
        if not ref:
            return _nan_or_skip(self.name, "no reference answer provided (run.raw['reference'])")
        ref_claims = self._claims(run.question, ref)
        ans_claims = self._claims(run.question, run.text)
        if not ref_claims and not ans_claims:
            factuality = 1.0
        else:
            cls = self.judge.complete_json(_judge_user(_f(
                self.CLASSIFY_TPFP_FN, question=run.question, ref=_fmt_list(ref_claims),
                answer=_fmt_list(ans_claims))))
            if not cls:
                return CheckResult.error_(self.name, "claim classifier returned unparseable output")
            tp, fp, fn = len(cls.get("TP", [])), len(cls.get("FP", [])), len(cls.get("FN", []))
            factuality = _fbeta(tp, fp, fn, 1.0)
        sim = _string_similarity(run.text, ref)
        w1, w2 = (self.weights + [1 - self.weights[0]])[:2]
        score = w1 * factuality + w2 * sim
        return _cr(self.name, score, self.threshold, details={"factuality": factuality, "similarity": sim})


__all__ = ["AnswerCorrectness"]
