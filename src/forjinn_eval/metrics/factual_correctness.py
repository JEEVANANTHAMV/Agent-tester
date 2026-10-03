"""ragas FactualCorrectness: verify response claims against a reference, report
precision/recall/f1. Uses ``run.reference``; default mode ``f1``."""
from __future__ import annotations

from typing import List, Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default, as_int01
from ._common import _cr, _f, _fbeta, _fmt_list, _judge_user, _nan_or_skip
from .base import CheckResult, Evaluator


class FactualCorrectness(Evaluator):
    name = "factual_correctness"
    kind = "rag"

    NLI_JUDGE = (
        "Judge the faithfulness of a series of statements against a context. For each statement return "
        'a verdict of 1 if it can be DIRECTLY inferred from the context, else 0. Return JSON with key '
        '"statements" = a list of objects, each {"statement": <original statement>, "reason": str, '
        '"verdict": 0 or 1}. The number of returned statements MUST equal the number given. '
        "Context:\n{context}\nStatements: {statements}\nJSON:"
    )

    _DECOMP = (
        "Decompose the text into standalone, independently verifiable claims. "
        'Return JSON {"claims": [str]}. Text: '
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 mode: str = "f1", name: Optional[str] = None, kind: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.mode = mode
        if name:
            self.name = name
        if kind:
            self.kind = kind

    def _verify(self, claims: List[str], premise: str) -> List[int]:
        if not claims:
            return []
        out = self.judge.complete_json(_judge_user(_f(
            self.NLI_JUDGE, context=premise, statements=_fmt_list(claims))))
        if not out or not isinstance(out.get("statements"), list):
            return []
        return [as_int01(v.get("verdict")) for v in out["statements"]]

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = run.reference
        if not ref:
            return _nan_or_skip(self.name, "no reference provided")
        resp_out = self.judge.complete_json(_judge_user(self._DECOMP + run.text + "\nJSON:"))
        ref_out = self.judge.complete_json(_judge_user(self._DECOMP + ref + "\nJSON:"))
        resp_claims = (resp_out or {}).get("claims", [])
        ref_claims = (ref_out or {}).get("claims", [])
        tp = sum(self._verify(resp_claims, ref))
        fp = len(resp_claims) - tp
        if self.mode == "precision":
            fn = 0
            score = tp / (tp + fp + 1e-8)
        else:
            fn = len(ref_claims) - sum(self._verify(ref_claims, run.text))
            if self.mode == "recall":
                score = tp / (tp + fn + 1e-8)
            else:
                score = _fbeta(tp, fp, fn, 1.0)
        return _cr(self.name, round(score, 2), self.threshold, details={"tp": tp, "fp": fp, "fn": fn})


__all__ = ["FactualCorrectness"]
