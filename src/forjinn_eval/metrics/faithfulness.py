"""ragas Faithfulness: fraction of output statements inferable from the
retrieved context. ``context`` = ``run.retrieved_context()``; if none is
present the metric SKIPs (nothing was retrieved to be faithful to)."""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, as_int01, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _fmt_list, _judge_user, _nan_or_skip


class Faithfulness(Evaluator):
    name = "faithfulness"
    kind = "rag"

    DECOMPOSE_STATEMENTS = (
        "Given a question and an answer, break down each sentence in the answer into one or more "
        "fully understandable, standalone statements. Replace all pronouns with full antecedents. "
        'Return JSON with a single key "statements" that is a list of strings.\n'
        "Question: {question}\nAnswer: {answer}\nJSON:"
    )

    NLI_JUDGE = (
        "Judge the faithfulness of a series of statements against a context. For each statement return "
        'a verdict of 1 if it can be DIRECTLY inferred from the context, else 0. Return JSON with key '
        '"statements" = a list of objects, each {"statement": <original statement>, "reason": str, '
        '"verdict": 0 or 1}. The number of returned statements MUST equal the number given. '
        "Context:\n{context}\nStatements: {statements}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None, kind: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name
        if kind:
            self.kind = kind

    def evaluate(self, run: AgentRun) -> CheckResult:
        ctx = run.retrieved_context()
        if not ctx:
            return _nan_or_skip(self.name, "no retrieved context available to check faithfulness")
        j = self.judge
        out = j.complete_json(_judge_user(_f(
            self.DECOMPOSE_STATEMENTS, question=run.question, answer=run.text)))
        if not out or not out.get("statements"):
            return _nan_or_skip(self.name, "statement decomposition returned no statements")
        statements = out["statements"]
        nli = j.complete_json(_judge_user(_f(
            self.NLI_JUDGE, context="\n".join(ctx), statements=_fmt_list(statements))))
        if not nli or not isinstance(nli.get("statements"), list):
            return CheckResult.error_(self.name, "NLI judge returned no verdicts")
        verdicts = [as_int01(v.get("verdict")) for v in nli["statements"]]
        if not verdicts:
            return _nan_or_skip(self.name, "NLI judge returned no verdicts")
        score = sum(verdicts) / len(verdicts)
        return _cr(self.name, score, self.threshold,
                   details={"faithful": sum(verdicts), "total": len(verdicts), "verdicts": verdicts})


__all__ = ["Faithfulness"]
