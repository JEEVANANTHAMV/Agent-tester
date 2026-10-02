"""ragas AnswerRelevancy (embedding-free variant): the judge self-scores how
well an answer addresses the question. Score = mean over ``strictness`` samples
of an LLM 0-10 self-relevance, /10."""
from __future__ import annotations

from typing import List, Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _judge_user


class AnswerRelevancy(Evaluator):
    name = "answer_relevancy"
    kind = "rag"

    _PROMPT = (
        "Judge how directly the answer is relevant to the question. Score 0-10 (10 = fully "
        'relevant, 0 = unrelated). Return JSON {"score": int 0-10, "reason": str}.\n'
        "Question: {question}\nAnswer: {answer}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 strictness: int = 3, name: Optional[str] = None, kind: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.strictness = strictness if strictness % 2 != 0 else strictness + 1
        if name:
            self.name = name
        if kind:
            self.kind = kind

    def evaluate(self, run: AgentRun) -> CheckResult:
        scores: List[float] = []
        for _ in range(self.strictness):
            out = self.judge.complete_json(_judge_user(_f(
                self._PROMPT, question=run.question, answer=run.text)))
            if not out:
                continue
            s = out.get("score")
            if isinstance(s, (int, float)):
                scores.append(max(0.0, min(10.0, float(s))) / 10.0)
        if not scores:
            return CheckResult.error_(self.name, "no relevancy score produced")
        score = sum(scores) / len(scores)
        return _cr(self.name, score, self.threshold,
                   details={"samples": scores, "threshold": self.threshold})


__all__ = ["AnswerRelevancy"]
