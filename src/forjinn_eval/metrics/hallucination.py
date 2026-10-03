"""deepeval Hallucination: fraction of contexts the output does NOT contradict
(yes = agrees). Needs ``run.retrieved_context()``."""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from ._common import _cr, _fmt_list, _judge_user, _nan_or_skip
from .base import CheckResult, Evaluator


class Hallucination(Evaluator):
    name = "hallucination"
    kind = "hallucination"

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ctx = run.retrieved_context()
        if not ctx:
            return _nan_or_skip(self.name, "no contexts to check hallucination against")
        n = len(ctx)
        prompt = (
            f"For each context in the list, decide whether the actual output AGREES with it. "
            f'Return JSON with key "verdicts" = a list of {n} objects '
            f'{{"verdict": "yes"|"no", "reason": str}}. Use "no" ONLY for a '
            f"contradiction; forgive missing detail.\nContexts: {_fmt_list(ctx)}\n"
            f"Actual output: {run.text}\nJSON:"
        )
        out = self.judge.complete_json(_judge_user(prompt))
        verdicts = (out or {}).get("verdicts", [])
        if not isinstance(verdicts, list) or not verdicts:
            return CheckResult.error_(self.name, "hallucination judge returned no verdicts")
        yes = sum(1 for v in verdicts if str(v.get("verdict", "")).lower() in {"yes", "1", "true"})
        score = yes / len(verdicts)
        return _cr(self.name, score, self.threshold, details={"agreement": yes, "total": len(verdicts)})


__all__ = ["Hallucination"]
