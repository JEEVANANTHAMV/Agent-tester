"""giskard Groundedness: is the answer faithful to ``run.retrieved_context()``?
Omissions are allowed; ungrounded additions fail."""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, as_bool, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _judge_user, _nan_or_skip


class Groundedness(Evaluator):
    name = "groundedness"
    kind = "rag"

    _PROMPT = (
        "Evaluate whether the AI agent's answer is factually GROUNDED in the reference context. "
        "Only penalize ungrounded facts; omissions are allowed. Return JSON "
        '{"reason": str, "passed": true|false}\n'
        "AGENT ANSWER:\n{answer}\nREFERENCE CONTEXT (joined):\n{context}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 1.0,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ctx = run.retrieved_context()
        if not ctx:
            return _nan_or_skip(self.name, "no reference context to ground against")
        if not run.text:
            return CheckResult.error_(self.name, "no answer text")
        out = self.judge.complete_json(_judge_user(_f(
            self._PROMPT, answer=run.text, context="\n".join(ctx))))
        if not out or "passed" not in out:
            return CheckResult.error_(self.name, "groundedness judge returned no verdict")
        passed = as_bool(out.get("passed"))
        reason = str(out.get("reason", ""))
        return _cr(self.name, 1.0 if passed else 0.0, self.threshold,
                   reason=reason, details={"reason": reason})


__all__ = ["Groundedness"]
