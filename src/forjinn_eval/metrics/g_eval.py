"""deepeval GEval: user-supplied ``criteria`` (+ optional evaluation params /
pre-written steps). The judge self-scores 0-10; final = score / 10 in [0,1].
This is the "build your own LLM metric" escape hatch."""
from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from ._common import _cr, _f, _fmt_list, _judge_user, _tool_repr
from .base import CheckResult, Evaluator


class GEval(Evaluator):
    name = "g_eval"
    kind = "llm"

    GENERATE_EVAL_STEPS = (
        "Given an evaluation criteria outlining how to judge {params}, generate 3-4 concise evaluation "
        "steps. Make clear how to evaluate {params} in relation to one another. Return JSON "
        '{"steps": [str, ...]}.\nCriteria: {criteria}\nJSON:'
    )

    SCORE_STEPS = (
        "You are an evaluator. Given the following evaluation steps, assess the response and return JSON "
        '{"score": an integer between 0 and 10, "reason": str}. 10 = strong alignment with the steps, '
        "0 = no alignment. Be specific and grounded in the steps. Only return valid JSON.\n"
        "Evaluation steps:\n{steps}\nTest case:\n{test_case}\nParams: {params}\nJSON:"
    )

    # NOTE: built as an f-string in evaluate() (the JSON braces are literal, so we
    # cannot use str.format on it - that would treat "score" as a field).

    def __init__(self, judge: Optional[JudgeClient] = None, criteria: str = "",
                 evaluation_steps: Optional[Sequence[str]] = None,
                 name: Optional[str] = None, threshold: float = 0.5,
                 strict_mode: bool = False):
        self.judge = _judge_or_default(judge)
        self.criteria = criteria
        self.evaluation_steps: List[str] = list(evaluation_steps or [])
        self.threshold = 1.0 if strict_mode else threshold
        self.strict_mode = strict_mode
        if not criteria and not self.evaluation_steps:
            raise ValueError("GEval requires either criteria or evaluation_steps")
        if name:
            self.name = name

    def _test_case(self, run: AgentRun) -> str:
        parts = [f"Input:\n{run.question}", f"Actual Output:\n{run.text}"]
        ctx = run.retrieved_context()
        if ctx:
            parts.append("Retrieval Context:\n" + "\n".join(ctx))
        if run.all_tool_calls:
            parts.append("Tools called:\n" + _fmt_list(map(_tool_repr, run.all_tool_calls)))
        if run.reference:
            parts.append("Reference Output:\n" + run.reference)
        return "\n\n".join(parts)

    def evaluate(self, run: AgentRun) -> CheckResult:
        steps = self.evaluation_steps
        if not steps:
            out = self.judge.complete_json(_judge_user(_f(
                self.GENERATE_EVAL_STEPS, params="the input and the actual output",
                criteria=self.criteria)))
            steps = (out or {}).get("steps", [])
        if not steps:
            return CheckResult.error_(self.name, "could not derive evaluation steps")
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        if self.strict_mode:
            out = self.judge.complete_json(_judge_user(
                f"You are a strict evaluator. Given these steps, return JSON "
                f'{{"score": 1 if the response follows the criteria 100%, else 0, "reason": str}}.\n'
                f"Steps:\n{numbered}\nTest case:\n{self._test_case(run)}\nJSON:"))
        else:
            out = self.judge.complete_json(_judge_user(_f(
                self.SCORE_STEPS, steps=numbered, test_case=self._test_case(run),
                params="input and actual output")))
        if not out or "score" not in out:
            return CheckResult.error_(self.name, "GEval judge returned no score")
        raw = float(out["score"])
        if self.strict_mode:
            score = 1.0 if raw >= 0.5 else 0.0
        else:
            score = max(0.0, min(10.0, raw)) / 10.0
        return _cr(self.name, score, self.threshold,
                   reason=str(out.get("reason", "")),
                   details={"raw_score": raw, "steps": steps, "reason": out.get("reason")})


__all__ = ["GEval"]
