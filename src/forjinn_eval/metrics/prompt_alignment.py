"""deepeval PromptAlignment: fraction of given prompt instructions the output
followed (yes = passing)."""
from __future__ import annotations

from typing import Optional, Sequence

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _fmt_list, _judge_user, _nan_or_skip


class PromptAlignment(Evaluator):
    name = "prompt_alignment"
    kind = "alignment"

    PROMPT_ALIGNMENT = (
        "For each of the following prompt instructions, decide whether the LLM's actual output followed it. "
        'Answer "yes" ONLY if it COMPLETELY follows the instruction, "no" otherwise; be extra strict. '
        'Return JSON with key "verdicts" = a list of {"verdict": "yes" or "no", "reason": str}, '
        "one per instruction (number MUST equal). Provide a reason only for \"no\".\n"
        "Prompt instructions: {instructions}\nInput: {input}\nActual output: {actual}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, instructions: Sequence[str] = (),
                 threshold: float = 0.5, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.instructions = list(instructions)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not self.instructions:
            sysp = run.system_prompt()
            if not sysp:
                return _nan_or_skip(self.name, "no prompt instructions supplied")
            self.instructions = [ln.strip("- \u2022") for ln in sysp.splitlines() if ln.strip()]
            if len(self.instructions) > 12 or len(sysp) > 4000:
                return _nan_or_skip(
                    self.name,
                    "falling back to a large system prompt is not a usable instruction set",
                )
        out = self.judge.complete_json(_judge_user(_f(
            self.PROMPT_ALIGNMENT, instructions=_fmt_list(self.instructions),
            input=run.question, actual=run.text)))
        verdicts = (out or {}).get("verdicts", [])
        if not isinstance(verdicts, list) or not verdicts:
            return CheckResult.error_(self.name, "alignment judge returned no verdicts")
        yes = sum(1 for v in verdicts if str(v.get("verdict", "")).lower() in {"yes", "1", "true"})
        score = yes / len(verdicts)
        return _cr(self.name, score, self.threshold, details={"followed": yes, "total": len(verdicts)})


__all__ = ["PromptAlignment"]
