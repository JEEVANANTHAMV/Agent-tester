"""deepeval TaskCompletion: extract task+outcome, judge a 0-1 verdict."""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _fmt_list, _judge_user, _tool_repr


class TaskCompletion(Evaluator):
    name = "task_completion"
    kind = "task"

    TASK_AND_OUTCOME = (
        "From the input and the agent's actual output, identify (1) the task (the objective the user asked "
        'for) and (2) the outcome (a strictly factual description of what actually happened, no subjective '
        'words like "successfully"). Return JSON {"task": str, "outcome": str}.\nInput: {input}\n'
        "Tools called: {tools}\nActual output: {actual}\nJSON:"
    )

    TASK_VERDICT = (
        "Given the task (desired outcome) and the actual achieved outcome, compare how well the actual "
        'outcome aligns with the desired task. Return JSON {"verdict": a float between 0 and 1, "reason": '
        'str} where 1 = perfectly achieved, 0 = not achieved at all.\nTask: {task}\n'
        "Actual outcome: {outcome}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 task: Optional[str] = None, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.task = task
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        tools = _fmt_list(_tool_repr(t) for t in run.all_tool_calls)
        out = self.judge.complete_json(_judge_user(_f(
            self.TASK_AND_OUTCOME, input=run.question, tools=tools, actual=run.text)))
        task = self.task or (out or {}).get("task")
        outcome = (out or {}).get("outcome")
        if not task or not outcome:
            return CheckResult.error_(self.name, "task/outcome extraction failed")
        v = self.judge.complete_json(_judge_user(_f(self.TASK_VERDICT, task=task, outcome=outcome)))
        if not v or "verdict" not in v:
            return CheckResult.error_(self.name, "task verdict judge returned no score")
        score = float(v["verdict"])
        return _cr(self.name, score, self.threshold,
                   details={"task": task, "outcome": outcome, "reason": v.get("reason")})


__all__ = ["TaskCompletion"]
