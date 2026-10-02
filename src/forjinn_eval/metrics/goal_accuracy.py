"""ragas AgentGoalAccuracy: infer goal+end_state, judge 0/1. With
``desired_outcome`` it compares to that; else goal-vs-end_state."""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, as_int01, _judge_or_default
from .base import CheckResult, Evaluator
from ._common import _cr, _f, _judge_user, _trace_text


class GoalAccuracy(Evaluator):
    name = "goal_accuracy"
    kind = "goal"

    INFER_GOAL = (
        "Given an agentic workflow (Human, AI and Tools), identify the user_goal (the task/objective the "
        "user wants to achieve) and the end_state (the final outcome/result of the workflow). Return JSON "
        '{"user_goal": str, "end_state": str}.\nWorkflow:\n{workflow}\nJSON:'
    )

    COMPARE_OUTCOME = (
        "Given the desired outcome and the actual achieved outcome, compare them and identify if they are "
        'the same (1) or different (0). Return JSON {"reason": str, "verdict": "0" or "1"}.\n'
        "Desired outcome: {desired}\nAchieved outcome: {achieved}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, desired_outcome: Optional[str] = None,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.desired_outcome = desired_outcome
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        o = self.judge.complete_json(_judge_user(_f(self.INFER_GOAL, workflow=_trace_text(run))))
        if not o:
            return CheckResult.error_(self.name, "goal inference failed")
        desired = self.desired_outcome or o.get("user_goal")
        achieved = o.get("end_state")
        if not desired or not achieved:
            return CheckResult.error_(self.name, "goal/end_state inference failed")
        v = self.judge.complete_json(_judge_user(_f(self.COMPARE_OUTCOME, desired=desired, achieved=achieved)))
        if not v or "verdict" not in v:
            return CheckResult.error_(self.name, "goal comparison failed")
        score = float(as_int01(v.get("verdict")))
        return _cr(self.name, score, self.threshold,
                   details={"goal": desired, "end_state": achieved, "reason": v.get("reason")})


__all__ = ["GoalAccuracy"]
