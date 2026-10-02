"""Agent plan / efficiency metrics (deepeval PlanAdherence, PlanQuality,
StepEfficiency; ragas ConversationCompleteness, KnowledgeRetention,
TopicAdherence multi-turn).

* :class:`PlanAdherence`    - did the agent follow its stated plan?
* :class:`PlanQuality`      - is the agent's plan itself good?
* :class:`StepEfficiency`   - how many steps did it take vs the minimum?
* :class:`ConversationCompleteness` - were all user intentions addressed
  (multi-turn)?
* :class:`KnowledgeRetention`      - did the agent use knowledge the user
  provided across turns (multi-turn)?
"""
from __future__ import annotations

from typing import List, Optional

from ..capture import AgentRun
from ..judge import JudgeClient, as_int01, _judge_or_default
from ._common import (
    _LEVEL_ORDERS,
    _cr,
    _f,
    _fmt,
    _fmt_list,
    _judge_user,
    _nan_or_skip,
    _scale,
    _trace_text,
    _verdict_list,
)
from .base import CheckResult, Evaluator


class PlanAdherence(Evaluator):
    name = "plan_adherence"
    kind = "agent"

    _GEN = (
        "Extract the agent's PLAN (the ordered steps it set out to follow) from the interaction. "
        'Return JSON {"plan": [str, ...]}.\nInteraction:\n{interaction}\nJSON:'
    )
    _SCORE = (
        "Given the agent's plan and how it actually behaved, judge HOW WELL it adhered. Use a "
        "5-level scale: {levels}. Return JSON {\"adherence\": <level label>, \"reason\": str}.\n"
        "Plan:\n{plan}\nInteraction:\n{interaction}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.levels = _LEVEL_ORDERS["plan_adherence"]
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        interaction = _trace_text(run)
        if not run.text:
            return _nan_or_skip(self.name, "no agent output")
        gen = self.judge.complete_json(_judge_user(_f(self._GEN, interaction=interaction)))
        plan = (gen or {}).get("plan", [])
        if not plan:
            return CheckResult.error_(self.name, "no plan extracted")
        out = self.judge.complete_json(_judge_user(_fmt(
            self._SCORE, levels=_fmt_list(self.levels), plan=_fmt_list(plan),
            interaction=interaction)))
        verdict = (out or {}).get("adherence")
        score = _scale(verdict, self.levels)
        if score is None:
            return CheckResult.error_(self.name, f"unrecognized adherence verdict: {verdict!r}")
        return _cr(self.name, score, self.threshold,
                   reason=str((out or {}).get("reason", "")),
                   details={"plan": plan, "verdict": verdict})


class PlanQuality(Evaluator):
    name = "plan_quality"
    kind = "agent"

    _PROMPT = (
        "Assess the QUALITY of the agent's plan (is it a sensible, complete, well-ordered set of "
        "steps for the task?). Use a 5-level scale: {levels}. Return JSON "
        "{\"plan_quality\": <level label>, \"reason\": str}.\nTask/interaction:\n{interaction}\n"
        "Plan:\n{plan}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.levels = _LEVEL_ORDERS["plan_quality"]
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        interaction = _trace_text(run)
        gen = self.judge.complete_json(_judge_user(
            "Extract the agent's plan (ordered steps). Return JSON {\"plan\": [str, ...]}.\n"
            f"Interaction:\n{interaction}\nJSON:"))
        plan = (gen or {}).get("plan", [])
        if not plan:
            return CheckResult.error_(self.name, "no plan extracted")
        out = self.judge.complete_json(_judge_user(_fmt(
            self._PROMPT, levels=_fmt_list(self.levels), interaction=_trace_text(run),
            plan=_fmt_list(plan))))
        verdict = (out or {}).get("plan_quality")
        score = _scale(verdict, self.levels)
        if score is None:
            return CheckResult.error_(self.name, f"unrecognized verdict: {verdict!r}")
        return _cr(self.name, score, self.threshold,
                   reason=str((out or {}).get("reason", "")), details={"plan": plan})


class StepEfficiency(Evaluator):
    name = "step_efficiency"
    kind = "agent"

    _PROMPT = (
        "How EFFICIENTLY did the agent accomplish the task given the number of tool calls / steps "
        "taken? Use a 5-level scale: {levels}. Return JSON "
        "{\"efficiency\": <level label>, \"reason\": str}.\nInteraction:\n{interaction}\n"
        "Steps: {steps}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.levels = _LEVEL_ORDERS["step_efficiency"]
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not run.text:
            return _nan_or_skip(self.name, "no agent output")
        steps = _describe_steps(run)
        out = self.judge.complete_json(_judge_user(_fmt(
            self._PROMPT, levels=_fmt_list(self.levels), interaction=_trace_text(run),
            steps=steps)))
        verdict = (out or {}).get("efficiency")
        score = _scale(verdict, self.levels)
        if score is None:
            return CheckResult.error_(self.name, f"unrecognized verdict: {verdict!r}")
        return _cr(self.name, score, self.threshold,
                   reason=str((out or {}).get("reason", "")), details={"steps": steps})


class ConversationCompleteness(Evaluator):
    name = "conversation_completeness"
    kind = "agent"

    _INTENT = (
        "Extract the user's INTENTIONS/goals from the conversation. Return JSON "
        "{\"intentions\": [str, ...]}.\nConversation:\n{conversation}\nJSON:"
    )
    _ADDRESSED = (
        "For each user intention, decide whether the conversation FULLY addressed it by the end. "
        'Return JSON {"addressed": [0|1, ...]} one per intention, in order.\n'
        "Conversastion:\n{conversation}\nIntentions:\n{intentions}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.levels = _LEVEL_ORDERS["conversation_completeness"]
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        conv = _trace_text(run)
        if not conv:
            return _nan_or_skip(self.name, "empty conversation")
        it = self.judge.complete_json(_judge_user(_f(self._INTENT, conversation=conv)))
        intentions = (it or {}).get("intentions", [])
        if not intentions:
            return CheckResult.error_(self.name, "no intentions extracted")
        ad = self.judge.complete_json(_judge_user(_f(
            self._ADDRESSED, conversation=conv,
            intentions="\n".join(f"{i}. {x}" for i, x in enumerate(intentions, 1)))))
        verdicts = _verdict_list((ad or {}).get("addressed"))
        if not verdicts:
            return CheckResult.error_(self.name, "no addressed verdicts")
        score = sum(verdicts) / len(verdicts)
        return _cr(self.name, score, self.threshold,
                   details={"intentions": intentions, "addressed": verdicts})


class KnowledgeRetention(Evaluator):
    name = "knowledge_retention"
    kind = "agent"

    _ITEMS = (
        "Extract facts/knowledge the USER provided to the agent across the conversation (things the "
        "agent should remember and reuse). Return JSON {\"knowledge_items\": [str, ...]}. If none, "
        "return [].\nConversation:\n{conversation}\nJSON:"
    )
    _USED = (
        "For each item the user provided, decide whether the agent USED it in a later response. "
        'Return JSON {"used": [0|1, ...]} one per item, in order.\n'
        "Conversation:\n{conversation}\nItems:\n{items}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        conv = _trace_text(run)
        if not conv:
            return _nan_or_skip(self.name, "empty conversation")
        items = self.judge.complete_json(_judge_user(_f(self._ITEMS, conversation=conv)))
        knowledge = (items or {}).get("knowledge_items", [])
        if not knowledge:
            return CheckResult.skip_(self.name, "no user-provided knowledge to retain")
        used = self.judge.complete_json(_judge_user(_f(
            self._USED, conversation=conv,
            items="\n".join(f"{i}. {x}" for i, x in enumerate(knowledge, 1)))))
        verdicts = _verdict_list((used or {}).get("used"))
        if not verdicts:
            return CheckResult.error_(self.name, "no used verdicts")
        score = sum(verdicts) / len(verdicts)
        return _cr(self.name, score, self.threshold,
                   details={"items": knowledge, "used": verdicts})


class RoleAdherence(Evaluator):
    name = "role_adherence"
    kind = "conversational"

    def __init__(self, role: str, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        if not role:
            raise ValueError("RoleAdherence requires a role")
        self.role = role
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        conv = _trace_text(run)
        if not conv:
            return _nan_or_skip(self.name, "empty conversation")
        from ..judge import as_int01

        prompt = (
            'Across the conversation, did the agent STAY in the role "' + self.role + '" (no '
            'out-of-character or off-role behaviour)? Return JSON {"adhering": 0|1, '
            '"reason": str}.\nConversation:\n' + conv + "\nJSON:"
        )
        out = self.judge.complete_json(_judge_user(prompt))
        if not out:
            return CheckResult.error_(self.name, "role judge returned nothing")
        score = 1.0 if as_int01(out.get("adhering")) else 0.0
        return _cr(self.name, score, self.threshold,
                   reason=str(out.get("reason", "")), details={"role": self.role})


def _describe_steps(run: AgentRun) -> str:
    parts: List[str] = []
    if run.question:
        parts.append(f"1. User asked: {run.question}")
    for i, t in enumerate(run.all_tool_calls, start=2):
        from ._common import _tool_repr

        parts.append(f"{i}. Tool: {_tool_repr(t)}")
    if run.text:
        parts.append(f"{len(parts) + 1}. Answer: {run.text[:300]}")
    return "\n".join(parts) if parts else "(none)"


__all__ = [
    "PlanAdherence",
    "PlanQuality",
    "StepEfficiency",
    "ConversationCompleteness",
    "KnowledgeRetention",
    "RoleAdherence",
]