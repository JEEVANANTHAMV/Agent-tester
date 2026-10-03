"""deepeval ToolUse: LLM-judge tool *selection* + *argument* correctness.
final = min(selection, argument correctness); needs ``run.all_tool_calls`` and
(optionally) ``available_tools`` from the agent node. If no tools were called,
the metric SKIPs (nothing to judge)."""
from __future__ import annotations

from typing import List, Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from ._common import _cr, _f, _fmt_list, _judge_user, _nan_or_skip, _tool_repr
from .base import CheckResult, Evaluator


class ToolUse(Evaluator):
    name = "tool_use"
    kind = "tool"

    _SEL = (
        "Assess the TOOL SELECTION quality of an agent (which tools were chosen, not how used). "
        'Return JSON {"score": float 0-1, "reason": str}. 1.0 = every tool necessary and '
        "perfectly matched; 0 = unjustified. Be strict.\nUser input: {user}\nAssistant: {assistant}\n"
        "Tools called: {tools}\nAvailable tools: {available}\nJSON:"
    )
    _ARG = (
        "Assess the ARGUMENT/parameter quality of a tool call. Return JSON "
        '{"score": float 0-1, "reason": str}. 1.0 = all args accurate/specific/aligned.\n'
        "User input: {user}\nAssistant: {assistant}\nTool call: {tool}\nAvailable tools: {available}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        tools = run.all_tool_calls
        if not tools:
            return _nan_or_skip(self.name, "no tool calls to judge")
        user, assistant = run.question, run.text
        avail = run.available_tool_names
        sel_scores: List[float] = []
        out = self.judge.complete_json(_judge_user(_f(
            self._SEL, user=user, assistant=assistant,
            tools=_fmt_list(map(_tool_repr, tools)), available=_fmt_list(avail))))
        if out and isinstance(out.get("score"), (int, float)):
            sel_scores.append(max(0.0, min(1.0, float(out["score"]))))
        arg_scores: List[float] = []
        for t in tools:
            o = self.judge.complete_json(_judge_user(_f(
                self._ARG, user=user, assistant=assistant, tool=_tool_repr(t),
                available=_fmt_list(avail))))
            if o and isinstance(o.get("score"), (int, float)):
                arg_scores.append(max(0.0, min(1.0, float(o["score"]))))
        sel = sum(sel_scores) / len(sel_scores) if sel_scores else 1.0
        arg = sum(arg_scores) / len(arg_scores) if arg_scores else 0.0
        score = min(sel, arg)
        return _cr(self.name, score, self.threshold,
                   details={"tool_selection": sel, "argument_correctness": arg})


__all__ = ["ToolUse"]
