"""Tool-correctness LLM metrics (deepeval ``ToolCorrectness`` +
``ArgumentCorrectness``).

* :class:`ToolCorrectness`      - did the agent call the *right* tools compared
  to a reference set, with correct use (selection + argument quality)?
* :class:`ArgumentCorrectness`  - were the arguments passed to the called tools
  correct/appropriate, judged against the input?

Both are LLM-judged; they SKIP when no tool calls are present (use the
deterministic :mod:`tool_deterministic` metrics when you want a hard gate).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default, as_int01
from ._common import _cr, _f, _fmt_list, _judge_user, _nan_or_skip, _tool_repr
from .base import CheckResult, Evaluator


class ToolCorrectness(Evaluator):
    name = "tool_correctness"
    kind = "tool"

    _PROMPT = (
        "Assess TOOL CORRECTNESS of an agent: were the right tools selected AND used with correct "
        "arguments, relative to the reference tool calls? Return JSON "
        '{"selection": float 0-1, "argument": float 0-1, "reason": str}. '
        "1.0 = every tool necessary, correctly called with correct arguments.\n"
        "User input: {user}\nAssistant: {assistant}\nTools called: {tools}\n"
        "Expected tools: {expected}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, expected_tools: Optional[Sequence] = None,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.expected_tools = list(expected_tools or [])
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        tools = run.all_tool_calls
        if not tools:
            return _nan_or_skip(self.name, "no tool calls to assess")
        if self.expected_tools:
            exp = [str(e) for e in self.expected_tools]
        elif run.reference_tool_calls:
            exp = [t.name for t in run.reference_tool_calls]
        else:
            exp = ["(none)"]
        out = self.judge.complete_json(_judge_user(_f(
            self._PROMPT, user=run.question, assistant=run.text,
            tools=_fmt_list(map(_tool_repr, tools)), expected=_fmt_list(exp))))
        if not out:
            return CheckResult.error_(self.name, "tool judge returned nothing")
        sel = _clamp(out.get("selection"))
        arg = _clamp(out.get("argument"))
        score = min(sel, arg) if (sel is not None and arg is not None) else (sel or arg or 0.0)
        return _cr(self.name, score, self.threshold,
                   reason=str(out.get("reason", "")),
                   details={"selection": sel, "argument": arg})


class ArgumentCorrectness(Evaluator):
    name = "argument_correctness"
    kind = "tool"

    _PROMPT = (
        "For each tool call, judge whether its ARGUMENTS are correct and appropriate given the "
        "user input. Return JSON {{\"judgments\": [{{\"name\": str, \"correct\": 0|1, \"reason\": "
        "str}}, ...]}}.\nUser input: {user}\nTools called: {tools}\nJSON:"
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
            return _nan_or_skip(self.name, "no tool calls to assess")
        prompt = self._PROMPT.format(user=run.question, tools=_fmt_list(map(_tool_repr, tools)))
        out = self.judge.complete_json(_judge_user(prompt))
        if not out:
            return CheckResult.error_(self.name, "argument judge returned nothing")
        raw_judg = (out or {}).get("judgments", [])
        if not isinstance(raw_judg, (list, tuple)):
            raw_judg = [raw_judg] if raw_judg else []
        # normalise entries to dicts with a name + correct field
        judg = []
        for idx, j in enumerate(raw_judg):
            if isinstance(j, dict):
                judg.append(j)
            else:
                # scalar/default entry -> assume correct, name = the idx-th tool
                nm = tools[idx].name if idx < len(tools) else "unknown"
                judg.append({"name": nm, "correct": 1})
        if not judg:
            return CheckResult.error_(self.name, "no per-call judgments")
        # map per-call by name (absent -> assume correct)
        per_call = []
        for t in tools:
            for j in judg:
                if j.get("name") == t.name:
                    per_call.append(as_int01(j.get("correct", 1)))
                    break
            else:
                per_call.append(1)
        score = sum(per_call) / len(per_call) if per_call else 0.0
        return _cr(self.name, score, self.threshold, details={"per_call": per_call})


def _clamp(x):
    if x is None:
        return None
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return None


def _to01(x):
    from ..judge import as_int01

    return as_int01(x)


__all__ = ["ArgumentCorrectness", "ToolCorrectness"]
