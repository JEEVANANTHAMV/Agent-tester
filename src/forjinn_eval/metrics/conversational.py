"""Multi-turn / conversational metrics (deepeval ``Turn*`` family + ragas agent
conversational metrics).

Each operates on every (human, ai) exchange of an :class:`AgentRun` (the full
transcript via :attr:`AgentRun.conversation`), scoring the agent's reply at each
turn against a window of the preceding context, then averaging.

* :class:`TurnFaithfulness`        - is each reply grounded in the context seen up to
  that turn? (ragas/deepeval)
* :class:`TurnRelevancy`           - is each reply relevant to the question at that turn?
* :class:`TurnContextualPrecision` - of the contexts at that turn, what fraction is useful?
* :class:`TurnContextualRecall`    - do the contexts so far contain the reference's facts?
* :class:`TurnContextualRelevancy` - are the contexts at that turn on-topic?
* :class:`MultiTurnTopicAdherence` - did the agent stay on-topic across turns?
* :class:`MultiTurnToolUse`        - did the agent use its tools correctly across turns?
* :class:`GoalAccuracy`            - did the agent reach the goal by the end?
* :class:`ConversationalGEval`     - a GEval over the whole conversation.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default, as_int01
from ._common import (
    _LEVEL_ORDERS,
    _cr,
    _f,
    _fmt_list,
    _judge_user,
    _nan_or_skip,
    _scale,
    _trace_text,
    _verdict_list,
)
from ._conversational import context_for, render_history, to_turns, turns_of
from .base import CheckResult, Evaluator


def _avg(scores: List[float]) -> Optional[float]:
    return (sum(scores) / len(scores)) if scores else None


class _Windowed(Evaluator):
    """Base for the per-turn sliding-window metrics."""

    kind = "conversational"
    window_size: int = 10

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 window_size: int = 10, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        self.window_size = window_size
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        turns = turns_of(run)
        if not turns:
            return _nan_or_skip(self.name, "no turns")
        scores: List[float] = []
        for i, t in enumerate(turns):
            upto = i + 1  # turn index (1-based)
            score = self._score_one(run, t, upto)
            if score is not None:
                scores.append(score)
        if not scores:
            return _nan_or_skip(self.name, "no turn produced a score")
        return _cr(self.name, _avg(scores), self.threshold,
                   details={"per_turn": scores, "turns": len(scores)})

    def _score_one(self, run: AgentRun, turn, upto: int) -> Optional[float]:  # pragma: no cover
        raise NotImplementedError

    def _window(self, run: AgentRun, upto: int) -> str:
        return render_history(to_turns(run.conversation), upto)


class TurnFaithfulness(_Windowed):
    name = "turn_faithfulness"

    _PROMPT = (
        "For this turn, decide whether the AI reply is GROUNDED in the context available up to this "
        "turn. Omissions are allowed; penalise only ungrounded additions. Return JSON "
        "{\"faithful\": 0|1, \"reason\": str}.\nHistory up to this turn:\n{history}\n"
        "Question: {question}\nReply: {reply}\nJSON:"
    )

    def _score_one(self, run: AgentRun, turn, upto: int) -> Optional[float]:
        out = self.judge.complete_json(_judge_user(_f(
            self._PROMPT, history=self._window(run, upto) + "\n\nContext:\n" +
            "\n".join(run.retrieved_context()) + "\n\nCurrent:\nHuman: " + turn.human +
            "\nAI: " + turn.ai)))
        if not out:
            return None
        return 1.0 if as_int01(out.get("faithful")) else 0.0


class TurnRelevancy(_Windowed):
    name = "turn_relevancy"

    _PROMPT = (
        "Is the AI reply relevant to the question asked at this turn (using conversation history)? "
        "Return JSON {\"relevant\": 0|1, \"score\": int 0-10, \"reason\": str}.\n"
        "History:\n{history}\nCurrent:\nHuman: {question}\nAI: {reply}\nJSON:"
    )

    def _score_one(self, run: AgentRun, turn, upto: int) -> Optional[float]:
        out = self.judge.complete_json(_judge_user(_f(
            self._PROMPT, history=self._window(run, upto), question=turn.human, reply=turn.ai)))
        if not out:
            return None
        s = out.get("score")
        if isinstance(s, (int, float)):
            return max(0.0, min(10.0, float(s))) / 10.0
        return 1.0 if as_int01(out.get("relevant")) else 0.0


class TurnContextualPrecision(_Windowed):
    name = "turn_contextual_precision"

    _PROMPT = (
        "For the contexts retrieved at this turn, decide how many are USEFUL for producing the "
        "answer/relevant reference. Return JSON {{\"useful\": [0|1, ...], \"reason\": str}} "
        "one per context (in order).\nContexts at this turn:\n{ctx}\n"
        "Current question: {question}\nJSON:"
    )

    def _score_one(self, run: AgentRun, turn, upto: int) -> Optional[float]:
        ctx = run.retrieved_context()
        if not ctx:
            return None
        out = self.judge.complete_json(_judge_user(_f(
            self._PROMPT, ctx="\n".join(f"[{i}] {c}" for i, c in enumerate(ctx, 1)),
            question=turn.human)))
        verdicts = _verdict_list((out or {}).get("useful"))
        if not verdicts:
            return None
        return sum(verdicts) / len(verdicts)


class TurnContextualRecall(_Windowed):
    name = "turn_contextual_recall"

    _PROMPT = (
        "Given the reference, decide whether the contexts retrieved so far contain the facts needed "
        "to produce it. Return JSON {{\"covered\": [0|1, ...], \"reason\": str}}.\n"
        "Reference: {reference}\nContexts so far:\n{ctx}\nJSON:"
    )

    def __init__(self, *a, reference: Optional[str] = None, **kw):
        super().__init__(*a, **kw)
        self.reference = reference

    def _score_one(self, run: AgentRun, turn, upto: int) -> Optional[float]:
        ref = self.reference if self.reference is not None else run.reference
        ctx = run.retrieved_context()
        if not ref or not ctx:
            return None
        out = self.judge.complete_json(_judge_user(_f(
            self._PROMPT, reference=ref, ctx="\n".join(ctx))))
        verdicts = _verdict_list((out or {}).get("covered"))
        if not verdicts:
            return None
        return sum(verdicts) / len(verdicts)


class TurnContextualRelevancy(_Windowed):
    name = "turn_contextual_relevancy"

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 0.5,
                 window_size: int = 10, name: Optional[str] = None):
        super().__init__(judge=judge, threshold=threshold, window_size=window_size, name=name)
        from ..judge import Embeddings

        self.embeddings = Embeddings()

    def _score_one(self, run: AgentRun, turn, upto: int) -> Optional[float]:
        q = turn.human or run.question
        ctx = run.retrieved_context()
        if not q or not ctx:
            return None
        sims = [max(0.0, self.embeddings.similarity(q, c)) for c in ctx]
        return sum(sims) / len(sims)


class MultiTurnTopicAdherence(Evaluator):
    name = "multi_turn_topic_adherence"
    kind = "conversational"

    def __init__(self, reference_topics: Sequence[str], judge: Optional[JudgeClient] = None,
                 threshold: float = 0.5, name: Optional[str] = None):
        if not reference_topics:
            raise ValueError("reference_topics required")
        self.reference_topics = list(reference_topics)
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        conv = _trace_text(run)
        if not conv:
            return _nan_or_skip(self.name, "empty conversation")
        out = self.judge.complete_json(_judge_user(
            "Given the allowed topics and the conversation, decide whether the AI STAYED within "
            "the allowed topics for its replies (ignoring refusals). Return JSON "
            '{"on_topic": 0|1, "reason": str}.\nAllowed topics: '
            f"{_fmt_list(self.reference_topics)}\nConversation:\n{conv}\nJSON:"))
        if not out:
            return CheckResult.error_(self.name, "topic judge returned nothing")
        score = 1.0 if as_int01(out.get("on_topic")) else 0.0
        return _cr(self.name, score, self.threshold,
                   reason=str(out.get("reason", "")), details={"topics": self.reference_topics})


class MultiTurnToolUse(Evaluator):
    name = "multi_turn_tool_use"
    kind = "conversational"

    _PROMPT = (
        "Across the whole conversation, did the agent use its available tools CORRECTLY (right "
        "selection, right arguments, no misuse)? Return JSON "
        "{\"tool_selection\": 0|1, \"argument_correctness\": 0|1, \"reason\": str}. "
        "1 = correct.\nConversation:\n{conv}\nTools called: {tools}\nAvailable: {avail}\nJSON:"
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
            return _nan_or_skip(self.name, "no tool calls across the conversation")
        from ._common import _tool_repr

        out = self.judge.complete_json(_judge_user(_f(
            self._PROMPT, conv=_trace_text(run),
            tools=_fmt_list(map(_tool_repr, tools)),
            avail=_fmt_list(run.available_tool_names))))
        if not out:
            return CheckResult.error_(self.name, "tool judge returned nothing")
        sel = as_int01(out.get("tool_selection"))
        arg = as_int01(out.get("argument_correctness"))
        score = min(sel, arg)
        return _cr(self.name, score, self.threshold,
                   reason=str(out.get("reason", "")),
                   details={"tool_selection": sel, "argument_correctness": arg})


class GoalAccuracyMulti(Evaluator):
    """Multi-turn goal accuracy (does not shadow the single-turn ``GoalAccuracy``)."""

    name = "goal_accuracy_multi"
    kind = "conversational"

    _PROMPT = (
        "By the end of the conversation, did the agent achieve the user's GOAL? Use a 5-level "
        "scale: {levels}. Return JSON {{\"goal\": <level label>, \"reason\": str}}.\n"
        "Conversation:\n{conv}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, goal: Optional[str] = None,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.goal = goal
        self.threshold = threshold
        self.levels = _LEVEL_ORDERS["goal_accuracy"]
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        conv = _trace_text(run)
        if not conv:
            return _nan_or_skip(self.name, "empty conversation")
        if self.goal:
            head = f"User's GOAL: {self.goal}\n"
        else:
            head = ""
        out = self.judge.complete_json(_judge_user(
            self._PROMPT.format(levels=_fmt_list(self.levels), conv=conv).replace(
                "Conversation:", head + "Conversation:", 1)))
        verdict = (out or {}).get("goal")
        score = _scale(verdict, self.levels)
        if score is None:
            return CheckResult.error_(self.name, f"unrecognized verdict: {verdict!r}")
        return _cr(self.name, score, self.threshold,
                   reason=str((out or {}).get("reason", "")), details={"verdict": verdict})


class ConversationalGEval(Evaluator):
    """A GEval applied over the whole multi-turn conversation."""

    name = "conversational_g_eval"
    kind = "conversational"

    def __init__(self, judge: Optional[JudgeClient] = None, criteria: str = "",
                 threshold: float = 0.5, name: Optional[str] = None, strict_mode: bool = False):
        if not criteria:
            raise ValueError("ConversationalGEval requires criteria")
        self.criteria = criteria
        self.judge = _judge_or_default(judge)
        self.threshold = 1.0 if strict_mode else threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        conv = _trace_text(run)
        if not conv:
            return _nan_or_skip(self.name, "empty conversation")
        # generate steps then score
        gen = self.judge.complete_json(_judge_user(
            f"Given this evaluation criteria for a CONVERSATION, generate 3-4 evaluation steps. "
            f'Return JSON {{"steps": [str, ...]}}.\nCriteria: {self.criteria}\nJSON:'))
        steps = (gen or {}).get("steps", [])
        if not steps:
            return CheckResult.error_(self.name, "no steps generated")
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        out = self.judge.complete_json(_judge_user(
            "You are an evaluator of a multi-turn conversation. Given the steps, score it 0-10. "
            'Return JSON {"score": int 0-10, "reason": str}.\nSteps:\n'
            f"{numbered}\n\nConversation:\n{conv}\nJSON:"))
        if not out or "score" not in out:
            return CheckResult.error_(self.name, "judge returned no score")
        raw = float(out["score"])
        score = max(0.0, min(10.0, raw)) / 10.0
        return _cr(self.name, score, self.threshold,
                   reason=str(out.get("reason", "")), details={"raw": raw, "steps": steps})


__all__ = [
    "ConversationalGEval",
    "GoalAccuracyMulti",
    "MultiTurnToolUse",
    "MultiTurnTopicAdherence",
    "TurnContextualPrecision",
    "TurnContextualRecall",
    "TurnContextualRelevancy",
    "TurnFaithfulness",
    "TurnRelevancy",
    "context_for",
    "render_history",
    "to_turns",
    "turns_of",
]
