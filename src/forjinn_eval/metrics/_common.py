"""Shared plumbing for the LLM-judged metric catalog.

Every LLM metric in :mod:`forjinn_eval.metrics` is "the atom" (RAGAS / DeepEval
style): one tiny class, a declared ``name``/``kind``, and a single
``evaluate(run) -> CheckResult``. The judge is a **Forjinn chatflow** (see
:mod:`forjinn_eval.judge`); a metric sends its fully-rendered rubric as the
Forjinn ``question`` and parses the JSON it gets back.

This module holds the small helpers every metric needs:

* :func:`_cr`            - build a PASS/FAIL/SKIP CheckResult from a 0..1 score.
* :func:`_nan_or_skip`   - the "no input available" (SKIP) shortcut.
* :func:`_judge_user`    - the rubric *is* the question (Forjinn chat agent).
* :func:`_f`             - ``str.format`` for prompts that also contain literal
  JSON example braces (escapes them first).
* :func:`_fbeta`         - F-beta score.
* :func:`_norm_list`     - normalize a judge field to a list.
* :func:`_string_similarity` / similarity helpers.
* :func:`_trace_text`    - render a human+agent+tool trace for prompt context.
* :func:`_tool_repr`     - one tool call as ``name(arguments=...)``.
* :data:`_OFFLINE_DEFAULTS` - the permissive verdicts the offline MockJudge emits.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Optional

from ..capture import AgentRun
from ..judge import (
    JudgeClient,
    _judge_or_default,
    as_bool,
    as_int01,
    extract_json,
)
from ..results import CheckResult, Status
from .base import Evaluator

NAN = float("nan")

# Permissive "as-if-passing" verdict defaults returned by the offline MockJudge
# (FORJINN_OFFLINE) so the whole LLM catalog runs with no network. Keys cover the
# JSON shapes the individual metrics request.
_OFFLINE_DEFAULTS: Dict[str, Any] = {
    "score": 9,
    "raw_score": 9,
    "score_0to10": 9,
    "reason": "offline judge (permissive)",
    "passed": True,
    "verdict": "yes",
    "statements": [{"statement": "s", "reason": "ok", "verdict": 1}],
    "claims": ["claim"],
    "questions": [1],
    "classifications": [True],
    "opinions": [],
    "extracted_pii": [],
    "topics": [],
    "refused_to_answer": False,
    "steps": ["offline step 1", "offline step 2"],
    "task": "offline task",
    "outcome": "offline outcome",
    "user_goal": "offline goal",
    "end_state": "offline end state",
}


def _nan_or_skip(name: str, why: str) -> CheckResult:
    return CheckResult.skip_(name, why)


def _cr(name: str, score: Optional[float], threshold: float,
        details: Optional[Dict[str, Any]] = None, reason: str = "") -> CheckResult:
    """Build a PASS/FAIL CheckResult from a numeric score (or a nan SKIP)."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return CheckResult.skip_(name, "no score produced", details=details or {})
    details = dict(details or {})
    score = round(float(score), 4)
    details.setdefault("score", score)
    if score >= threshold:
        return CheckResult.pass_(name, reason or f"score={score:.3f} >= {threshold}", score=score, details=details)
    return CheckResult.fail_(name, reason or f"score={score:.3f} < {threshold}", score=score, details=details)


def _judge_user(content: str) -> str:
    """The judge is a Forjinn chat agent: the fully-rendered rubric *is* the
    question we send (the judge's system prompt lives in its canvas)."""
    return content


def _f(template: str, **fields: Any) -> str:
    """``str.format`` for prompt templates that contain BOTH real ``{placeholder}``
    fields and literal JSON example braces (e.g. ``{"verdict": "yes"}``). Literal
    braces are escaped automatically; real fields are then substituted."""
    escaped = template.replace("{", "{{").replace("}", "}}")
    return escaped.format(**fields)


def _fbeta(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def _norm_list(items) -> List[Any]:
    if isinstance(items, str):
        return [items]
    if isinstance(items, (list, tuple)):
        return list(items)
    return []


def _fmt_list(xs: Sequence[str]) -> str:
    return ", ".join(str(x) for x in xs)


def _fbeta_tpfpfn(tp: int, fp: int, fn: int) -> float:
    return _fbeta(tp, fp, fn, 1.0)


def _string_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein via rapidfuzz when available, else a Jaccard
    bigram fallback (always in [0,1])."""
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore

        return 1.0 - Levenshtein.normalized_distance(a, b)
    except Exception:
        a, b = (a or "").lower(), (b or "").lower()
        ga = {a[i:i + 2] for i in range(len(a) - 1)}
        gb = {b[i:i + 2] for i in range(len(b) - 1)}
        if not ga and not gb:
            return 1.0
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / len(ga | gb)


def _trace_text(run: AgentRun) -> str:
    parts: List[str] = []
    start = run.nodes[0] if run.nodes else None
    if start is not None and start.is_start:
        parts.append(f"Human: {run.question}")
    a = run.final_agent_node
    if a is not None:
        if a.messages:
            for m in a.messages:
                r = (m.get("role") or "").lower()
                if r == "system":
                    continue
                prefix = "Human" if r == "user" else "AI"
                parts.append(f"{prefix}: {m.get('content','')}")
        tool_bits = [f"AI tool: {_tool_repr(t)}" for t in a.called_tools]
        parts.extend(tool_bits)
        if a.content:
            parts.append(f"AI: {a.content}")
    if not parts:
        parts.append(f"AI: {run.text}")
    return "\n".join(parts)


def _tool_repr(tc) -> str:
    return f"{tc.name}(arguments={tc.arguments})"


__all__ = [
    "Evaluator",
    "CheckResult",
    "Status",
    "JudgeClient",
    "as_bool",
    "as_int01",
    "extract_json",
    "_judge_or_default",
    "_cr",
    "_nan_or_skip",
    "_judge_user",
    "_f",
    "_fbeta",
    "_norm_list",
    "_fmt_list",
    "_string_similarity",
    "_trace_text",
    "_tool_repr",
    "_OFFLINE_DEFAULTS",
    "NAN",
]
