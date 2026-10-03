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
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

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
# JSON shapes the individual metrics request. :func:`_offline_responder_shape`
# builds a shape-specific response on top of this so every metric runs offline.
_OFFLINE_DEFAULTS: Dict[str, Any] = {
    "score": 9,
    "reason": "offline judge (permissive)",
    "passed": True,
    "verdict": "yes",
}


def _offline_responder_shape(question: str) -> Dict[str, Any]:
    """Build a permissive offline judge response that fits the requested shape.

    The Forjinn judge is asked for JSON in many shapes (a 0-10 score, a list of
    sentence verdicts, an entity table, a rating, ...). The offline responder
    scans the rubric for every JSON key it names and returns each with a
    favourable value - so the *entire* metric catalog executes with no network.
    List-valued keys are sized to the number of numbered items in the rubric.
    """
    import re as _re

    def _n_list_items() -> int:
        nums = _re.findall(r"^\s*(\d+)\.", question, _re.MULTILINE)
        return max(1, min(len(nums), 8)) if nums else 3

    def _first_key() -> str:
        m = _re.search(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', question)
        return m.group(1) if m else "score"

    n = _n_list_items()
    first = _first_key()

    # Collect every key the rubric names, whether quoted ({"key": ...}) or bare
    # (Return JSON "key": [..] / "useful": [0|1, ...]).
    named_keys: List[str] = []
    for m in _re.finditer(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', question):
        k = m.group(1)
        if k not in named_keys:
            named_keys.append(k)
    for m in _re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)"\s*:\s*(\[|\{|\d|"|true|false|0\|1)', question):
        k = m.group(1)
        if k not in named_keys:
            named_keys.append(k)

    def table() -> Dict[str, Any]:
        # response-key -> favourable value; build lazily so `n` is captured once.
        return {
            "score": 9, "raw_score": 9, "score_0to10": 9, "rating": 1, "score_1to5": 5,
            "statements": [{"statement": "s", "reason": "ok", "verdict": 1} for _ in range(n)],
            "verdicts": [{"statement": "s", "verdict": 1, "reason": "ok"}] * n,
            "claims": [f"claim{i}" for i in range(1, n + 1)],
            "questions": [f"q{i}" for i in range(1, n + 1)],
            "classifications": [True] * n,
            "opinions": [],
            "extracted_pii": [],
            "topics": [],
            "refused_to_answer": False,
            "steps": ["offline step 1", "offline step 2"],
            "task": "offline task",
            "outcome": "offline outcome",
            "user_goal": "offline goal",
            "end_state": "offline end state",
            "relevant": [1] * n,
            "supported": [1] * n,
            "recall_verdicts": [{"entity": f"e{i}", "in_context": 1} for i in range(1, n + 1)],
            "reference_entities": [f"e{i}" for i in range(1, n + 1)],
            "context_entities": [f"e{i}" for i in range(1, n + 1)],
            "hallucinations": [],
            "misuse": False,
            "advice_given": False,
            "violations": [],
            "adhered": True,
            "quality_verdict": "Good",
            "correctness": True,
            "argument_correct": True,
            "plan": ["step 1", "step 2"],
            "adherence": "Mostly adhered",
            "plan_quality": "Good",
            "efficiency": "Mostly efficient",
            "intentions": ["intent"],
            "intents_addressed": [True] * n,
            "knowledge_items": ["k1", "k2"],
            "knowledge_used": [True, True],
            "tool_selection": True,
            "faithful": True,
            "in_context": 1,
            "advice_requested": False,
            "appropriately_handled": True,
            "citation_verdicts": [{"cite": i, "faithful": 1, "reason": "ok"} for i in range(1, n + 1)],
            # per-item verdict lists -> default to a favourables list sized to n
            "useful": [1] * n,
            "covered": [1] * n,
            "correct": [1] * n,
        }

    tbl = table()
    shape: Dict[str, Any] = dict(_OFFLINE_DEFAULTS)
    # first named key is the primary one the metric reads; always include it.
    shape[first] = tbl.get(first, 9)
    # include every other named key; unknown keys default to a favourables list
    # (per-item verdict shape) so iterating metrics never break.
    for key in named_keys:
        if key in shape:
            continue
        shape[key] = tbl.get(key, [1] * n)
    return shape


def _offline_responder(question: str) -> Any:
    """The canonical offline judge: permissive + shape-aware, no network."""
    return _offline_responder_shape(question)


def _nan_or_skip(name: str, why: str) -> CheckResult:
    return CheckResult.skip_(name, why)


def _clamp01(x: Optional[float]) -> Optional[float]:
    """Clamp a numeric score into [0, 1] (``None``/non-float pass through as None)."""
    if x is None:
        return None
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return None


def _cr(name: str, score: Optional[float], threshold: float,
        details: Optional[Dict[str, Any]] = None, reason: str = "") -> CheckResult:
    """Build a PASS/FAIL CheckResult from a numeric score (or a nan SKIP)."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return CheckResult.skip_(name, "no score produced", details=details or {})
    details = dict(details or {})
    score = _clamp01(score)
    if score is None:
        return CheckResult.skip_(name, "no numeric score produced", details=details)
    score = round(score, 4)
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


def _fmt(template: str, **fields: Any) -> str:
    """Substitute only the named ``{field}`` placeholders, leaving ALL other braces
    (literal JSON examples) untouched. Use this for multi-brace rubric templates
    that must keep their example JSON verbatim.
    """

    out = template
    for k, v in fields.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _fbeta(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


_LEVEL_ORDERS: Dict[str, List[str]] = {
    "task_completion": ["Not achieved", "Partly achieved", "Mostly achieved", "Fully achieved"],
    "plan_adherence": ["No adherence", "Poor adherence", "Partial adherence", "Mostly adhered", "Perfect adherence"],
    "plan_quality": ["Inadequate plan", "Weak plan", "Adequate plan", "Good plan", "Excellent plan"],
    "step_efficiency": ["Very inefficient", "Inefficient", "Fairly efficient", "Efficient", "Very efficient"],
    "goal_accuracy": ["Not achieved", "Partly achieved", "Mostly achieved", "Fully achieved", "Perfectly achieved"],
    "conversation_completeness": ["Very incomplete", "Incomplete", "Partially complete", "Mostly complete", "Complete"],
}


def _scale(verdict: Any, levels: Sequence[str]) -> Optional[float]:
    """Map a judge rubric verdict (label string or int 1..N) to a 0..1 score.

    ``levels`` is ordered worst->best. A matching label (case-insensitive,
    substring) maps to its index/(N-1); a numeric value k maps to k/(N) for a
    1-indexed scale. Returns ``None`` if the verdict is unrecognised.
    """
    if verdict is None:
        return None
    n = len(levels)
    if isinstance(verdict, (int, float)):
        v = int(round(float(verdict)))
        # treat as 1..n
        if 1 <= v <= n:
            return (v - 1) / (n - 1) if n > 1 else 1.0
        return (max(0.0, min(1.0, float(verdict) / n)))
    s = str(verdict).strip().lower()
    for i, lab in enumerate(levels):
        if lab.lower() in s or s in lab.lower():
            return i / (n - 1) if n > 1 else 1.0
    # numeric label like "4"
    try:
        v = int(s)
        if 1 <= v <= n:
            return (v - 1) / (n - 1) if n > 1 else 1.0
    except ValueError:
        pass
    return None


def _verdict_list(raw: Any, key: Optional[str] = None) -> List[int]:
    """Coerce a judge's per-item verdict field (list of int / dict / str) to 0/1.

    Canonical helper for every metric that iterates a per-item list (``verdicts``,
    ``useful``, ``covered``, ``relevant``, ``supported``, ...). ``key`` selects a
    nested field on dict entries (e.g. ``"verdict"``); without it each entry is
    coerced directly. Tolerates the offline shape-aware responder's permissive
    values and a single scalar.
    """
    from ..judge import as_int01

    def one(v):
        if key is not None and isinstance(v, dict):
            return as_int01(v.get(key, v))
        return as_int01(v)

    if raw is None:
        return []
    if isinstance(raw, (int, float, str, bool)):
        return [one(raw)]
    if not isinstance(raw, (list, tuple)):
        return [one(raw)]
    return [one(v) for v in raw]


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
    "NAN",
    "_LEVEL_ORDERS",
    "_OFFLINE_DEFAULTS",
    "CheckResult",
    "Evaluator",
    "JudgeClient",
    "Status",
    "_cr",
    "_f",
    "_fbeta",
    "_fmt",
    "_fmt_list",
    "_judge_or_default",
    "_judge_user",
    "_nan_or_skip",
    "_norm_list",
    "_offline_responder",
    "_offline_responder_shape",
    "_scale",
    "_string_similarity",
    "_tool_repr",
    "_trace_text",
    "_verdict_list",
    "as_bool",
    "as_int01",
    "extract_json",
]
